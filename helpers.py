import cv2
import math
import random
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from scipy.ndimage.morphology import distance_transform_edt

########################################[ General ]########################################

def show_anno_points(gt, pos_points, neg_points):
    gt_point_map=(gt==1).astype(np.uint8)
    for pt in pos_points: gt_point_map[pt[1], pt[0]] = 2
    for pt in neg_points: gt_point_map[pt[1], pt[0]] = 3
    return gt_point_map

def get_points_mask(size, points=np.empty([0,2],dtype=np.int64)):
    mask=np.zeros(size[::-1]).astype(np.uint8)
    mask[points[:,1], points[:,0]]=1
    return mask

def fill_up_list(arr,num=None,value=-1):
    if num is not None:
        if not isinstance(value,(list,np.ndarray)): value=[value]*(len(arr[0]) if len(arr)>0 else 3)
        arr=np.concatenate([arr,np.array([value]).repeat(num-len(arr),axis=0)],axis=0) if len(arr)<num else arr[:num]
    return arr

def get_points_list(mask,if_shuffle=False):
    points_list=np.argwhere(mask==1)[:,::-1] if mask.ndim==2 else np.argwhere(mask==1)[:,[1,0,2]]
    if if_shuffle:points_list=np.random.permutation(points_list)
    return points_list

def encode_mask(mask):
    return coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))

def decode_mask(encoded_info):
    return coco_mask.decode(encoded_info)

#rect=(x,y,w,h)    relax (LTRB) m->(m,m,m,m), (x,y)->(x,y,x,y), (l,t,r,b)
def expand_rect(rect,relax):
    if not(isinstance(relax,list) or isinstance(relax,tuple)):relax=[relax]
    relax=relax*(4//len(relax))
    pt_lt=np.array( [rect[0],rect[1]] )
    pt_rb_plus =np.array( [rect[0]+rect[2],rect[1]+rect[3]])
    pt_lt-=np.array(relax[:2])
    pt_rb_plus+=np.array(relax[2:])
    rect=( pt_lt[0] , pt_lt[1] ,  pt_rb_plus[0]-pt_lt[0] , pt_rb_plus[1]-pt_lt[1] )
    return rect

#rect=(x,y,w,h)  shape=(h,w)
def crop_rect_in_shape(rect,shape):
    pt_lt=np.array( [rect[0],rect[1]] )
    pt_rb_plus =np.array( [rect[0]+rect[2],rect[1]+rect[3]])
    pt_lt=np.maximum(pt_lt, np.array([0,0]))
    pt_lt=np.minimum(pt_lt, np.array([shape[1],shape[0]]))
    pt_rb_plus=np.maximum(pt_rb_plus, np.array([0,0]))
    pt_rb_plus=np.minimum(pt_rb_plus, np.array([shape[1],shape[0]]))
    rect=( pt_lt[0] , pt_lt[1] ,  pt_rb_plus[0]-pt_lt[0] , pt_rb_plus[1]-pt_lt[1] )
    return rect

def mask_from_bbox(bbox,shape):
    mask=np.zeros(shape).astype(np.uint8)
    mask[bbox[1]:bbox[1]+bbox[3] ,bbox[0]:bbox[0]+bbox[2]  ]=1
    return mask

def recover_from_bbox(result,bbox,shape):
    result=cv2.resize(result,tuple(bbox[2:]),interpolation=cv2.INTER_LINEAR)
    result_new=np.zeros(shape,dtype=result.dtype)
    result_new[ bbox[1]: bbox[1]+bbox[3], bbox[0]: bbox[0]+bbox[2] ]=result
    return result_new

def img_rotate(img, angle, center=None, if_expand=False, scale=1.0, mode=None):
    (h, w) = img.shape[:2]
    if center is None: center = (w // 2 ,h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    if mode is None: mode=cv2.INTER_LINEAR if len(img.shape)==3 else cv2.INTER_NEAREST
    if if_expand:
        h_new=int(w*math.fabs(math.sin(math.radians(angle)))+h*math.fabs(math.cos(math.radians(angle))))
        w_new=int(h*math.fabs(math.sin(math.radians(angle)))+w*math.fabs(math.cos(math.radians(angle)))) 
        M[0,2] +=(w_new-w)/2 
        M[1,2] +=(h_new-h)/2 
        h, w =h_new, w_new  
    rotated = cv2.warpAffine(img, M, (w, h),flags=mode)
    return rotated

def img_rotate_point(img, angle, center=None, if_expand=False, scale=1.0):
    (h, w) = img.shape[:2]
    if center is None: center = (w // 2 ,h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    if if_expand:
        h_new=int(w*math.fabs(math.sin(math.radians(angle)))+h*math.fabs(math.cos(math.radians(angle))))
        w_new=int(h*math.fabs(math.sin(math.radians(angle)))+w*math.fabs(math.cos(math.radians(angle)))) 
        M[0,2] +=(w_new-w)/2 
        M[1,2] +=(h_new-h)/2 
        h, w =h_new, w_new  

    pts_xy=np.argwhere(img==1)[:,::-1]
    pts_xy_new= np.rint(np.dot( np.insert(pts_xy,2,1,axis=1), M.T)).astype(np.int64)
    img_new=np.zeros((h,w),dtype=np.uint8)
    img_new[pts_xy_new[:,1],pts_xy_new[:,0]]=1
    return img_new

def crop_points(seq_pts,crop_bbox):
    seq_pts=seq_pts.copy()
    ignore_mask=(seq_pts<0).any(axis=1)
    seq_pts[~ignore_mask,:2]-=crop_bbox[:2]
    seq_pts=seq_pts[((seq_pts[:,:2]>=0) & (seq_pts[:,:2]<crop_bbox[2:])).all(axis=1) | ignore_mask]
    return seq_pts

#seq_pts_or_pt_mask [Nx3,Nx2,HxW,HxWxN]
def resize_points(seq_pts_or_pt_mask,dsize,ssize=None):
    def trans(pts_src,ssize,dsize): return  np.column_stack((np.int64((pts_src[:,:2]+0.5)*np.array(dsize)/np.array(ssize)),pts_src[:,2:]))
    if isinstance(dsize,int): dsize=(dsize,dsize)
    if ssize is None:
        ex=[1,0] if seq_pts_or_pt_mask.ndim==2 else [1,0,2]
        pts_src,ssize=np.argwhere(seq_pts_or_pt_mask!=0)[:,ex], seq_pts_or_pt_mask.shape[:2][::-1]
        pts_dst=trans(pts_src,ssize,dsize)
        pts_dst_mask=np.zeros(tuple(dsize[::-1])+seq_pts_or_pt_mask.shape[2:],dtype=seq_pts_or_pt_mask.dtype)
        pts_dst_mask[tuple(pts_dst[:,ex].T)]= seq_pts_or_pt_mask[tuple(pts_src[:,ex].T)]
        return pts_dst_mask
    else:
        pts_src=seq_pts_or_pt_mask
        pts_dst=trans(pts_src,ssize,dsize)
        mask=(pts_src<0).any(axis=1)
        pts_dst[mask]=pts_src[mask]
        return pts_dst

#seq_pts_or_pt_mask [Nx3,Nx2,HxW,HxWxN]
def flip_points(seq_pts_or_pt_mask,direction=1,ssize=None):
    if ssize is None:
        if direction==1:
            pts_dst_mask=seq_pts_or_pt_mask[:,::-1,...]
        else:
            pts_dst_mask=seq_pts_or_pt_mask[::-1,:,...]
        return pts_dst_mask
    else:
        pts_src=seq_pts_or_pt_mask
        pts_dst=pts_src.copy()
        pts_dst[:,1-direction]*=-1
        pts_dst[:,1-direction]-=1
        pts_dst[:,1-direction]+=ssize[1-direction]
        mask=(pts_src<0).any(axis=1)
        pts_dst[mask]=pts_src[mask]
        return pts_dst

def set_fixed_seed(id):     
    str_seed=0
    for c in id:
        str_seed+=ord(c)
    str_seed=str_seed%50
    random.seed(str_seed)
    np.random.seed(str_seed)

########################################[ Robot Strategy ]########################################

def get_next_anno_point(pred, gt, ignore_pts=np.empty([0,3],dtype=np.int64)):
    fndist_map=distance_transform_edt(np.pad((gt==1)&(pred==0),1,mode='constant'))[1:-1, 1:-1]
    fpdist_map=distance_transform_edt(np.pad((gt==0)&(pred==1),1,mode='constant'))[1:-1, 1:-1]

    if ignore_pts.shape==gt.shape:    
        fndist_map[ignore_pts>0]=0
        fpdist_map[ignore_pts>0]=0
    else:
        fndist_map[ignore_pts[:,1],ignore_pts[:,0]]=0
        fpdist_map[ignore_pts[:,1],ignore_pts[:,0]]=0

    fndist_max=fndist_map.max()
    fpdist_max=fpdist_map.max()

    if fndist_max==0 and fpdist_max==0:
        if (gt==1).any():
            pt_next=get_next_anno_point(np.zeros_like(gt),gt)
        else:
            pt_next=(gt.shape[1]//2,gt.shape[0]//2,1)
    else:
        [usr_map,if_pos] = [fndist_map, 1] if fndist_max>fpdist_max else [fpdist_map, 0]
        [y_mlist, x_mlist] = np.where(usr_map == usr_map.max())
        pt_next=(x_mlist[0],y_mlist[0],if_pos)

    return pt_next

########################################[ Train Sample Strategy ]########################################

def get_points_walk(gt, point_num, mode='fg', margin_min=15, margin_max=10000, step=0, pre_points=[], if_get_mask=False):
    if point_num==0: return np.zeros_like(gt) if if_get_mask else  np.empty([0,3],dtype=np.int64)

    dist_map_fg = distance_transform_edt(np.pad(gt==1, 1, mode='constant'))[1:-1, 1:-1]

    if isinstance(margin_min, list):margin_min=random.choice(margin_min)
    if isinstance(margin_max, list):margin_max=random.choice(margin_max)
    if isinstance(margin_min,float):margin_min= int(dist_map_fg.max()*margin_min)
    if isinstance(margin_max,float):margin_max= int(dist_map_fg.max()*margin_max)

    if isinstance(step, list):step=random.choice(step)
    if isinstance(margin_min,float):step= int(np.sqrt((gt==1).sum()/np.pi)*2*step)

    if mode=='fg':
        pts_cand= np.argwhere((dist_map_fg>margin_min)&(dist_map_fg<margin_max))[:,::-1]
    elif mode=='bg':
        dist_map_bg=distance_transform_edt(gt==0)
        pts_cand= np.argwhere((dist_map_bg>margin_min)&(dist_map_bg<margin_max))[:,::-1]
    elif mode=='bd':
        assert(margin_min==0)
        dist_map_bg=distance_transform_edt(gt==0)
        pts_cand= np.argwhere(((dist_map_fg>margin_min)&(dist_map_fg<margin_max)) | ((dist_map_bg>margin_min)&(dist_map_bg<margin_max)) )[:,::-1]

    step_squ=step**2

    for pt_cur in pre_points:
        pts_cand=pts_cand[(((pts_cand-pt_cur)**2).sum(axis=1))>step_squ]

    if step==0:
        point_num=min(point_num,len(pts_cand))

        if point_num>0:
            points=pts_cand[np.random.choice(range(len(pts_cand)),size=point_num,replace=False)]
        else:
            points=np.array([])

    else:
        points = []
        for i in range(point_num):
            if len(pts_cand)==0:break
            pt_cur=pts_cand[np.random.randint(len(pts_cand))]
            points.append(pt_cur)
            pts_cand=pts_cand[(((pts_cand-pt_cur)**2).sum(axis=1))>step_squ]
        points=np.array(points)

    points= np.empty([0,3],dtype=np.int64) if len(points)==0 else  np.column_stack((points,gt[points[:,1], points[:,0]]))    #TODO check all 0 or all 1

    if if_get_mask:
        mask=np.zeros_like(gt)
        mask[points[:,1], points[:,0]]=1
        return mask
    else:
        return points


def get_first_pt(gt,min_ratio=1.0):
    dist=distance_transform_edt(np.pad(gt,1,mode='constant'))[1:-1, 1:-1]
    pts_cand=np.argwhere(dist>=(dist.max()*min_ratio))[:,::-1]
    assert(len(pts_cand)>0)
    pt=pts_cand[np.random.randint(len(pts_cand))]
    return (pt[0],pt[1],1)

########################################[ Evaluation for IIS ]########################################

def get_noc_miou(record_dict,ids=None,point_num=20):
    if ids is None:ids=list(record_dict.keys())
    noc_miou= np.mean(np.array([record_dict[id]['ious'][:point_num+1] for id in ids]),axis=0) if len(ids)!=0 else None#[None for _ in range(point_num+1)]
    return noc_miou
    
def get_mnoc(record_dict,ids=None,iou_targets=[0.85]):
    if ids is None:ids=list(record_dict.keys())
    if len(ids)==0: return None
    mNoCs=[]
    for iou_target in iou_targets:
        nocs=[]
        for id in ids:
            ious=np.array(record_dict[id]['ious'])
            nocs.append(np.argmax(ious>=iou_target) if (ious>=iou_target).any() else (len(ious)-1))
        mNoCs.append(np.mean(nocs))
    return mNoCs

def get_nof(record_dict,ids=None,iou_targets=[0.85],max_point_num=20):
    if ids is None:ids=list(record_dict.keys())
    if len(ids)==0: return None
    NoFs=[]
    for iou_target in iou_targets:
        nof= sum([int((np.array(record_dict[id]['ious'])>=iou_target).any()==False) for id in ids])
        NoFs.append(nof)
    return NoFs

########################################[ de_transform ]########################################
def get_random_structure(size):
    choice = np.random.randint(1, 5)

    if choice == 1:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif choice == 2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif choice == 3:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size//2))
    elif choice == 4:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size//2, size))

def random_dilate(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.dilate(seg,kernel,iterations = 1)
    return seg

def random_erode(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.erode(seg,kernel,iterations = 1)
    return seg

def compute_iou(seg, gt):
    intersection = seg*gt
    union = seg+gt
    return (np.count_nonzero(intersection) + 1e-6) / (np.count_nonzero(union) + 1e-6)

def perturb_seg(gt, iou_target=0.6):
    h, w = gt.shape
    seg = gt.copy()

    _, seg = cv2.threshold(seg, 127, 255, 0)

    # Rare case
    if h <= 2 or w <= 2:
        print('GT too small, returning original')
        return seg

    # Do a bunch of random operations
    for _ in range(250):
        for _ in range(4):
            lx, ly = np.random.randint(w), np.random.randint(h)
            lw, lh = np.random.randint(lx+1,w+1), np.random.randint(ly+1,h+1)

            # Randomly set one pixel to 1/0. With the following dilate/erode, we can create holes/external regions
            if np.random.rand() < 0.25:
                cx = int((lx + lw) / 2)
                cy = int((ly + lh) / 2)
                seg[cy, cx] = np.random.randint(2) * 255

            if np.random.rand() < 0.5:
                seg[ly:lh, lx:lw] = random_dilate(seg[ly:lh, lx:lw])
            else:
                seg[ly:lh, lx:lw] = random_erode(seg[ly:lh, lx:lw])

        if compute_iou(seg, gt) < iou_target:
            break

    return seg


def my_perturb(gt,alpha_range=(0.2,0.8),beta_range=0.3,iou_target=(0.5,0.9)):
    gt=np.uint8(gt>0)
    boundary=np.zeros_like(gt)
    cv2.drawContours(boundary,cv2.findContours(gt,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0],-1,1,1)
    boundary[[0,-1],:]=boundary[:,[0,-1]]=0
    boundary_pts=np.argwhere(boundary==1)[:,::-1]

    dist_center=np.ones_like(gt)
    dist_center[gt.shape[0]//2,gt.shape[1]//2]=0
    dist_center=distance_transform_edt(dist_center)
    dist_flat=dist_center[boundary_pts[:,1],boundary_pts[:,0]]
    dist_flat=1.0-(dist_flat/dist_flat.max())

    k=np.sqrt(gt.sum())

    mask=gt.copy()
    if  isinstance(iou_target,(tuple,list)):
        iou_target=np.random.uniform(*iou_target)

    for pt_idx in random.choices(range(len(boundary_pts)),k=100,weights=dist_flat):
        pt=boundary_pts[pt_idx]

        alpha=np.random.uniform(*alpha_range)
        expand_r=max(int(alpha*k),5)

        pt_lt = np.minimum(np.maximum(pt-expand_r,np.array([0,0])),np.array(gt.shape[::-1])-1)
        pt_rb = np.minimum(np.maximum(pt+expand_r,np.array([0,0])),np.array(gt.shape[::-1])-1)

        if np.random.rand() < 0.5:
            mask[pt_lt[1]:pt_rb[1],pt_lt[0]:pt_rb[0]] = random_dilate(mask[pt_lt[1]:pt_rb[1],pt_lt[0]:pt_rb[0]],min=10,max=30)  
        else:
            mask[pt_lt[1]:pt_rb[1],pt_lt[0]:pt_rb[0]] = random_erode(mask[pt_lt[1]:pt_rb[1],pt_lt[0]:pt_rb[0]],min=10,max=30)
    
        if compute_iou(mask, gt) < iou_target: break

    return mask


# metric :['miou','dice','assd']
def get_metric(pred,gt,metric='iou'):
    if metric=='iou':
        mask = (gt >= 0) & (gt <= 1)
        label = 2 * gt[mask] + pred[mask]
        count = np.bincount(label, minlength=4)
        cm = count.reshape(2, 2)
        score = cm[1,1]/(cm[0,1]+cm[1,0]+cm[1,1])
    elif metric=='dice':
        score=  (((gt==1)&(pred==1)).sum()*2)/((gt==1).sum()+(pred==1).sum())
    elif metric=='assd':
        pred=pred.astype(np.uint8)
        gt=(gt==1).astype(np.uint8)
        gt_contours,_=cv2.findContours(gt,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        gt_boundary_mask=np.zeros_like(gt)
        cv2.drawContours(gt_boundary_mask,gt_contours,-1,1,1)

        pred_contours,_=cv2.findContours(pred,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        pred_boundary_mask=np.zeros_like(pred)
        cv2.drawContours(pred_boundary_mask,pred_contours,-1,1,1)

        gt_dist=distance_transform_edt(1-gt_boundary_mask)
        pred_dist=distance_transform_edt(1-pred_boundary_mask)
        
        score=(gt_dist[pred_boundary_mask==1].sum()+pred_dist[gt_boundary_mask==1].sum())/(gt_boundary_mask.sum()+pred_boundary_mask.sum())
        

    elif metric=='biou':
        def mask_to_boundary(mask, dilation_ratio=0.02):
            h, w = mask.shape
            img_diag = np.sqrt(h ** 2 + w ** 2)
            dilation = int(round(dilation_ratio * img_diag))
            if dilation < 1:
                dilation = 1
            # Pad image so mask truncated by the image border is also considered as boundary.
            new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            kernel = np.ones((3, 3), dtype=np.uint8)
            new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
            mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
            # G_d intersects G in the paper.
            return mask - mask_erode

        pred_bd=mask_to_boundary(pred)
        gt_bd=mask_to_boundary(gt)
        score= ((pred_bd==1)&(gt_bd==1)).sum()/(((pred_bd==1)|(gt_bd==1))&(gt_bd!=255)).sum()

    return score


########################################[ RITM click simulated (not used)]########################################

from functools import lru_cache
import math

class BasePointSampler:
    def __init__(self):
        self._selected_mask = None
        self._selected_masks = None

    def sample_object(self, sample):
        raise NotImplementedError

    def sample_points(self):
        raise NotImplementedError

    @property
    def selected_mask(self):
        assert self._selected_mask is not None
        return self._selected_mask

    @selected_mask.setter
    def selected_mask(self, mask):
        self._selected_mask = mask[np.newaxis, :].astype(np.float32)

class MultiPointSampler(BasePointSampler):
    def __init__(self, max_num_points=12, prob_gamma=0.7, expand_ratio=0.1,
                 positive_erode_prob=0.9, positive_erode_iters=3,
                 negative_bg_prob=0.1, negative_other_prob=0.4, negative_border_prob=0.5,
                 first_click_center=False, only_one_first_click=False,
                 sfc_inner_k=1.7, sfc_full_inner_prob=0.0):
        super().__init__()
        self.max_num_points = max_num_points
        self.expand_ratio = expand_ratio
        self.positive_erode_prob = positive_erode_prob
        self.positive_erode_iters = positive_erode_iters
        self.first_click_center = first_click_center
        self.only_one_first_click = only_one_first_click
        self.sfc_inner_k = sfc_inner_k
        self.sfc_full_inner_prob = sfc_full_inner_prob

        self.neg_strategies = ['bg', 'other', 'border']
        self.neg_strategies_prob = [negative_bg_prob, negative_other_prob, negative_border_prob]
        assert math.isclose(sum(self.neg_strategies_prob), 1.0)


        self._pos_probs = generate_probs(max_num_points, gamma=prob_gamma)
        self._neg_probs = generate_probs(max_num_points + 1, gamma=prob_gamma)
        self._neg_masks = None

    def sample_points(self):
        assert self._selected_mask is not None
        pos_points = self._multi_mask_sample_points(self._selected_masks,
                                                    is_negative=[False] * len(self._selected_masks),
                                                    with_first_click=self.first_click_center)

        neg_strategy = [(self._neg_masks[k], prob)
                        for k, prob in zip(self.neg_strategies, self.neg_strategies_prob)]
        neg_masks = self._neg_masks['required'] + [neg_strategy]
        neg_points = self._multi_mask_sample_points(neg_masks,
                                                    is_negative=[False] * len(self._neg_masks['required']) + [True])

        #return pos_points + neg_points
        return [pos_points , neg_points]

    def _multi_mask_sample_points(self, selected_masks, is_negative, with_first_click=False):
        selected_masks = selected_masks[:self.max_num_points]

        each_obj_points = [
            self._sample_points(mask, is_negative=is_negative[i],
                                with_first_click=with_first_click)
            for i, mask in enumerate(selected_masks)
        ]
        each_obj_points = [x for x in each_obj_points if len(x) > 0]

        points = []
        if len(each_obj_points) == 1:
            points = each_obj_points[0]
        elif len(each_obj_points) > 1:
            if self.only_one_first_click:
                each_obj_points = each_obj_points[:1]

            points = [obj_points[0] for obj_points in each_obj_points]

            aggregated_masks_with_prob = []
            for indx, x in enumerate(selected_masks):
                if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)):
                    for t, prob in x:
                        aggregated_masks_with_prob.append((t, prob / len(selected_masks)))
                else:
                    aggregated_masks_with_prob.append((x, 1.0 / len(selected_masks)))

            other_points_union = self._sample_points(aggregated_masks_with_prob, is_negative=True)
            if len(other_points_union) + len(points) <= self.max_num_points:
                points.extend(other_points_union)
            else:
                points.extend(random.sample(other_points_union, self.max_num_points - len(points)))

        # if len(points) < self.max_num_points:
        #     points.extend([(-1, -1, -1)] * (self.max_num_points - len(points)))

        return points

    def _sample_points(self, mask, is_negative=False, with_first_click=False):
        if is_negative:
            num_points = np.random.choice(np.arange(self.max_num_points + 1), p=self._neg_probs)
        else:
            num_points = 1 + np.random.choice(np.arange(self.max_num_points), p=self._pos_probs)

        indices_probs = None
        if isinstance(mask, (list, tuple)):
            indices_probs = [x[1] for x in mask]
            indices = [(np.argwhere(x), prob) for x, prob in mask]
            if indices_probs:
                assert math.isclose(sum(indices_probs), 1.0)
        else:
            indices = np.argwhere(mask)

        points = []
        for j in range(num_points):
            first_click = with_first_click and j == 0 and indices_probs is None

            if first_click:
                point_indices = get_point_candidates(mask, k=self.sfc_inner_k, full_prob=self.sfc_full_inner_prob)
            elif indices_probs:
                point_indices_indx = np.random.choice(np.arange(len(indices)), p=indices_probs)
                point_indices = indices[point_indices_indx][0]
            else:
                point_indices = indices

            num_indices = len(point_indices)
            if num_indices > 0:
                point_indx = 0 if first_click else 100
                click = point_indices[np.random.randint(0, num_indices)].tolist() + [point_indx]
                points.append(click)

        return points

    def _positive_erode(self, mask):
        if random.random() > self.positive_erode_prob:
            return mask

        kernel = np.ones((3, 3), np.uint8)
        eroded_mask = cv2.erode(mask.astype(np.uint8),
                                kernel, iterations=self.positive_erode_iters).astype(np.bool)

        if eroded_mask.sum() > 10:
            return eroded_mask
        else:
            return mask

    def _get_border_mask(self, mask):
        expand_r = int(np.ceil(self.expand_ratio * np.sqrt(mask.sum())))
        kernel = np.ones((3, 3), np.uint8)
        expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=expand_r)
        expanded_mask[mask.astype(np.bool)] = 0
        return expanded_mask


    def get_clicks_lz(self,gt,mask_obj=None):
        if gt.any()==False:
            return np.empty([0,3],dtype=np.int64),np.empty([0,3],dtype=np.int64)

        #init 
        gt=gt.astype(np.int32)
        gt_mask, pos_masks, neg_masks = gt, [self._positive_erode(x) for x in [gt] ], []
        
        binary_gt_mask =  gt_mask > 0

        self.selected_mask = gt
        self._selected_masks = pos_masks
    
        neg_mask_bg = np.logical_not(binary_gt_mask)
        neg_mask_other = neg_mask_bg if mask_obj is None else  ((mask_obj==1) & (gt==0))   
        neg_mask_border = self._get_border_mask(binary_gt_mask)

        self._neg_masks = {
            'bg': neg_mask_bg,
            'other': neg_mask_other,
            'border': neg_mask_border,
            'required': neg_masks
        }
        # get clicks
        pos_points , neg_points=self.sample_points()
        if len(pos_points)==0:pos_points=np.empty([0,3],dtype=np.int64)
        if len(neg_points)==0:neg_points=np.empty([0,3],dtype=np.int64)

        pos_points=np.int64(np.array(pos_points))
        neg_points=np.int64(np.array(neg_points))

        pos_points[:,2]=1
        neg_points[:,2]=0

        pos_points=pos_points[:,[1,0,2]]
        neg_points=neg_points[:,[1,0,2]]

        return pos_points,neg_points

@lru_cache(maxsize=None)
def generate_probs(max_num_points, gamma):
    probs = []
    last_value = 1
    for i in range(max_num_points):
        probs.append(last_value)
        last_value *= gamma

    probs = np.array(probs)
    probs /= probs.sum()

    return probs


def get_point_candidates(obj_mask, k=1.7, full_prob=0.0):
    if full_prob > 0 and random.random() < full_prob:
        return obj_mask

    padded_mask = np.pad(obj_mask, ((1, 1), (1, 1)), 'constant')

    dt = cv2.distanceTransform(padded_mask.astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]
    if k > 0:
        inner_mask = dt > dt.max() / k
        return np.argwhere(inner_mask)
    else:
        prob_map = dt.flatten()
        prob_map /= max(prob_map.sum(), 1e-6)
        click_indx = np.random.choice(len(prob_map), p=prob_map)
        click_coords = np.unravel_index(click_indx, dt.shape)
        return np.array([click_coords])




    














































    