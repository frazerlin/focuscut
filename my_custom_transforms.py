import cv2
import torch
import random
import numpy as np
from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt
import helpers
from copy import deepcopy
from torchvision import transforms as tr

########################################[ General ]########################################

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class ToTensor(object):
    def __init__(self, div_list=None, elems_do=None, elems_undo=[]):
        self.div_list = div_list
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
    def __call__(self, sample):
        for elem in sample.keys():
            if self.elems_do!= None and elem not in self.elems_do :continue
            if elem in self.elems_undo:continue
            tmp = sample[elem]
            tmp = tmp[np.newaxis,:,:] if tmp.ndim == 2 else tmp.transpose((2, 0, 1))
            tmp = torch.from_numpy(tmp)
            if self.div_list is None:
                tmp = tmp.float().div(255) if isinstance(tmp, torch.ByteTensor) else tmp.float()
            else:
                tmp = tmp.float().div(255) if elem in self.div_list else tmp.float()
            sample[elem] = tmp                          
        return sample

class Normalize(object):
    def __init__(self, mean_std=[[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]], elems_do=['img'], elems_undo=[]):
        self.mean_std = mean_std 
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
    def __call__(self, sample):
        if self.mean_std is None:return sample
        for elem in sample.keys():
            if self.elems_do!= None  and elem not in self.elems_do :continue
            if elem in self.elems_undo:continue
            tensor = sample[elem]
            for t, m, s in zip(tensor, self.mean_std[0], self.mean_std[1]):
                t.sub_(m).div_(s)
        return sample

class Show(object):
    def __init__(self, elems_show=['img','gt'],elems_info=[],pre_process=None,elems_do=None, elems_undo=[]):
        self.elems_show = elems_show
        self.elems_info = elems_info
        self.pre_process=pre_process
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
    def __call__(self, sample):
        return sample

########################################[ Basic Image Augmentation ]########################################

class RandomFlip(object):
    def __init__(self, direction=1, p=0.5,elems_point=['seq_points'],elems_do=None, elems_undo=[]):
        self.direction, self.p = direction, p
        self.elems_point = elems_point
        self.elems_do, self.elems_undo = elems_do, (['meta','ori']+elems_undo+elems_point)
    def __call__(self, sample):
        if_flip= (random.random() < self.p)
        if if_flip:
            for elem in self.elems_point:
                if elem in sample.keys():
                    sample[elem]=helpers.flip_points(sample[elem],ssize=sample['img'].shape[:2][::-1],direction=self.direction)

            for elem in sample.keys():
                if self.elems_do!= None  and elem not in self.elems_do :continue
                if elem in self.elems_undo:continue
                sample[elem]=cv2.flip(sample[elem],self.direction)
        sample['meta']['if_flip']=if_flip
        return sample

class Resize(object):
    def __init__(self, size, interpolation=None,  elems_point=['seq_points'], elems_do=None, elems_undo=['img_ori','gt_ori']):
        self.size =(size,size) if isinstance(size,int) else tuple(size)
        self.interpolation = interpolation
        self.elems_point = elems_point
        self.elems_do, self.elems_undo = elems_do, (['meta','ori']+elems_undo+elems_point)

    def __call__(self, sample):
        for elem in self.elems_point:
            if elem in sample.keys():
                sample[elem]=helpers.resize_points(sample[elem],self.size,ssize=sample['img'].shape[:2][::-1])

        for elem in sample.keys():
            if self.elems_do!= None  and elem not in self.elems_do :continue
            if elem in self.elems_undo:continue
            if sample[elem].shape[:2]==self.size: continue
            sample[elem]=cv2.resize(sample[elem], self.size, interpolation=(cv2.INTER_LINEAR if sample[elem].ndim==3 else cv2.INTER_NEAREST) if self.interpolation is None else self.interpolation)
        return sample

#expand pad(TBLR)
class Expand(object):
    def __init__(self, pad=(0,0,0,0),elems_point=['seq_points'],elems_do=None, elems_undo=[]):
        if isinstance(pad, int):
            self.pad=(pad, pad, pad, pad)
        elif len(pad)==2:
            self.pad=(pad[0],pad[0],pad[1],pad[1])
        elif len(pad)==4:
            self.pad= pad
        self.elems_point = elems_point
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo+elems_point)
    def __call__(self, sample):
        for elem in self.elems_point:
            if elem in sample.keys():
                pass

        for elem in sample.keys():
            if self.elems_do!= None  and elem not in self.elems_do :continue
            if elem in self.elems_undo:continue
            sample[elem]=cv2.copyMakeBorder(sample[elem],self.pad[0],self.pad[1],self.pad[2],self.pad[3],cv2.BORDER_CONSTANT)  
        return sample

class Crop(object):
    def __init__(self, crop_bbox, pad=0, expand_mode=cv2.BORDER_CONSTANT,elems_point=['seq_points'],elems_do=None, elems_undo=['img_ori','gt_ori']):
        self.crop_bbox,self.pad,self.expand_mode = crop_bbox,pad,expand_mode
        self.elems_point = elems_point
        self.elems_do, self.elems_undo = elems_do, (['meta','ori']+elems_undo+elems_point)
    def __call__(self, sample):
        for elem in self.elems_point:
            if elem in sample.keys():
                if self.pad>0:
                    sample[elem]=sample[elem].copy()
                    sample[elem][sample[elem][:,2]>=0,:2]+=self.pad
                sample[elem]= helpers.crop_points(sample[elem],self.crop_bbox)
                
        for elem in sample.keys():
            if self.elems_do!= None  and elem not in self.elems_do :continue
            if elem in self.elems_undo:continue
            tmp=sample[elem]
            if self.pad>0: tmp=cv2.copyMakeBorder(tmp,self.pad,self.pad,self.pad,self.pad,self.expand_mode)
            sample[elem]=tmp[self.crop_bbox[1]: self.crop_bbox[1]+self.crop_bbox[3], self.crop_bbox[0]: self.crop_bbox[0]+self.crop_bbox[2], ...]

        sample['meta']['crop_bbox']= np.array(self.crop_bbox)
        return sample

class RandomScale(object):
    def __init__(self, scale=(0.75, 1.25), elems_do=None, elems_undo=[]):
        self.scale = scale
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
    def __call__(self, sample):
        scale_tmp = random.uniform(self.scale[0], self.scale[1])
        src_size=sample['gt'].shape[::-1]
        dst_size= ( int(src_size[0]*scale_tmp), int(src_size[1]*scale_tmp))
        Resize(size=dst_size)(sample)   
        return sample


class ImgAug(object):
    def __init__(self, mode='bri,con,sat,hue,gam', elems_do=['img'], elems_undo=[]):
        self.processes = None if mode=='' else [ self.__get_aug_parm(m) for m in mode.split(',')]
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
    def __call__(self, sample):
        for elem in sample.keys():
            if self.elems_do!= None and elem not in self.elems_do :continue
            if elem in self.elems_undo:continue
            if self.processes is not None:
                tmp=Image.fromarray(sample[elem])
                for func,choice_range in self.processes: tmp=func(tmp,np.random.uniform(*choice_range))
                sample[elem]=np.array(tmp)
        return sample 

    def __get_aug_parm(self,mode):
        if mode.startswith('bri'):
            base=1.0
            delta_default=0.25
            func=tr.functional.adjust_brightness
        elif mode.startswith('con'):
            base=1.0
            delta_default=0.25
            func=tr.functional.adjust_contrast
        elif mode.startswith('sat'):
            base=1.0
            delta_default=0.25
            func=tr.functional.adjust_saturation
        elif mode.startswith('hue'):
            base=0.0
            delta_default=0.1
            func=tr.functional.adjust_hue
        elif mode.startswith('gam'):
            base=1.0
            delta_default=0.25
            func=tr.functional.adjust_gamma

        mode_split=mode.split('-')
        if len(mode_split)==1:
            choice_range=[base-delta_default,base+delta_default]
        elif len(mode_split)==2:
            mode_split2=mode_split[1].split('_')
            if len(mode_split2)==1:
                delta=float(mode_split2[0])
                choice_range=[base-delta,base+delta]
            elif len(mode_split2)==2:
                choice_range=[base-float(mode_split2[0]),base+float(mode_split2[1])]

        return [func,choice_range]

class RandomRotate(object):
    def __init__(self, angle_range=30, if_expand=True, mode=None, elems_point=['seq_points'], elems_do=None, elems_undo=[]):
        self.angle_range=angle_range
        self.if_expand, self.mode = if_expand, mode
        self.elems_point = elems_point
        self.elems_do, self.elems_undo = elems_do, (['meta','ori']+elems_undo+elems_point)

    def __call__(self, sample):
        if isinstance(self.angle_range,tuple):
            angle=random.randint(*self.angle_range)
        elif isinstance(self.angle_range,list):
            angle=random.choice(self.angle_range)
        else:
            angle=random.randint(-self.angle_range,self.angle_range)

        for elem in self.elems_point:
            if elem in sample.keys():
                pass
                #sample[elem]=helpers.img_rotate_point(sample[elem], angle, if_expand=self.if_expand)

        for elem in sample.keys():
            if self.elems_do!= None  and elem not in self.elems_do :continue
            if elem in self.elems_undo:continue
            sample[elem]=helpers.img_rotate(sample[elem], angle, if_expand=self.if_expand, mode=self.mode)  

        sample['meta']['rotate']=angle

        return sample


########################################[ Interactive Segmentation ]########################################

class CatCutAux(object):
    def __init__(self, mode='no', elems_do=None, elems_undo=[]):
        self.mode = mode
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
    def __call__(self, sample):
        if self.mode == 'bbox_mask':
            bbox = cv2.boundingRect((sample['gt']>127).astype(np.uint8))
            mask=helpers.mask_from_bbox(bbox,sample['gt'].shape)*255
            sample['img'] = np.concatenate((sample['img'],mask[:,:,np.newaxis]),axis=2)
        elif self.mode == 'gt':
            sample['img'] = np.concatenate((sample['img'],sample['gt'][:,:,np.newaxis]),axis=2)
        return sample

class MatchSideResize(object):
    def __init__(self, size=512,if_short=True,if_expand=False,elems_do=None, elems_undo=[]):
        self.size,self.if_short,self.if_expand = size,if_short,if_expand
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
        
    def __call__(self, sample):
        src_size = sample['img'].shape[:2][::-1]
        src_size_ref = min(src_size) if self.if_short else max(src_size)
        dst_size= (int(self.size*src_size[0]/src_size_ref), int(self.size*src_size[1]/src_size_ref))
        assert(dst_size[0]==self.size or dst_size[1] == self.size)
        Resize(size=dst_size)(sample)   
        if self.if_expand and if_short==False:
            Expand(pad=(0,self.size-dst_size[1],0,self.size-dst_size[0]))(sample)
        return sample

class FgContainCrop(object):
    def __init__(self, mask_elem='gt',crop_size=384, if_whole=False, if_middle=False, if_with_crop=True,elems_do=None, elems_undo=[]):
        self.mask_elem=mask_elem
        self.crop_size = (crop_size,crop_size) if isinstance(crop_size,int) else crop_size
        self.if_whole,self.if_middle,self.if_with_crop =  if_whole, if_middle, if_with_crop
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
    def __call__(self, sample):
        ref = sample[self.mask_elem].copy()
        src_size=ref.shape[:2][::-1]

        if isinstance(self.crop_size,str):
            crop_size=min(src_size) if self.crop_size=='short' else max(src_size)
            crop_size=(crop_size,crop_size)
        else:
            crop_size=self.crop_size

        crop_size=np.minimum(np.array(crop_size),np.array(src_size))
        x_range, y_range = [0,src_size[0]-crop_size[0]], [0,src_size[1]-crop_size[1]] 
        bbox=cv2.boundingRect((ref>0).astype(np.uint8))
        if (ref>0).any()==False:
            pass
        elif self.if_whole: 
            if bbox[2]<=crop_size[0]:
                x_range[1]=min(x_range[1], bbox[0])  
                x_range[0]=max(x_range[0], bbox[0]+bbox[2]-crop_size[0]) 
            else:
                x_range=[bbox[0], bbox[0]+bbox[2]-crop_size[0]]

            if bbox[3]<=crop_size[1]:
                y_range[1]=min(y_range[1], bbox[1])
                y_range[0]=max(y_range[0], bbox[1]+bbox[3]-crop_size[1])
            else:
                y_range=[bbox[1], bbox[1]+bbox[3]-crop_size[1]]
        else:
            pts_xy=np.argwhere(ref>0)[:,::-1]
            sp_x,sp_y =  pts_xy[random.randint(0,len(pts_xy)-1)]
            x_range[1], y_range[1] = min(x_range[1], sp_x), min(y_range[1], sp_y)
            x_range[0], y_range[0] = max(x_range[0], sp_x+1-crop_size[0]), max(y_range[0], sp_y+1-crop_size[1])

        if self.if_middle:
            x_st= min(max(bbox[0]+(bbox[2]//2)+1-(crop_size[0]//2),x_range[0]),x_range[1])
            y_st= min(max(bbox[1]+(bbox[3]//2)+1-(crop_size[1]//2),y_range[0]),y_range[1])
        else:
            x_st=random.randint(x_range[0],x_range[1])
            y_st=random.randint(y_range[0],y_range[1])

        
        crop_bbox=(x_st,y_st,crop_size[0],crop_size[1])
        if self.if_with_crop:
            Crop(crop_bbox=crop_bbox)(sample)
        else:
            sample['meta']['crop_bbox']= np.array(crop_bbox)

        return sample


########################################[ Interactive Segmentation (Points) ]########################################

class CatPointMask(object):
    def __init__(self, mode='no', max_dist=255, if_repair=True):
        self.mode,self.max_dist,self.if_repair = mode, max_dist, if_repair
    def __call__(self, sample):

        img_size=sample['img'].shape[:2][::-1]

        if self.mode == 'dist_src':
            sample['pos_points_mask'] = helpers.get_points_mask(img_size, sample['seq_points'][sample['seq_points'][:,2]==1])
            sample['neg_points_mask'] = helpers.get_points_mask(img_size, sample['seq_points'][sample['seq_points'][:,2]==0])
            sample['first_point_mask'] = helpers.get_points_mask(img_size, sample['seq_points'][0:1])

            for dist_mask,point_mask in zip(['pos_map_dist_src','neg_map_dist_src','first_map_dist_src'], ['pos_points_mask','neg_points_mask','first_point_mask']):
                sample[dist_mask]=np.minimum(np.float32(distance_transform_edt(1-sample[point_mask])),self.max_dist) if sample[point_mask].any() else np.ones(img_size[::-1],dtype=np.float32)*self.max_dist
        
        elif self.mode == 'first_dist_src':
            sample['first_point_mask'] = helpers.get_points_mask(img_size, sample['seq_points'][0:1])
            
            for dist_mask,point_mask in zip(['first_map_dist_src'], ['first_point_mask']):
                sample[dist_mask]=np.minimum(np.float32(distance_transform_edt(1-sample[point_mask])),self.max_dist) if sample[point_mask].any() else np.ones(img_size[::-1],dtype=np.float32)*self.max_dist

        sample['seq_points']=helpers.fill_up_list(sample['seq_points'],50)

        return sample

class SimulatePoints(object):
    def __init__(self, simulate_strategy='random',if_fixed=False):
        ref_dict={'fcanet':'02-02','first':'02-00','ritm':'00-03'}
        self.simulate_strategy =ref_dict[simulate_strategy] if simulate_strategy in ref_dict else simulate_strategy
        self.if_fixed = if_fixed
        self.mps_ritm=helpers.MultiPointSampler(max_num_points=24,prob_gamma=0.80)

    def __call__(self, sample):
        if self.if_fixed: helpers.set_fixed_seed(sample['meta']['id'])
        gt=(sample['gt']>127).astype(np.uint8)
        first_strategy,other_strategy=self.simulate_strategy.split('-')

        if first_strategy in ['00','none']:
            first_click=np.empty([0,3],dtype=np.int64)
        elif first_strategy in ['01','random']:
            first_click=helpers.get_points_walk(gt,1,mode='fg',margin_min=0,step=0,if_get_mask=False)
        elif first_strategy in ['02','best']:
            first_click=np.array([helpers.get_next_anno_point(np.zeros_like(gt), gt)])
        elif first_strategy in ['03','range']:
            first_click=np.array([helpers.get_first_pt(gt,0.6)])
        else:
            raise ValueError('simulate_strategy')

        pos_point_num, neg_point_num=random.randint(int(len(first_click)==0),10),random.randint(0,10)

        if other_strategy in ['00','none']:
            pos_clicks=np.empty([0,3],dtype=np.int64)
            neg_clicks=np.empty([0,3],dtype=np.int64)
        elif other_strategy in ['01','random']:
            pos_clicks=helpers.get_points_walk(gt,pos_point_num,mode='fg',margin_min=0,step=0,if_get_mask=False)
            neg_clicks=helpers.get_points_walk(gt,neg_point_num,mode='bg',margin_min=0,step=0,if_get_mask=False)
        elif other_strategy in ['02','fcanet']:
            pos_clicks=helpers.get_points_walk(gt,pos_point_num,mode='fg',margin_min=[5,10,15,20],step=[7,10,20],pre_points=first_click[:,:2],if_get_mask=False)   #to add first 
            neg_clicks=helpers.get_points_walk(gt,neg_point_num,mode='bg',margin_min=[15,40,60],margin_max=[80],step=[10,15,25],  if_get_mask=False)
        elif other_strategy in ['03','ritm']:
            pos_clicks,neg_clicks=self.mps_ritm.get_clicks_lz(gt)
        else:
            raise ValueError('simulate_strategy')

        seq_points=np.concatenate([first_click,pos_clicks,neg_clicks],axis=0)
        seq_points[1:]=np.random.permutation(seq_points[1:])
        sample['seq_points']=seq_points

        return sample


current_epoch=0
record_itis={'pred':{},'if_flip':{},'crop_bbox':{},'seq_points':{},'rotate':{}}

class ItisSimulatePoints(object):
    def __init__(self, itis_strategy='best_one',if_fixed=False):
        self.itis_strategy,self.if_fixed = itis_strategy,if_fixed

    def __call__(self, sample):
        global record_itis
        if self.if_fixed: helpers.set_fixed_seed(sample['meta']['id'])
        id=sample['meta']['id']
        pred=helpers.decode_mask(record_itis['pred'][id])
        gt=np.uint8(sample['gt']>127)
        
        if self.itis_strategy in ['00','best_one']:
            pt_next=helpers.get_next_anno_point(pred,gt)

        seq_points=record_itis['seq_points'][id]
        sample['seq_points']=np.array(seq_points+[pt_next])

        return sample

class ZoomIn(object):
    def __init__(self,mask_elem='gt',if_whole=True,if_middle=False,if_with_crop=True,square_mode='must',relax_ratio=0.2,min_crop_size=-1):
        self.mask_elem=mask_elem
        self.if_whole,self.if_middle,self.if_with_crop,self.square_mode=if_whole,if_middle,if_with_crop,square_mode
        self.relax_ratio,self.min_crop_size=relax_ratio,min_crop_size

    def __call__(self, sample):
        ref = sample[self.mask_elem].copy()
        bbox=cv2.boundingRect(np.uint8(ref>0))
        relax_ratio=  random.choice(self.relax_ratio) if isinstance(self.relax_ratio,list) else ( random.uniform(*self.relax_ratio) if isinstance(self.relax_ratio,tuple) else self.relax_ratio )
        min_crop_size=self.min_crop_size if self.min_crop_size>=0 else (min(ref.shape)//2)
        
        #crop_size will be shorter than the shorter side
        if self.square_mode=='must':
            crop_size=min(max(max(bbox[2:])+int(min(bbox[2:])*relax_ratio*2) ,min_crop_size), min(ref.shape[:2]))
        #crop_size will be shorter than the longer side
        elif self.square_mode=='better':
            crop_size=min(max(max(bbox[2:])+int(min(bbox[2:])*relax_ratio*2) ,min_crop_size), max(ref.shape[:2]))
        #no square, crop_size will be (x,y)
        elif self.square_mode=='no':
            crop_size=np.maximum(np.array(bbox[2:])+int(min(bbox[2:])*relax_ratio*2),min_crop_size)

        FgContainCrop(mask_elem=self.mask_elem,crop_size=crop_size,if_whole=self.if_whole,if_middle=self.if_middle,if_with_crop=self.if_with_crop)(sample)
    
        return sample

class ZoomInVal(object):
    def __init__(self,recompute_thresh_iou=0.5):
        self.recompute_thresh_iou=recompute_thresh_iou

    def __call__(self, sample):
        pos_points_mask=helpers.get_points_mask(sample['img'].shape[:2][::-1], sample['seq_points'][sample['seq_points'][:,2]==1])

        if 'pre_pred' not in sample or  (not sample['pre_pred'].any()):
            sample['meta']['crop_bbox']=np.array([0,0,sample['img'].shape[1],sample['img'].shape[0]]) 
        else:
            sample['zoom_ref']=np.maximum(sample['pre_pred'],pos_points_mask)
            ZoomIn(mask_elem='zoom_ref',if_whole=True,if_middle=True,if_with_crop=False,square_mode='better',relax_ratio=0.2)(sample)

        if 'pre_crop_bbox' in sample['meta'] and self.check_pos_all_in_bbox(pos_points_mask,sample['meta']['pre_crop_bbox']) and \
                (self.get_bbox_iou(sample['meta']['crop_bbox'],sample['meta']['pre_crop_bbox'])>=self.recompute_thresh_iou  or  (not sample['pre_pred'].any()) ) :
            sample['meta']['crop_bbox']=sample['meta']['pre_crop_bbox']

        Crop(crop_bbox=sample['meta']['crop_bbox'])(sample)
        return sample

    def check_pos_all_in_bbox(self,pos_points_mask,bbox):
        bbox_mask=helpers.mask_from_bbox(bbox,pos_points_mask.shape)
        return not ((pos_points_mask & (1-bbox_mask)).any())

    def get_bbox_iou(self, b1, b2):
        x1_s,x1_e,x2_s,x2_e =b1[0], b1[0]+b1[2], b2[0], b2[0]+b2[2]
        y1_s,y1_e,y2_s,y2_e =b1[1], b1[1]+b1[3], b2[1], b2[1]+b2[3]
        x_iou=max(0, min(x1_e,x2_e) - max(x1_s,x2_s) )/ max(1e-6, max(x1_e,x2_e) - min(x1_s,x2_s))
        y_iou=max(0, min(y1_e,y2_e) - max(y1_s,y2_s) )/ max(1e-6, max(y1_e,y2_e) - min(y1_s,y2_s))
        return x_iou*y_iou


def gene_pt_with_expand_r(gt,alpha_range=(0.2,0.8),beta_range=0.3):
    if not isinstance(beta_range,(tuple,list)):beta_range=(-beta_range,beta_range)
    k=np.sqrt(gt.sum())
    alpha=np.random.uniform(*alpha_range)
    beta_x=np.random.uniform(*beta_range)
    beta_y=np.random.uniform(*beta_range)
    boundary=np.zeros_like(gt)
    cv2.drawContours(boundary,cv2.findContours(gt,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0],-1,1,1)
    boundary_pts=np.argwhere(boundary==1)[:,::-1]
    expand_r= max(int(alpha*k),5)
    pt=boundary_pts[np.random.randint(len(boundary_pts))]
    pt= np.int64(np.array([pt[0]+expand_r*beta_x,pt[1]+expand_r*beta_y]))
    pt=np.minimum(np.maximum(pt,np.array([0,0])),np.array(gt.shape[::-1])-1)
    return pt,expand_r

class HRSimulatePoints(object):
    def __init__(self,mode='perturb',size=384):
        self.size=list(size) if isinstance(size,(tuple,list)) else (size,size)

    def __call__(self, sample):
        sample_hr=deepcopy(sample['ori'])
        gt=np.uint8(sample_hr['gt']>127)
        pt_hr,expand_r=gene_pt_with_expand_r(gt)
        pt_hr=[pt_hr[0],pt_hr[1],gt[pt_hr[1],pt_hr[0]]]
        sample_hr['center_point']=np.array([pt_hr])
        tsh=15
        sample_hr['gt_weight']=np.minimum(np.maximum(distance_transform_edt(gt),distance_transform_edt(1-gt)),tsh)/tsh
        crop_bbox=(pt_hr[0],pt_hr[1],2*expand_r+1,2*expand_r+1)
        Crop(crop_bbox,pad=expand_r,elems_point=['seq_points','center_point'])(sample_hr)
        Resize(self.size,elems_point=['seq_points','center_point'])(sample_hr)

        sample['img_hr']=sample_hr['img']
        sample['gt_hr']=sample_hr['gt']
        sample['gt_weight_hr']=sample_hr['gt_weight']

        sample['center_point_hr']=sample_hr['center_point']
        sample['pre_pred_hr']=np.uint8(helpers.perturb_seg(sample['gt_hr'])>127)

        gt_hr=np.uint8(sample['gt_hr']>127)

        pos_point_num, neg_point_num=random.randint(0,3),random.randint(0,3)
        pos_clicks=helpers.get_points_walk(gt_hr,pos_point_num,mode='fg',margin_min=[5,10,15,20],step=[7,10,20],if_get_mask=False)
        neg_clicks=helpers.get_points_walk(gt_hr,neg_point_num,mode='bg',margin_min=[15,40,60],margin_max=[80],step=[10,15,25],  if_get_mask=False)
        seq_points=np.concatenate([sample['center_point_hr'],pos_clicks,neg_clicks],axis=0)
        sample['seq_points_hr']=seq_points

        return sample

class HRCatPointMask(object):
    def __init__(self, mode='no', max_dist=255, if_repair=True):
        self.mode,self.max_dist,self.if_repair = mode, max_dist, if_repair
    def __call__(self, sample):
        img_size=sample['img_hr'].shape[:2][::-1]

        if self.mode == 'dist_src':
            sample['pos_points_mask_hr'] = helpers.get_points_mask(img_size, sample['seq_points_hr'][sample['seq_points_hr'][:,2]==1])
            sample['neg_points_mask_hr'] = helpers.get_points_mask(img_size, sample['seq_points_hr'][sample['seq_points_hr'][:,2]==0])
            sample['center_point_mask_hr'] = helpers.get_points_mask(img_size, np.array(sample['center_point_hr']))

            for dist_mask,point_mask in zip(['pos_map_dist_src_hr','neg_map_dist_src_hr','center_map_dist_src_hr'], ['pos_points_mask_hr','neg_points_mask_hr','center_point_mask_hr']):
                sample[dist_mask]=np.minimum(np.float32(distance_transform_edt(1-sample[point_mask])),self.max_dist) if sample[point_mask].any() else np.ones(img_size[::-1],dtype=np.float32)*self.max_dist
        
        sample['seq_points_hr']=helpers.fill_up_list(sample['seq_points_hr'],50)
        return sample

class IIS(object):
    def __init__(self, p):
        self.p=p
        #Aug
        self.FgContainCrop=FgContainCrop(mask_elem='gt',crop_size='short',if_whole=False,if_middle=False,if_with_crop=True)
        self.ZoomIn= ZoomIn(mask_elem='gt',if_whole=True,if_middle=False,if_with_crop=True,square_mode='must',relax_ratio=(0.1,0.3))
        self.Resize=Resize(abs(self.p['ref_size']))
        self.RandomFlip=RandomFlip(p=(0.5 if p['random_flip'] else -1))
        self.RandomRotate=RandomRotate()

        #BaseNet
        self.ItisSimulatePoints=ItisSimulatePoints(itis_strategy=self.p['itis_strategy'])
        self.SimulatePoints=SimulatePoints(simulate_strategy=self.p['simulate_strategy'])
        self.CatPointMask= CatPointMask(mode=self.p['point_map_mode'])

        #SeqNet
        pass

        #After
        self.ToTensor=ToTensor(div_list=['img','gt','img_hr','gt_hr'],elems_undo=['seq_points','seq_points_hr','center_point','center_point_hr'])
        self.Normalize=Normalize( mean_std= None if self.p['no_norm'] else [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]],elems_do=['img','img_hr'])

        #HRCut
        self.if_hrc=self.p['model'].startswith('hrcnet')
        self.HRCatPointMask= HRCatPointMask(mode=self.p['point_map_mode'])
        self.HRSimulatePoints=HRSimulatePoints(size=abs(self.p['hr_size']))

        #Aug
        self.ImgAug=ImgAug(mode=p['augmentation'],elems_do=['img','img_hr'])

    def __call__(self, sample):
        global current_epoch, record_itis
        id=sample['meta']['id']
        sample['ori']=deepcopy(sample)

        if self.p['seq_mode']:
            sample=self.SimulatePoints(self.RandomFlip(self.Resize(self.FgContainCrop(sample) if (self.p['zoom_in']<0 or random.random()<0.5) else self.ZoomIn(sample))))
        else:
            if (random.random()<self.p['itis_pro']) and (current_epoch!=0) and (id in record_itis['pred']):
                if self.p['random_flip']: sample=RandomFlip(p=(record_itis['if_flip'][id]))(sample)
                if self.p['random_rotate']: sample=RandomRotate(angle_range=[record_itis['rotate'][id]])(sample)

                sample=self.ItisSimulatePoints(self.Resize(Crop(record_itis['crop_bbox'][id])(sample)))
                sample['pre_pred']=helpers.decode_mask(record_itis['pred'][id])
            else:
                if self.p['random_flip']: sample=self.RandomFlip(sample)
                if self.p['random_rotate']: sample=self.RandomRotate(sample)
                sample=self.SimulatePoints(self.Resize(self.FgContainCrop(sample) if (self.p['zoom_in']<0 or random.random()<0.5) else self.ZoomIn(sample)))
                sample['pre_pred']=np.zeros_like(sample['gt'])

        if self.if_hrc:
            sample=self.HRSimulatePoints(sample)
            sample=self.HRCatPointMask(sample)
        
        sample=self.CatPointMask(sample)
        del sample['ori']
        sample=self.ImgAug(sample)
        sample=self.Normalize(self.ToTensor(sample))
        return sample


class SelectRadius(object):
    def __init__(self, p):
        self.p=p
        #Aug
        self.FgContainCrop=FgContainCrop(mask_elem='gt',crop_size='short',if_whole=False,if_middle=False,if_with_crop=True)
        self.Resize=Resize(abs(self.p['ref_size']))
        self.RandomFlip=RandomFlip(p=(0.5 if p['random_flip'] else -1))

        #After
        self.ToTensor=ToTensor(div_list=['img','gt','img_hr','gt_hr'],elems_undo=['seq_points','seq_points_hr','center_point','center_point_hr'])
        self.Normalize=Normalize( mean_std= None if self.p['no_norm'] else [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]],elems_do=['img','img_hr'])

    def __call__(self, sample):
        global current_epoch, record_itis
        sample=self.RandomFlip(self.Resize(self.FgContainCrop(sample)))
        sample['pre_pred']=np.uint8(helpers.perturb_seg(sample['gt'])>127)
        pt_next=helpers.get_next_anno_point(sample['pre_pred'],np.uint8(sample['gt']>127))
        sample['target_point_mask'] = helpers.get_points_mask(sample['img'].shape[:2][::-1], np.array([pt_next]))
        sample['target_map_dist_src']=np.float32(distance_transform_edt(1-sample['target_point_mask']))
        sample=self.Normalize(self.ToTensor(sample))
        return sample





