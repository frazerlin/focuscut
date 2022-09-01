
import cv2
import numpy as np
from copy import deepcopy
from scipy.ndimage.morphology import distance_transform_edt,distance_transform_cdt
from torch.utils.data.dataloader import default_collate

import math
import torch
import helpers
import my_custom_transforms as mtr

# [predict for the whole object]
def predict_wo(p,model,sample,seq_points,mode='eval'):
    sample_cpy=deepcopy(sample)
    img_size=sample_cpy['img'].shape[:2][::-1]
    sample_cpy['seq_points']=seq_points.copy()

    if p['zoom_in']>=0 and len(seq_points)>p['zoom_in'] : mtr.ZoomInVal()(sample_cpy)

    if p['eval_resize_mode']!=-1:
        if p['eval_resize_mode']!=1:
            mtr.MatchSideResize(size=abs(p['eval_size']),if_short=(p['eval_resize_mode']==0) )(sample_cpy)
        else:
            mtr.Resize(size=abs(p['eval_size']))(sample_cpy)

    mtr.CatPointMask(mode=p['point_map_mode'], if_repair=False)(sample_cpy)
    mtr.ToTensor(div_list=['img'],elems_undo=['seq_points'])(sample_cpy)
    mtr.Normalize(mean_std=None if p['no_norm'] else [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]])(sample_cpy)
    sample_batched=default_collate([sample_cpy])

    with torch.no_grad():
        if p['eval_flip']:
            output = model(sample_batched,mode=mode,if_eval_flip=True)   #eval flip
        else:
            output = model(sample_batched,mode=mode)

    result = model.get_result(output,index=0)

    if 'crop_bbox' in sample_cpy['meta']:
        sample['meta']['pre_crop_bbox']= sample_cpy['meta']['crop_bbox']
        result=helpers.recover_from_bbox(result,sample_cpy['meta']['crop_bbox'],img_size[::-1])
    else:
        result = cv2.resize(result,img_size, interpolation=cv2.INTER_LINEAR)

    pred = (result>p['pred_tsh']).astype(np.uint8)

    sample['pre_pred']=pred
    return pred,result

# [calculate expand radius]
def cal_expand_r_new_final(pt,pre_pred,pred):
    pred_delta = np.abs(pred.astype(np.int64) - pre_pred.astype(np.int64))

    if pred_delta[pt[1],pt[0]]==0:
        d=distance_transform_cdt((pre_pred  if pre_pred[pt[1],pt[0]]==1 else (1-pre_pred)))[pt[1],pt[0]]  
        if_hr=True
    else:
        mask = np.zeros((pred.shape[0] + 2, pred.shape[1] + 2), np.uint8)
        pred_delta_new=pred_delta.astype(np.uint8)
        cv2.floodFill(pred_delta_new, mask, (pt[0],pt[1]),2)
        field = (pred_delta_new == 2).astype(np.uint8)
        k = pred.sum()
        q = field.sum()
        d=  np.max(np.abs(np.argwhere(field>0)[:,::-1]- np.array([pt[:2]])))
        if_hr=(q < 0.2 * k)

    d=max(d,5)
    return d,if_hr

# [calculate the ratio of the next expand radius]
def get_new_ratio_final(pre_pred,pred,tsh=5):
    r=pre_pred.shape[0]//2
    delta=np.uint8(np.abs(pre_pred.astype(np.int64)-pred.astype(np.int64)))
    dist=distance_transform_edt(delta)
    tsh=pre_pred.shape[0]*tsh/384.0
    if dist.max()>=tsh:
        dist_max=np.max(np.abs(np.argwhere(dist>=tsh)-np.array([[r,r]])))
        return (dist_max/r)*1.1
    else:
        return None

# [predict for the local view]
def predict_hr_new_final(p,model,sample,seq_points,hr_points=[],mode='eval-hr',pred=None,result=None):
    initial_ratio=1.75 if p['hr_val_setting']['fv']<=0 else 1.00
    turn_num = int(abs(p['hr_val_setting']['pfs']))
    img_size=sample['img'].shape[:2][::-1]
    hr_point_idx_ignore=[]
    scale_ratios=[initial_ratio]*len(hr_points)

    for turn in range(turn_num):
        if len(hr_points)>0:
            sample_hrs=[]
            for hr_point_idx,hr_point in enumerate(hr_points):
                pt_hr=hr_point['pt_hr']
                scale_ratio=scale_ratios[hr_point_idx]
                if scale_ratio is None:continue
                expand_r=int(hr_point['expand_r']*scale_ratio)
                if expand_r<=5:continue
                sample_hr = deepcopy(sample)
                sample_hr['seq_points']=seq_points.copy()
                sample_hr['center_point']=np.array([pt_hr])
                sample_hr['pre_pred']=pred.copy()
                crop_bbox=(pt_hr[0],pt_hr[1],2*expand_r+1,2*expand_r+1)
                mtr.Crop(crop_bbox,pad=expand_r,elems_point=['seq_points','center_point'])(sample_hr)
                if turn==0:
                    if hr_point['pre_pred_hr'] is not None:
                        miou=helpers.compute_iou(sample_hr['pre_pred'],hr_point['pre_pred_hr'])
                        if miou>100000 and len(sample_hr['seq_points_hr'])==len(hr_point['seq_points_hr']) :
                            hr_point_idx_ignore.append(hr_point_idx)
                            continue        
                    hr_point['pre_pred_hr']=sample_hr['pre_pred'].copy()
                    hr_point['seq_points_hr']=sample_hr['seq_points'].copy()
                    hr_point['img_hr']=sample_hr['img'].copy()
                    hr_point['crop_bbox_hr']=sample_hr['meta']['crop_bbox'].copy()
                    hr_point['expand_r_real']=deepcopy(expand_r)
                    if 'gt' in sample_hr.keys():
                        hr_point['gt_hr'] =sample_hr['gt'].copy()

                if hr_point_idx in hr_point_idx_ignore:continue
                mtr.Resize(abs(p['hr_size']),elems_point=['seq_points','center_point'])(sample_hr)
                sample_hr['img_hr'] = sample_hr.pop('img')
                if 'gt' in sample_hr.keys(): sample_hr['gt_hr'] = sample_hr.pop('gt')
                sample_hr['seq_points_hr'] = sample_hr.pop('seq_points')
                sample_hr['center_point_hr'] = sample_hr.pop('center_point')
                sample_hr['pre_pred_hr'] = sample_hr.pop('pre_pred')
                mtr.HRCatPointMask(mode=p['point_map_mode'])(sample_hr)
                mtr.ToTensor(div_list=['img_hr'],elems_undo=['seq_points_hr','center_point_hr'])(sample_hr)
                mtr.Normalize( mean_std= None if p['no_norm'] else [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]],elems_do=['img_hr'])(sample_hr)
                sample_hr['meta']['hr_point_idx']=hr_point_idx
                sample_hrs.append(sample_hr)

            if len(sample_hrs)>0:
                sample_batched_hr=default_collate(sample_hrs)
                with torch.no_grad(): output = model(sample_batched_hr,mode='eval-hr')
            
                hr_results = model.get_result(output,index=None)
                for hr_result, sample_hr in zip(hr_results,sample_hrs):
                    hr_point_idx=sample_hr['meta']['hr_point_idx']
                    pt_hr =  hr_points[hr_point_idx]['pt_hr']

                    scale_ratio=scale_ratios[hr_point_idx]
                    expand_r = int(hr_points[hr_point_idx]['expand_r']*scale_ratio)
                    
                    hr_result_count=np.ones_like(hr_result)
                    crop_bbox=(pt_hr[0],pt_hr[1],2*expand_r+1,2*expand_r+1)
                    hr_points[hr_point_idx]['hr_result_src']=helpers.recover_from_bbox(hr_result,crop_bbox,np.array(img_size[::-1])+2*expand_r)[expand_r:-expand_r,expand_r:-expand_r]
                    hr_points[hr_point_idx]['hr_result_count_src']=helpers.recover_from_bbox(hr_result_count,crop_bbox,np.array(img_size[::-1])+2*expand_r)[expand_r:-expand_r,expand_r:-expand_r]

                    pfs_d=math.modf(abs(p['hr_val_setting']['pfs']))[0]

                    if p['hr_val_setting']['if_fast']:
                        assert((p['hr_val_setting']['pfs']<0) and (pfs_d>0.01))
                        ratio=pfs_d
                    else:
                        ratio = get_new_ratio_final(np.uint8(sample_hr['pre_pred_hr'].numpy()[0]>0.5),np.uint8(hr_result>0.5),tsh=2.0)

                        if ratio is not None and pfs_d>0.01:
                            ratio=pfs_d

                    scale_ratios[hr_point_idx] = None if ratio is None else  max((scale_ratios[hr_point_idx]*ratio),0.2)

            hr_result_src_all,hr_result_count_src_all=np.zeros(img_size[::-1],dtype=np.float64),np.zeros(img_size[::-1],dtype=np.float64)

            for hr_point in hr_points:
                hr_result_src_all+=hr_point['hr_result_src']
                hr_result_count_src_all+=hr_point['hr_result_count_src']

            hr_mask=(hr_result_count_src_all>0)
            result[hr_mask]=hr_result_src_all[hr_mask]/hr_result_count_src_all[hr_mask]

            if p['hr_val_setting']['pfs']>0:
                pred = (result>p['pred_tsh']).astype(np.uint8)

    pred = (result>p['pred_tsh']).astype(np.uint8)

    for hr_point_idx,hr_point in enumerate(hr_points):
        expand_r=hr_point['expand_r_real']
        sample_tmp={'pred':pred.copy(),'meta':{}}
        mtr.Crop(crop_bbox=hr_point['crop_bbox_hr'],pad=expand_r)(sample_tmp)
        hr_point['pred_hr']=sample_tmp['pred'].copy()

    sample['pre_pred']=pred
    return pred

























































