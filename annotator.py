
import cv2
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.ndimage.morphology import distance_transform_edt
from torch.utils.data.dataloader import default_collate

import torch
import utils
import helpers
import my_custom_transforms as mtr
from model.general.sync_batchnorm import patch_replication_callback

import inference

########################################[ Interface ]########################################

def init_model(p):
    model = CutNet(output_stride=p['output_stride'],if_sync_bn=p['sync_bn'],special_lr=p['special_lr'],size=abs(p['ref_size']),aux_parameter=p['aux_parameter'])
    if not p['cpu']: model=model.cuda()
    model.eval()

    if p['resume'] is not None:
        state_dict=torch.load(p['resume'])  if (not p['cpu']) else  torch.load(p['resume'] ,map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        print('load from [{}]!'.format(p['resume'] ))
    return model

class Annotator(object):
    def __init__(self,p):
        self.p=p
        self.save_path=p['output']
        self.if_cuda=not p['cpu']
        self.model=init_model(p)
        self.file = Path(p['image']).name
        self.img = np.array(Image.open(p['image']).convert('RGB'))
        self.__reset()

    def __gene_merge(self,pred,img,clicks,r=0,cb=2,b=0,if_first=False):
        pred_mask=cv2.merge([pred*255,pred*255,np.zeros_like(pred)])
        result= np.uint8(np.clip(img*0.7+pred_mask*0.3,0,255))
        if b>0:
            contours,_=cv2.findContours(pred,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result,contours,-1,(255,255,255),b)

        if r>0:
            for pt in clicks:
                cv2.circle(result,tuple(pt[:2]),r,(255,0,0) if pt[2]==1 else (0,0,255),-1)
                cv2.circle(result,tuple(pt[:2]),r,(255,255,255),cb) 
        
            if if_first and len(clicks)!=0:
                cv2.circle(result,tuple(clicks[0,:2]),r,(0,255,0),cb) 
        return result

    def __update(self):
        self.ax.imshow(self.merge)
        for t in self.point_show:t.remove()
        self.point_show=[]
        if len(self.clicks)>0:
            pos_clicks=self.clicks[self.clicks[:,2]==1,:]
            neg_clicks=self.clicks[self.clicks[:,2]==0,:]
            if len(pos_clicks)>0:self.point_show.append(self.ax.scatter(pos_clicks[:,0],pos_clicks[:,1],color='red'))
            if len(neg_clicks)>0:self.point_show.append(self.ax.scatter(neg_clicks[:,0],neg_clicks[:,1],color='green'))
        self.fig.canvas.draw()

    def __reset(self):
        self.clicks =  np.empty([0,3],dtype=np.int64)
        self.pred = np.zeros(self.img.shape[:2],dtype=np.uint8)
        self.merge =  self.__gene_merge(self.pred, self.img, self.clicks)
        self.sample={'img':self.img,'pre_pred':self.pred}
        self.sample['meta']={'id': str(Path(self.p['image']).stem),'img_path' : self.p['image'], 'source_size':np.array(self.img.shape[:2][::-1])} 
        self.sample_backup=[]
        self.hr_points=[]
        self.point_show=[]

    def __predict(self):
        if len(self.clicks)==1 and self.p['zoom_in']==0: inference.predict_wo(self.p,self.model,self.sample,self.clicks)
        pred_tmp,result_tmp = inference.predict_wo(self.p,self.model,self.sample,self.clicks)

        if len(self.clicks)>1 and self.p['model'].startswith('hrcnet') and self.p['hr_val']:
            expand_r,if_hr=inference.cal_expand_r_new_final(self.clicks[-1],self.pred,pred_tmp)
            if if_hr: 
                hr_point={'point_num':len(self.clicks),'pt_hr':self.clicks[-1],'expand_r':expand_r,'pre_pred_hr':None,'seq_points_hr':None,'hr_result_src':None,'hr_result_count_src':None}
                self.hr_points.append(hr_point)

        self.pred= inference.predict_hr_new_final(self.p,self.model,self.sample,self.clicks,self.hr_points,pred=pred_tmp,result=result_tmp) if len(self.hr_points)>0 else pred_tmp

        self.merge =  self.__gene_merge(self.pred, self.img, self.clicks)
        self.__update()

    def __on_key_press(self,event):
        if event.key=='ctrl+z':
            if len(self.clicks)<=1:
                self.__reset()
                self.__update()
            else:
                self.clicks=self.clicks[:-1,:]
                self.sample_backup.pop()
                self.sample=deepcopy(self.sample_backup[-1])
                self.__predict()

        elif event.key=='ctrl+r':
            self.__reset()
            self.__update()
        elif event.key=='escape':
            plt.close()
        elif event.key=='enter':
            if self.save_path is not None:
                Image.fromarray(self.pred*255).save(self.save_path)
                print('save mask in [{}]!'.format(self.save_path))
            plt.close()

    def __on_button_press(self,event):
        if (event.xdata is None) or (event.ydata is None):return
        if event.button==1 or  event.button==3:
            x,y= int(event.xdata+0.5), int(event.ydata+0.5)
            self.clicks=np.append(self.clicks,np.array([[x,y,(3-event.button)/2]],dtype=np.int64),axis=0)
            self.__predict()
            self.sample_backup.append(deepcopy(self.sample))

    def main(self):
        self.fig = plt.figure('Annotator')
        self.fig.canvas.mpl_connect('key_press_event', self.__on_key_press)
        self.fig.canvas.mpl_connect("button_press_event",  self.__on_button_press)
        self.fig.suptitle('( file : {} )'.format(self.file),fontsize=16)
        self.ax = self.fig.add_subplot(1,1,1)
        self.ax.axis('off')
        self.ax.imshow(self.merge)
        plt.show()


if __name__ == "__main__":
    p=utils.create_parser()
    exec('from model.{}.my_net import MyNet as CutNet'.format(p['model']))
    anno=Annotator(p)
    anno.main()
