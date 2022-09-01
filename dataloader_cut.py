import os
import cv2
import random
import numpy as np
from PIL import Image
from pathlib import Path
from copy import deepcopy
from torch.utils.data import Dataset
from tqdm import tqdm

#[dataset loader]
class GeneralCutDataset(Dataset):
    def __init__(self,dataset_path, list_file='train.txt',
                max_num=0, batch_size=0, remove_small_obj=0,
                gt_mode='0255',transform=None, if_memory=False, data_loader_num=0):

        super().__init__()
        dataset_path=Path(dataset_path)
        self.img_paths, self.gt_paths = [], []
        with open(dataset_path/'list'/list_file) as f:
            ids = f.read().splitlines()

        img_suffix_ref={file.stem:file.suffix for file in (dataset_path/'img').glob('*.*')}
        gt_suffix_ref={file.stem:file.suffix for file in (dataset_path/'gt').glob('*.*')}

        if remove_small_obj!=0: 
            info_file=dataset_path/'list'/'{}_info.txt'.format(list_file.split('.')[0])
            num_origin = len(ids)
            if os.path.exists(info_file):
                with open(info_file) as f:
                    infos = f.read().splitlines()
                ids= [info.split()[0] for info in infos if int(info.split()[-1])>=remove_small_obj]
            else:
                ids  =  [id for id in tqdm(ids) if  (np.array(Image.open(dataset_path/'gt'/(id+gt_suffix_ref[id])))==1).sum()>=remove_small_obj]

            remove_num=num_origin-len(ids)
            print('Removed {} objects whose pixel num < {}!'.format(remove_num, remove_small_obj))
            print('Now have {} objects!'.format(len(ids)))
            
        for id in ids:
            self.img_paths.append(dataset_path/'img'/(id.split('#')[0]+img_suffix_ref[id.split('#')[0]]))
            self.gt_paths.append(dataset_path/'gt'/(id+gt_suffix_ref[id]))

        if  max_num!=0 :   
            indices= range(min(max_num[0],len(ids)-1), min(max_num[1],len(ids)-1)+1)      if isinstance(max_num,(list,tuple)) else (random.sample(range(len(ids)),min(max_num,len(ids))) if max_num>0 else range(min(abs(max_num),len(ids))))
            self.img_paths = [self.img_paths[i] for i in indices]
            self.gt_paths = [self.gt_paths[i]  for i in indices]

        if  batch_size!=0:
            actual_num= (len(self.img_paths)//abs(batch_size) + int(batch_size<0 and (len(self.img_paths)%abs(batch_size)!=0)))*abs(batch_size)
            self.img_paths= (self.img_paths*abs(batch_size))[:actual_num]
            self.gt_paths= (self.gt_paths*abs(batch_size))[:actual_num]

        if if_memory: self.samples=[ self.get_sample(index) for index in range(len(self.img_paths))]
            
        self.gt_mode,self.transform, self.if_memory= gt_mode, transform, if_memory

        self.ids=[str(Path(gt_path).stem) for gt_path in self.gt_paths]

        self.data_loader_num=data_loader_num

        self.set_index_hash()

    def __len__(self):
        return len(self.img_paths) if self.data_loader_num==0 else abs(self.data_loader_num)

    def __getitem__(self, index):
        if self.data_loader_num!=0:  
            index=random.choice(range(len(self.img_paths))) if self.index_hash is None else self.index_hash[index]
        if self.if_memory:
            return self.transform(deepcopy(self.samples[index])) if self.transform !=None else deepcopy(self.samples[index])
        else:
            return self.transform(self.get_sample(index)) if self.transform !=None else self.get_sample(index)

    def get_sample(self,index):
        img, gt = np.array(Image.open(self.img_paths[index]).convert('RGB')), np.array(Image.open(self.gt_paths[index]))
        if self.gt_mode=='01':
            gt=(gt==1).astype(np.uint8)
        elif self.gt_mode=='0255':
            gt=(gt==1).astype(np.uint8)*255
        elif self.gt_mode=='01255':
            vp=(gt==255).astype(np.uint8)*255
            gt=(gt==1).astype(np.uint8)*255
        sample={'img':img,'gt':gt,'vp':vp} if self.gt_mode=='01255' else {'img':img,'gt':gt}
        sample['meta']={'id': str(Path(self.gt_paths[index]).stem) ,'img_path': str(self.img_paths[index]), 
                        'gt_path':str(self.gt_paths[index]), 'source_size':np.array(gt.shape[::-1])}
        return sample

    def get_size_div_ids(self,size_div=None):
        if size_div is not None and len(size_div):
            size_div=np.array(size_div+[1.0])
            size_bucket=[[] for _ in range(len(size_div))] 
            for gt_path in self.gt_paths:
                gt=np.uint8(np.array(Image.open(gt_path))==1)
                gt_bbox=cv2.boundingRect(gt)
                gt_ratio=(gt_bbox[2]*gt_bbox[3])/(gt.shape[0]*gt.shape[1])
                size_bucket[np.argmax(size_div>=gt_ratio)].append(str(Path(gt_path).stem))
            keys_ref={2:['smaller','larger'],3:['small','middle','large']}
            size_bucket_all={keys_ref[len(size_div)][i]:size_bucket[i] for i in range(len(size_div))}
        else:
            size_bucket_all={}

        size_bucket_all['all']=  [str(Path(gt_path).stem) for gt_path in self.gt_paths]
        return size_bucket_all

    def set_index_hash(self):
        self.index_hash=random.sample(range(len(self.img_paths)),abs(self.data_loader_num)) if self.data_loader_num<0 else None
