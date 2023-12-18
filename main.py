#[General module]
import os
import time
import random
import shutil
import numpy as np
from tqdm import tqdm

#[Other module]
import torch
from PIL import Image

#[Personal module]
import utils
import helpers
import my_custom_transforms as mtr
from dataloader_cut import GeneralCutDataset
from model.general.sync_batchnorm import patch_replication_callback
import inference 

#[Basic setting]
TORCH_VERSION=torch.__version__
torch.backends.cudnn.benchmark = True 
np.set_printoptions(precision=3, suppress=True)

#[Trainer]
class Trainer(object):
    def __init__(self,p):
        self.p=p

        self.transform_train = mtr.IIS(p)

        self.train_set = GeneralCutDataset(os.path.join(p['dataset_path'],p['dataset_train']),list_file='train.txt',max_num=p['max_num'],
                                           batch_size=-p['batch_size'], remove_small_obj=p['remove_small_obj'],gt_mode=p['gt_mode'],transform=self.transform_train, if_memory=p['if_memory'])

        self.val_robot_sets = [ GeneralCutDataset(os.path.join(p['dataset_path'],dataset_val),list_file='val.txt',max_num=p['max_num_robot_val'],batch_size=0, 
                                remove_small_obj=0,gt_mode=p['gt_mode'],transform=None, if_memory=p['if_memory']) for dataset_val in p['dataset_vals'] ]
        
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=p['batch_size'], shuffle=True, num_workers=p['num_workers'])

        self.model = CutNet(output_stride=p['output_stride'],if_sync_bn=p['sync_bn'],special_lr=p['special_lr'],size=abs(p['ref_size']),aux_parameter=p['aux_parameter']).cuda()

        if len(p['gpu_ids'])>1:
            self.model = (torch.nn.DataParallel(self.model, device_ids = p['gpu_ids']))
            patch_replication_callback(self.model)
            self.model_src=self.model.module
        else:
            self.model_src=self.model

        self.optimizer = utils.get_optimizer(p['optimizer'], self.model_src.get_train_params_lr(lr=p['learning_rate']))
        self.scheduler = utils.get_lr_scheduler(p['lr_scheduler'], self.optimizer)
        self.best_metric = None

        if p['resume'] is not None:
            self.model.load_state_dict(torch.load(p['resume']))
            print('Load model from [{}]!'.format(p['resume']))

    def training(self, epoch): 
        print('Training :')
        mtr.current_epoch = epoch
        loss_total,loss_show =0,'Loss: None'
        self.model.train()

        if self.p['hr_backbone_frozen']:self.model_src.freeze_main_bn()    

        tbar = tqdm(self.train_loader)
        for i, sample_batched in enumerate(tbar):
            self.optimizer.zero_grad()

            if self.p['seq_mode']:
                output,loss_items=self.model(sample_batched)
            else:
                if self.p['model'].startswith('hrcnet'):
                    output,loss_items=self.model(sample_batched)
                else:
                    output=self.model(sample_batched)
                    loss_items=self.model_src.get_loss_union(output,sample_batched=sample_batched)

            loss_total+=loss_items.mean(dim=0).cpu().numpy()
            self.optimizer.step()
            
            loss_show='Loss: {:.3f}{}'.format( loss_total.sum()/(i + 1), '' if len(loss_total)==1 else loss_total / (i + 1))
            tbar.set_description(loss_show)

            if (self.p['itis_pro']>0) and (not self.p['seq_mode']):
                preds = np.uint8(self.model_src.get_result(output)>self.p['pred_tsh_itis'])
                for j in range(sample_batched['img'].shape[0]):
                    id=sample_batched['meta']['id'][j]
                    mtr.record_itis['pred'][id]= helpers.encode_mask(preds[j,:,:])
                    seq_points=sample_batched['seq_points'][j].numpy()
                    mtr.record_itis['seq_points'][id]= seq_points[seq_points[:,2]!=-1].tolist()
                    mtr.record_itis['crop_bbox'][id]= list(sample_batched['meta']['crop_bbox'][j].numpy())

                    if self.p['random_flip']:mtr.record_itis['if_flip'][id] = int(sample_batched['meta']['if_flip'][j])
                    if self.p['random_rotate']:mtr.record_itis['rotate'][id] = int(sample_batched['meta']['rotate'][j])

        print(loss_show)

    def validation_robot(self, epoch,if_hrv=False):
        torch.backends.cudnn.benchmark = False
        print('+'*79)
        print('if_hrv : ',if_hrv)
        self.model.eval()
        for index, val_robot_set in enumerate(self.val_robot_sets):
            dataset=self.p['dataset_vals'][index]

            if dataset=='DAVIS':
                self.p['eval_size']=512
                self.p['hr_size']=512
            else:
                self.p['eval_size']=self.p['ref_size']
                self.p['hr_size']=self.p['ref_size']

            print('Validation Robot: [{}]'.format(dataset))
            max_miou_target=max(self.p['miou_target'][index])

            record_other_metrics = {}
    
            record_dict={}
            for i, sample in enumerate(tqdm(val_robot_set)):
                id = sample['meta']['id']
                gt = np.array(Image.open(sample['meta']['gt_path']))
                pred = np.zeros_like(gt) 
                seq_points=np.empty([0,3],dtype=np.int64)
                id_preds,id_ious=[helpers.encode_mask(pred)],[0.0]

                id_other_metrics= {metric: [0.0] for metric in self.p['other_metric']}

                hr_points=[]
                sample['pre_pred']=pred

                if self.p['zoom_in']==0:inference.predict_wo(self.p,self.model_src,sample,np.array([helpers.get_next_anno_point(np.zeros_like(gt), gt)],dtype=np.int64)) #add -wo

                for point_num in range(1, self.p['max_point_num']+1):
                    pt_next = helpers.get_next_anno_point(pred, gt, seq_points)
                    seq_points=np.append(seq_points,[pt_next],axis=0)
                    pred_tmp,result_tmp = inference.predict_wo(self.p,self.model_src,sample,seq_points)
                    if point_num>1 and self.p['model'].startswith('hrcnet') and if_hrv and p['hr_val_setting']['pfs']!=0:
                        expand_r,if_hr=inference.cal_expand_r_new_final(pt_next,pred,pred_tmp)
                        if if_hr: 
                            hr_point={'point_num':point_num,'pt_hr':pt_next,'expand_r':expand_r,'pre_pred_hr':None,'seq_points_hr':None,'hr_result_src':None,'hr_result_count_src':None,'img_hr':None,'pred_hr':None,'gt_hr':None}
                            hr_points.append(hr_point)

                    pred= inference.predict_hr_new_final(self.p,self.model_src,sample,seq_points,hr_points,pred=pred_tmp,result=result_tmp) if len(hr_points)>0 else pred_tmp

                    for metric in id_other_metrics: id_other_metrics[metric].append(helpers.get_metric(pred,gt,metric))

                    miou = ((pred==1)&(gt==1)).sum()/(((pred==1)|(gt==1))&(gt!=255)).sum()
                    id_ious.append(miou)
                    id_preds.append(helpers.encode_mask(pred))
                    if (np.array(id_ious)>=max_miou_target).any() and point_num>=self.p['record_point_num']:break

                record_dict[id]={'clicks':[None]+[tuple(pt) for pt in seq_points],'preds':id_preds,'ious':id_ious}
                record_other_metrics[id]=id_other_metrics

            # #[used for record result file]
            # if self.p['record_point_num']>5:
            #     np.save('{}/{}~{}~{}~infos.npy'.format(self.p['snapshot_path'],'FocusCut',dataset,'val'),record_dict,allow_pickle=True)

            for size_key,size_ids in val_robot_set.get_size_div_ids(size_div=[0.045,0.230][:] if dataset in ['TBD'] else None).items():
                print('[{}]({}):'.format(size_key,len(size_ids)))
                if len(size_ids)==0: print('N/A');continue
                noc_miou=helpers.get_noc_miou(record_dict,ids=size_ids,point_num=self.p['record_point_num'])
                mnoc=helpers.get_mnoc(record_dict,ids=size_ids,iou_targets=self.p['miou_target'][index])
                print('NoC-mIoU : [{}]'.format(' '.join(['{:.3f}'.format(t) for t in noc_miou ])))
                print('mNoC : {}'.format('  '.join(['{:.3f} (@{:.2f})'.format(t1,t2) for t1,t2 in zip(mnoc,self.p['miou_target'][index])])))
        
                nof=helpers.get_nof(record_dict,ids=size_ids,iou_targets=self.p['miou_target'][index],max_point_num=self.p['max_point_num'])
                print('NoF : {}'.format('  '.join(['{} (@{:.2f})'.format(t1,t2) for t1,t2 in zip(nof,self.p['miou_target'][index])])))

            for metric in self.p['other_metric']:
                metric_mean=np.array([v[metric][:self.p['record_point_num']+1] for v in record_other_metrics.values()]).mean(axis=0)
                print('{} : {}'.format(metric,metric_mean)) 

            if index==0:current_metric=[mnoc[0],noc_miou]

        torch.backends.cudnn.benchmark = True 
        return current_metric


if __name__ == "__main__": 
    p=utils.create_parser()

    random.seed(p['seed'])
    np.random.seed(p['seed'])
    torch.manual_seed(p['seed'])

    exec('from model.{}.my_net import MyNet as CutNet'.format(p['model']))
   
    if p['clear_snapshot']:shutil.rmtree('./snapshot');exit()
    os.makedirs(p['snapshot_path'],exist_ok=False)
    if p['backup_code']: shutil.copytree('.', '{}/code'.format(p['snapshot_path']), ignore=shutil.ignore_patterns('snapshot','__pycache__'))
    utils.set_log_file('{}/log.txt'.format(p['snapshot_path']))
    start_time=time.time()
    print('Start time : ',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(start_time)))
    print('---[   Note:({})    ]---'.format(p['note']))
    print('Using net : [{}]'.format(p['model']))
    print('-'*79,'\ninfos : ' , p, '\n'+'-'*79)

    mine =Trainer(p)

    if p['val']:
        mine.validation_robot(0,p['hr_val'])
    else:
        if TORCH_VERSION[0]=='0': mine.scheduler.step()
        for epoch in range(p['epochs']):
            lr_str = ['{:.7f}'.format(i) for i in mine.scheduler.get_lr()]
            print('-'*79+'\n'+'Epoch [{:03d}]=>  |-lr:{}-|  ({})\n'.format(epoch, lr_str,p['note']))

            #training
            if p['train_only_epochs']>=0:
                mine.training(epoch)

                if (p['lr_scheduler_stop'] is None) or (isinstance(p['lr_scheduler_stop'],int) and epoch<p['lr_scheduler_stop']) or (isinstance(p['lr_scheduler_stop'],float) and mine.scheduler.get_lr()[0]>p['lr_scheduler_stop']):
                    mine.scheduler.step()

            if epoch<p['train_only_epochs']: continue

            #validation-robot 
            if (epoch+1) % p['val_robot_interval']==0:
                if p['save_val_robot']=='each': torch.save(mine.model_src.state_dict(), '{}/model-epoch-{}.pth'.format(p['snapshot_path'],str(epoch).zfill(3)))
                current_metric=mine.validation_robot(epoch,False)
                current_metric=mine.validation_robot(epoch,True)
                if p['save_val_robot']=='best' and (mine.best_metric is None or current_metric[0]<mine.best_metric[0]):
                    mine.best_metric=current_metric
                    torch.save(mine.model_src.state_dict(), '{}/best.pth'.format(p['snapshot_path']))

    end_time=time.time()
    print('-'*79)
    print('End time : ', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(end_time)))
    delta_time=int(end_time-start_time)
    print('Delta time :   {}:{}:{}'.format(delta_time//3600,(delta_time%3600)//60,(delta_time%60)))
    print('Saved in [{}]({})!'.format(p['snapshot_path'],p['note']))



















