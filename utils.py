import argparse
import shutil
import time
import sys
import os
import torch

def para2dict(**kwargs):return kwargs

########################################[ Optimizer ]########################################

def get_optimizer(opt_info, train_params):
    parse_info=opt_info.split('-')
    opt_mode = parse_info[0]
    opt_default = {'SGD' : {'momentum':0.9, 'weight_decay':5e-4, 'nesterov':False},   
                   'Adam' : {'betas': (0.9, 0.999), 'eps': 1e-8},
                   }
    opt_kwargs = opt_default[opt_mode] if len(parse_info)==1 else eval('para2dict({})'.format(parse_info[1]))

    if opt_mode=='SGD':
        return  torch.optim.SGD(train_params,**opt_kwargs)
    elif opt_mode=='Adam':
        return torch.optim.Adam(train_params,**opt_kwargs)

########################################[ Scheduler ]########################################
from torch.optim.lr_scheduler import _LRScheduler

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, epoch_max, power=0.9, last_epoch=-1, cutoff_epoch=1000000):
        self.epoch_max,self.power,self.cutoff_epoch = epoch_max,power,cutoff_epoch
        super(PolyLR, self).__init__(optimizer, last_epoch)
    def get_lr(self):
        return [  base_lr *   ( 1-( 1.0*min(self.last_epoch,self.cutoff_epoch)/ self.epoch_max) )**self.power  for base_lr in self.base_lrs   ]
       
class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLR, self).__init__(optimizer, last_epoch)
    def get_lr(self):
        return [  base_lr * 1.0  for base_lr in self.base_lrs   ]

def get_lr_scheduler(ls_info, optimizer):
    parse_info=ls_info.split('-')
    ls_mode = parse_info[0]
    ls_default= {'poly':{'epoch_max':30,'cutoff_epoch':29},
                'exp':{'gamma':0.9},
                'multistep':{'milestones':[5,20]},
                'constant':{}}
    ls_kwargs = ls_default[ls_mode] if len(parse_info)==1 else eval('para2dict({})'.format(parse_info[1]))
    if ls_mode=='poly':
        return  PolyLR(optimizer, **ls_kwargs)
    elif ls_mode=='constant':
        return  ConstantLR(optimizer, **ls_kwargs)
    elif ls_mode=='step':
        return  torch.optim.lr_scheduler.StepLR(optimizer, **ls_kwargs)
    elif ls_mode=='multistep':
        return  torch.optim.lr_scheduler.MultiStepLR(optimizer, **ls_kwargs)
    elif ls_mode=='exp':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer,**ls_kwargs)

########################################[ Config ]########################################

def create_parser():
    parser = argparse.ArgumentParser(description="Argparse for IIS")

    #[basic]
    parser.add_argument('-n','--note', type=str, default=None)
    parser.add_argument('-t','--test', action='store_true', default=False)
    parser.add_argument('-bc','--backup_code', action='store_true', default=False)
    parser.add_argument('-cs','--clear_snapshot', action='store_true', default=False)
    
    #[path and dataset]
    parser.add_argument('-dp','--dataset_path', type=str, default='./dataset')
    parser.add_argument('-sp','--snapshot_path', type=str, default='auto')
    parser.add_argument('-dt','--dataset_train', type=str, default='SBD')
    parser.add_argument('-dv','--dataset_vals',type=str, default='GrabCut,Berkeley,DAVIS')

    #[net]
    parser.add_argument('-m','--model', type=str, default='hrcnet_base')
    parser.add_argument('-ap','--aux_parameter', default='', type=str)
    parser.add_argument('-sb','--sync_bn', action='store_true', default=False)
    parser.add_argument('-os','--output_stride', type=int, default=16)

    #[training data]
    parser.add_argument('-s','--seed', type=int, default=10)
    parser.add_argument('-nw','--num_workers', type=int, default=4)
    parser.add_argument('-mn','--max_num', type=str, default='0')
    parser.add_argument('-rs','--ref_size', type=int, default=384)
    parser.add_argument('-nn','--no_norm', action='store_true', default=False)
    parser.add_argument('-rso','--remove_small_obj', type=int, default=0)
    parser.add_argument('-dln','--data_loader_num', type=int, default=0) 

    #[training process]
    parser.add_argument('-e','--epochs', type=int, default=40)
    parser.add_argument('-bs','--batch_size', type=int, default=8)
    parser.add_argument('-lr','--learning_rate', type=float, default=7e-3)
    parser.add_argument('-sl','--special_lr', type=float, default=1.0)
    parser.add_argument('-ls','--lr_scheduler', type=str, default='exp')
    parser.add_argument('-opt','--optimizer', type=str, default='SGD')
    parser.add_argument('-gi','--gpu_ids', type=str, default=None)
    parser.add_argument('-lss','--lr_scheduler_stop', type=str, default=None)

    #[validation when training] 
    parser.add_argument('-mnrv','--max_num_robot_val', type=int, default=None)
    parser.add_argument('-toe','--train_only_epochs', type=int, default=19)
    parser.add_argument('-vri','--val_robot_interval', type=int, default=10)
    parser.add_argument('-svr','--save_val_robot', type=str, default='each',choices=['no','each','best'])

    #[validation only]
    parser.add_argument('-r','--resume', type=str, default=None)
    parser.add_argument('-v','--val', action='store_true', default=False)
    parser.add_argument('-es','--eval_size', type=int, default=None)
    parser.add_argument('-ef','--eval_flip', action='store_true', default=False) 

    #[iis]
    parser.add_argument('-ip','--itis_pro', type=float, default=0.7)
    parser.add_argument('-is','--itis_strategy', type=str, default='best_one')
    parser.add_argument('-ss','--simulate_strategy', type=str, default=None)
    parser.add_argument('-zi','--zoom_in', type=int, default=0)
    parser.add_argument('-rpn','--record_point_num', type=int, default=5)
    parser.add_argument('-mt','--miou_target', type=str, default='[0.85,0.90]')
    parser.add_argument('-pmm','--point_map_mode', type=str, default='dist_src')
    parser.add_argument('-mpn','--max_point_num', type=int, default=20)
    parser.add_argument('-om','--other_metric', type=str, default='biou,assd')

    #[aug]
    parser.add_argument('-rf','--random_flip', action='store_true', default=False)
    parser.add_argument('-rr','--random_rotate', action='store_true', default=False)
    parser.add_argument('-aug','--augmentation', type=str, default='')

    #[ui]
    parser.add_argument('-img','--image', type=str, default='./test.jpg', help='input image file')
    parser.add_argument('-gt','--groundtruth', type=str, default=None, help='input groundtruth file')
    parser.add_argument('-o','--output', type=str, default=None, help='output mask file')
    parser.add_argument('--cpu', action='store_true', default=False, help='use cpu (not recommended)')
    
    parser.add_argument('-mag','--magnifier', type=int, default=0, help='magnifier mode')
    parser.add_argument('-ms','--mag_scale', type=int, default=4, help='mag scale ')
    parser.add_argument('-fb','--focus_bbox', type=int, default=0, help='focus bbox')

    #[focuscut]
    parser.add_argument('-hrs','--hr_size', type=int, default=None)
    parser.add_argument('-hrv','--hr_val', action='store_true', default=False)
    parser.add_argument('-hr_dt','--hr_dataset_train', type=str, default='HRSOD_TRAIN')
    parser.add_argument('-hrlr','--hr_learning_rate', type=float, default=7e-4)
    parser.add_argument('-hrls','--hr_lr_scheduler', type=str, default='constant')
    parser.add_argument('-hropt','--hr_optimizer', type=str, default='SGD')
    parser.add_argument('-hrbf','--hr_backbone_frozen', action='store_true', default=False)

    parser.add_argument('-tmp','--temporary', type=str, default='a=175,b=200,c=110,d=20')

    #pfs>0 is standard version，<0 is fast version，pfs=0 is not do
    parser.add_argument('-hrvs','--hr_val_setting', type=str, default='pfs=3')
    parser.add_argument('-erm','--eval_resize_mode', type=int, default=0, help='-1= not resize,0=short ,1=fix, 2=long')

    #[for cocolvis]
    parser.add_argument('-lssi','--lr_scheduler_step_iteration', type=int, default=0)
    parser.add_argument('-ein','--epoch_iter_num', type=int, default=1) 

    args = parser.parse_args()
    p=vars(args)

    #[default]
    p['gt_mode']='0255'
    p['if_memory']=False
    p['pred_tsh']=0.5
    p['pred_tsh_itis']=0.5
    
    from importlib import import_module
    p['seq_mode']=(import_module('model.{}.my_net'.format(p['model'])).MyNet.__base__.__name__=='MyNetBaseSeq')

    #[basic]
    if p['note'] is None: p['note'] = (' '.join(sys.argv[1:])).replace('/','\\')
    if p['test']: p['ref_size'],p['batch_size'],p['max_num']=64,2,'-4'
    
    #[path and dataset]
    if p['snapshot_path']=='auto':
        p['snapshot_path']= './snapshot/{}_[{}]_[{}]'.format(p['model'],p['note'],time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time())))
    
    p['dataset_vals']=p['dataset_vals'].replace('DT',p['dataset_train']).split(',')

    #[net]
    p['aux_parameter']= eval('para2dict({})'.format(p['aux_parameter']))

    #[training data]
    p['max_num']=eval(p['max_num'])
    if p['max_num_robot_val'] is None: p['max_num_robot_val']= p['max_num']
    
    #[training process]
    p['gpu_ids']=  list(range(torch.cuda.device_count())) if p['gpu_ids'] is None else [int(t) for t in p['gpu_ids'].split(',')]
    if p['lr_scheduler_stop'] is not None: p['lr_scheduler_stop']=eval(p['lr_scheduler_stop'])


    #[validation when training]
    if p['train_only_epochs'] is None: p['train_only_epochs']=p['epochs']-1

    #[validation only]
    if p['resume'] is not None:
        if p['resume']=='auto':p['resume']=p['model']
        if not p['resume'].endswith('.pth'):p['resume']+='.pth'
        if (not os.path.exists(p['resume'])): p['resume']=os.path.join('./pretrained_model',p['resume'])

    if p['eval_size'] is None: p['eval_size']=p['ref_size']
    
    #[iis]
    miou_target_dict={'PASCAL_VOC':[0.85,0.90],'GrabCut':[0.90],'Berkeley':[0.90],'PASCAL_SBD':[0.85,0.90],'SBD':[0.85],'COCO_UNSEEN':[0.85,0.90],'COCO_SEEN':[0.85,0.90],'CoCA':[0.85,0.90],'CoSOD3k':[0.85,0.90]}
    p['miou_target']=[ (miou_target_dict[dataset] if dataset in miou_target_dict else [0.85,0.90]) if p['miou_target'] is None else eval(p['miou_target']) for  dataset in p['dataset_vals']]
    if p['simulate_strategy'] is None:p['simulate_strategy']='first' if p['seq_mode'] else 'fcanet'

    p['other_metric']=[] if p['other_metric'] is None else p['other_metric'].split(',')

    #[aug]
    if p['augmentation']=='all': p['augmentation']='bri,con,sat,hue,gam'

    #[focuscut]
    if p['hr_size'] is None: p['hr_size']=p['eval_size']

    p['temporary']= eval('para2dict({})'.format(p['temporary']))

    p['hr_val_setting']= eval('para2dict({})'.format(p['hr_val_setting']))

    # fv=-1 is adaptive
    hr_val_setting_default={'fv':-1,'pfs':3,'if_fast':False}
    for k,v in hr_val_setting_default.items():
        if k not in p['hr_val_setting']:
            p['hr_val_setting'][k]=v
    
    print('----->',p['hr_val_setting'])
    return p

########################################[ Log ]########################################

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
	    pass

def set_log_file(file_path='log'):
    sys.stdout = Logger(file_path, sys.stdout)

def print_random(num=10000):
    import random
    import numpy as np
    import torch
    print(random.randint(1,num),np.random.randint(1,num),torch.randint(num, (1,1)))
    