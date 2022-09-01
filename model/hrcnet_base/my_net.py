import torch
import torch.nn as nn
import torch.nn.functional as F
from  model.general.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from  model.general.backbone import resnet_lz as resnet
from  model.general.general import *

########################################[ Net ]########################################

class MyNet(MyNetBaseHR):
    def __init__(self,input_channel=5, output_stride=16, if_sync_bn=False, if_freeze_bn=False, special_lr=0.1,remain_lr=1.0,size=512,aux_parameter={}):
        BatchNorm = SynchronizedBatchNorm2d if if_sync_bn else nn.BatchNorm2d
        set_default_dict(aux_parameter,{'into_layer':-1,'if_pre_pred':True,'backward_each':False,'backbone':'resnet50'})
        super(MyNet, self).__init__(input_channel,output_stride,if_sync_bn,if_freeze_bn,special_lr,remain_lr,size,aux_parameter)
        print(self.aux_parameter)
       
        



































    

