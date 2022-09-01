import torch
import torch.nn as nn
import torch.nn.functional as F
from  model.general.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import random
import numpy as np
from model.general.backbone import resnet_lz as resnet

########################################[ Global Function ]########################################

def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, SynchronizedBatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, SynchronizedBatchNorm2d):
            m.eval()
        elif isinstance(m, nn.BatchNorm2d):
            m.eval()

def gene_map_gauss(map_dist_src, sigma=10):
    return torch.exp(-2.772588722*(map_dist_src**2)/(sigma**2))

def gene_map_dist(map_dist_src, max_dist=255):
    return 1.0-map_dist_src/max_dist

def my_resize(input,ref):
    return F.interpolate(input, size=(ref if isinstance(ref,(list,tuple)) else ref.size()[2:]), mode='bilinear', align_corners=True)

def make_list(input):
    return [] if input is None else (list(input) if isinstance(input,(list,tuple)) else [input])

def point_resize(mask,ref,if_return_mask=True,tsh=0.5):
    mask_h, mask_w = mask.shape[2:4]
    ref_h, ref_w = ref if isinstance(ref,(tuple,list)) else ref.shape[2:4]
    indices=torch.nonzero(mask>tsh)
    if mask_h==ref_h and mask_w==ref_w:
        mask_new= mask
    else:
        mask_new=torch.zeros([mask.shape[0],mask.shape[1],ref_h,ref_w])
        if len(indices)!=0:
            indices = indices.float()
            indices[:,2] = torch.floor(indices[:,2]*ref_h/mask_h)
            indices[:,3] = torch.floor(indices[:,3]*ref_w/mask_w)
            indices = indices.long()
            mask_new[indices[:,0],indices[:,1],indices[:,2],indices[:,3]]=1
    return mask_new.float().cuda() if if_return_mask else indices

def to_cuda(input):
    if isinstance(input,list):
        return [tmp.cuda() for tmp in input]
    elif isinstance(input,dict):
        return {key:input[key].cuda() for key in input}
    else:
        raise ValueError('Wrong Type!')

def get_key_sample(sample_batched,keys=['img'],if_list=True,if_cuda=True):
    if not isinstance(keys,list): keys=[keys]
    key_sample=[sample_batched[key] for key in keys] if if_list else {key:sample_batched[key] for key in keys}
    if if_cuda: key_sample=to_cuda(key_sample)
    return key_sample

def set_default_dict(src_dict, default_map):
    for k,v in default_map.items():
        if k not in src_dict:
            src_dict[k]=v

def get_click_map(sample_batched,mode='dist',click_keys=None):
    mode_split=mode.split('-')
    if len(mode_split)==1:
        if_with_para=False
    else:
        if_with_para=True
        para=mode_split[-1]

    if mode.startswith('dist'):
        if click_keys is None: click_keys=['pos_map_dist_src','neg_map_dist_src']
        click_map = torch.cat([(gene_map_dist(t,int(para)) if if_with_para else gene_map_dist(t)) for t in get_key_sample(sample_batched,click_keys,if_list=True)],dim=1)
    elif mode.startswith('gauss'):
        if click_keys is None: click_keys=['pos_map_dist_src','neg_map_dist_src']
        click_map = torch.cat([(gene_map_gauss(t,int(para)) if if_with_para else gene_map_gauss(t)) for t in get_key_sample(sample_batched,click_keys,if_list=True)],dim=1)
    elif mode=='point':
        if click_keys is None: click_keys=['pos_points_mask','neg_points_mask']
        click_map = torch.cat(get_key_sample(sample_batched,click_keys,if_list=True), dim=1)
    elif mode=='first_point':
        if click_keys is None: click_keys=['first_point_mask']
        click_map = get_key_sample(sample_batched,click_keys,if_list=True)[0]
    elif mode=='first_dist':
        if click_keys is None: click_keys=['first_map_dist_src']
        click_map = gene_map_dist(get_key_sample(sample_batched,click_keys,if_list=True)[0])
    return click_map

def get_input(sample_batched,mode='dist'):
    img=get_key_sample(sample_batched,['img'],if_list=True)[0]
    if mode in ['dist','gauss','point']:
        x= torch.cat((img, get_click_map(sample_batched,mode)),dim=1)
    elif mode in ['none','img']:
        x=img
    else:
        raise ValueError('mode')
    return x

def get_stack_flip_feat(x,if_horizontal=True):
    x_flip= torch.flip(x,[3 if if_horizontal else 2]) 
    return torch.cat([x,x_flip],dim=0)

def merge_stack_flip_result(x,if_horizontal=True,batch_num=1):
    return (x[:batch_num]+ torch.flip(x[batch_num:2*batch_num],[3 if if_horizontal else 2]))/2.0


########################################[ MultiConv ]########################################

class MultiConv(nn.Module):
    def __init__(self,in_ch, channels, kernel_sizes=None, strides=None, dilations=None, paddings=None, BatchNorm=nn.BatchNorm2d,if_w_bn=True,if_end_wo_relu=False, block_kind='conv'):
        super(MultiConv, self).__init__()
        self.num=len(channels)
        if kernel_sizes is None: kernel_sizes=[ 3 for c in channels]
        if strides is None: strides=[ 1 for c in channels]
        if dilations is None: dilations=[ 1 for c in channels]
        if paddings is None: paddings = [ ( (kernel_sizes[i]//2) if dilations[i]==1 else (kernel_sizes[i]//2 * dilations[i]) ) for i in range(self.num)]
        convs_tmp=[]
        for i in range(self.num):
            if block_kind=='conv':
                if channels[i]==1 or if_end_wo_relu:
                    convs_tmp.append(nn.Conv2d( in_ch if i==0 else channels[i-1] , channels[i], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i], dilation=dilations[i]))
                else:
                    if if_w_bn:
                        convs_tmp.append(nn.Sequential(nn.Conv2d( in_ch if i==0 else channels[i-1] , channels[i], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i], dilation=dilations[i],bias=False), BatchNorm(channels[i]), nn.ReLU(inplace=True)))
                    else:
                        convs_tmp.append(nn.Sequential(nn.Conv2d( in_ch if i==0 else channels[i-1] , channels[i], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i], dilation=dilations[i],bias=False), nn.ReLU(inplace=True)))
            elif block_kind=='bottleneck':
                in_ch_cur= in_ch if i==0 else channels[i-1]
                out_ch_cur=channels[i]
                down_sample_cur = None if in_ch_cur==out_ch_cur else nn.Sequential(nn.Conv2d(in_ch_cur,out_ch_cur,kernel_size=1, stride=strides[i], bias=False),BatchNorm(out_ch_cur))
                convs_tmp.append(resnet.Bottleneck(in_ch_cur,out_ch_cur//4,strides[i],dilations[i],down_sample_cur,BatchNorm))

        self.convs=nn.Sequential(*convs_tmp)
        init_weight(self)
    def forward(self, x):
        return self.convs(x)

########################################[ MyASPP ]########################################

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm=nn.BatchNorm2d):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        init_weight(self)

    def forward(self, x):
        x = self.relu(self.bn(self.atrous_conv(x)))
        return x

class MyASPP(nn.Module):
    def __init__(self, in_ch, out_ch, dilations, BatchNorm=nn.BatchNorm2d, if_global=True):
        super(MyASPP, self).__init__()
        self.if_global = if_global

        if len(dilations)==4 and dilations[0]==1: dilations=dilations[1:]
        assert len(dilations)==3

        self.aspp1 = _ASPPModule(in_ch, out_ch, 1, padding=0, dilation=1, BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(in_ch, out_ch, 3, padding=dilations[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(in_ch, out_ch, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(in_ch, out_ch, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)

        if if_global:
            self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                nn.Conv2d(in_ch, out_ch, 1, stride=1, bias=False),
                                                BatchNorm(out_ch),
                                                nn.ReLU(inplace=True))

        merge_channel=out_ch*5 if if_global else out_ch*4

        self.conv1 = nn.Conv2d(merge_channel, out_ch, 1, bias=False)
        self.bn1 = BatchNorm(out_ch)
        self.relu = nn.ReLU(inplace=True)
        init_weight(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        if self.if_global:
            x5 = self.global_avg_pool(x)
            x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        else:
            x = torch.cat((x1, x2, x3, x4), dim=1)
    
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

########################################[ MyDecoder ]########################################

class MyDecoder(nn.Module):
    def __init__(self, in_ch, in_ch_reduce, side_ch, side_ch_reduce, out_ch, BatchNorm=nn.BatchNorm2d, size_ref='side'):
        super(MyDecoder, self).__init__()
        self.size_ref=size_ref
        self.relu = nn.ReLU(inplace=True)
        self.in_ch_reduce, self.side_ch_reduce = in_ch_reduce, side_ch_reduce

        if in_ch_reduce is not None:
            self.in_conv = nn.Sequential( nn.Conv2d(in_ch, in_ch_reduce, 1, bias=False), BatchNorm(in_ch_reduce), nn.ReLU(inplace=True))
        if side_ch_reduce is not None:
            self.side_conv = nn.Sequential( nn.Conv2d(side_ch, side_ch_reduce, 1, bias=False), BatchNorm(side_ch_reduce), nn.ReLU(inplace=True))

        merge_ch=  (in_ch_reduce if in_ch_reduce is not None else in_ch) + (side_ch_reduce if side_ch_reduce is not None else side_ch) 
        
        self.merge_conv = nn.Sequential(nn.Conv2d(merge_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(out_ch),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(out_ch),
                                       nn.ReLU(inplace=True))
        init_weight(self)

    def forward(self, input, side):
        if self.in_ch_reduce is not None:
            input=self.in_conv(input)
        if self.side_ch_reduce is not None:
            side=self.side_conv(side)

        if self.size_ref=='side':
            input=F.interpolate(input, size=side.size()[2:], mode='bilinear', align_corners=True)
        elif self.size_ref=='input':
            side=F.interpolate(side, size=input.size()[2:], mode='bilinear', align_corners=True)

        merge=torch.cat((input, side), dim=1)
        output=self.merge_conv(merge)
        return output

########################################[ PredDecoder ]########################################

class PredDecoder(nn.Module):
    def __init__(self,in_ch,layer_num=1,BatchNorm=nn.BatchNorm2d,if_sigmoid=False):
        super(PredDecoder, self).__init__()
        convs_tmp=[]
        for i in range(layer_num-1):
            convs_tmp.append(nn.Sequential(nn.Conv2d(in_ch,in_ch//2,kernel_size=3,stride=1,padding=1,bias=False), BatchNorm(in_ch//2), nn.ReLU(inplace=True)))
            in_ch=in_ch//2
        convs_tmp.append(nn.Conv2d(in_ch,1,kernel_size=1,stride=1))
        if if_sigmoid: convs_tmp.append(nn.Sigmoid())
        self.pred_conv=nn.Sequential(*convs_tmp)
        init_weight(self)
    def forward(self, input):
        return self.pred_conv(input)

########################################[ Template for IIS ]########################################

class MyNetBase(nn.Module):
    def __init__(self,special_lr=0.1,remain_lr=1.0,size=512,aux_parameter={}):
        super(MyNetBase, self).__init__()
        self.diy_lr=[]
        self.special_lr=special_lr
        self.remain_lr=remain_lr
        self.size=size
        self.aux_parameter=aux_parameter

    def get_params(self,modules):
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], SynchronizedBatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_train_params_lr(self, lr):
        train_params,special_modules=[],[]
        for modules,ratio in self.diy_lr:
            train_params.append({'params':self.get_params(modules),'lr':lr*ratio})
            special_modules+=modules
        remain_modules=[ module for module in self.children() if module not in special_modules ]
        train_params.append({'params':self.get_params(remain_modules),'lr':lr*self.remain_lr})
        return train_params
    
    #can change
    def get_loss(self, output, gt=None,sample_batched=None,others=None):
        if gt is None:gt=sample_batched['gt'].cuda()
        losses = [F.binary_cross_entropy_with_logits(my_resize(t,gt),gt) for t in make_list(output)]
        return losses

    def get_loss_union(self,output,gt=None,sample_batched=None,others=None):
        losses=self.get_loss(output,gt,sample_batched,others)
        loss_items=torch.Tensor([loss.item() for loss in losses]).unsqueeze(0)
        losses=sum(losses)
        losses.backward()
        return loss_items

    def get_result(self, output, index=None):
        result = torch.sigmoid((make_list(output))[0]).data.cpu().numpy()
        return result[:,0,:,:] if index is None else result[index,0,:,:]

    def forward_union(self, sample_batched, mode='train'):
        output=self.forward(sample_batched,mode)
        if mode=='train':
            loss_items = self.get_loss(output,sample_batched=sample_batched)
            return output,loss_items
        else:
            return output


########################################[ Template for FocusCut ]########################################

class MyNetBaseHR(MyNetBase):
    def __init__(self,input_channel=5, output_stride=16, if_sync_bn=False, if_freeze_bn=False, special_lr=0.1,remain_lr=1.0,size=512,aux_parameter={}):
        super(MyNetBaseHR, self).__init__(special_lr,remain_lr,size,aux_parameter)

        BatchNorm = SynchronizedBatchNorm2d if if_sync_bn else nn.BatchNorm2d
        default_map={'backbone':'resnet50','point_map':'gauss','into_layer':-1,'pretrained':True,'if_pre_pred':True,'weight_loss':False,'backward_each':False}
        set_default_dict(self.aux_parameter,default_map)

        side_chs=[64,256,512,1024,2048] if self.aux_parameter['backbone'] in ['resnet50','resnet101','resnet152'] else [64,64,128,256,512]
        self.backbone = resnet.get_resnet_backbone(self.aux_parameter['backbone'],output_stride,BatchNorm,self.aux_parameter['pretrained'],(3 if self.aux_parameter['into_layer']>=0 else (5+int(self.aux_parameter['if_pre_pred']))),True)
        self.my_aspp = MyASPP(in_ch=side_chs[-1],out_ch=side_chs[-1]//8,dilations=[int(i*size/512 + 0.5)*(16//output_stride) for i in [6, 12, 18]],BatchNorm=BatchNorm, if_global=True)
        self.my_decoder=MyDecoder(in_ch=side_chs[-1]//8, in_ch_reduce=None, side_ch=side_chs[-1]//8, side_ch_reduce=side_chs[1]//16*3,out_ch=side_chs[-1]//8,BatchNorm=BatchNorm)
        self.pred_decoder=PredDecoder(in_ch=side_chs[-1]//8, BatchNorm=BatchNorm)

        if self.aux_parameter['into_layer']!=-1:
            in_ch= 3 if self.aux_parameter['into_layer']==0 else side_chs[self.aux_parameter['into_layer']-1]
            self.encoder_anno=nn.Sequential( nn.Conv2d(in_ch+2+int(self.aux_parameter['if_pre_pred']),in_ch , 1, bias=False), BatchNorm(in_ch), nn.ReLU(inplace=True))

        self.diy_lr =[[[self.backbone],self.special_lr]]
        if if_freeze_bn:freeze_bn(self)

        self.side_chs=side_chs

    #return_list ['img',0,1,2,3,4,'aspp','decoder','pred_decoder']
    def backbone_forward(self,sample_batched,img_key,click_keys,pre_pred_key,return_list=['final']):
        def return_results_func(key,tmp):
            if key in return_list: return_results.append(tmp)
            return len(return_results)==len(return_list)

        return_results=[]

        img=sample_batched[img_key].cuda()
        if return_results_func('img',img): return return_results

        click_map=get_click_map(sample_batched,self.aux_parameter['point_map'],click_keys=click_keys)
        if return_results_func('click_map',click_map): return return_results

        pre_pred=sample_batched[pre_pred_key].cuda()
        if return_results_func('pre_pred',pre_pred): return return_results

        aux= torch.cat([click_map,pre_pred],dim=1)  if self.aux_parameter['if_pre_pred'] else click_map
        if return_results_func('aux',aux): return return_results

        x=img
        if self.aux_parameter['into_layer']==-1:
            x=torch.cat((x,my_resize(aux,x)),dim=1)
            
        for i in range(5):
            if i==1:x=self.backbone.layers[0][-1](x)

            if self.aux_parameter['into_layer']==i:
                x=self.encoder_anno(torch.cat((x,my_resize(aux,x)),dim=1))

            x=self.backbone.layers[i][:-1](x) if i==0 else self.backbone.layers[i](x)

            if i==1:l1=x

            if i in return_list: 
                if return_results_func(i,x):
                    return return_results 
            
        if self.aux_parameter['into_layer']==5:
            x=self.encoder_anno(torch.cat((x,my_resize(aux,x)),dim=1))

        x=self.my_aspp(x)
        if return_results_func('aspp',x): return return_results 

        x=self.my_decoder(x,l1)
        if return_results_func('decoder',x): return return_results 

        x=self.pred_decoder(x)
        if return_results_func('pred_decoder',x): return return_results

        x=my_resize(x, img)
        if return_results_func('final',x): return return_results
        

    def wo_forward(self,sample_batched):
        return self.backbone_forward(sample_batched,'img',['pos_map_dist_src','neg_map_dist_src'],'pre_pred')[0]

    def hr_forward(self,sample_batched):
        return self.backbone_forward(sample_batched,'img_hr',['pos_map_dist_src_hr','neg_map_dist_src_hr'],'pre_pred_hr')[0]

    def forward(self, sample_batched, mode='train'):
        if mode=='eval':mode='eval-wo'
        results,losses=[],[]

        for part in ['wo','hr']:
            if mode in ['train','eval-{}'.format(part)]:
                result_part=getattr(self,'{}_forward'.format(part))(sample_batched)
                if result_part is not None:
                    results.append(result_part[0] if isinstance(result_part,(list,tuple)) else  result_part)
                    if mode in ['train']:
                        loss_part=getattr(self,'get_{}_loss'.format(part))(result_part,sample_batched)
                        losses.append(loss_part)
                        if self.aux_parameter['backward_each']:
                            loss_part.backward()
        
        if mode in ['train']:
            loss_items=torch.Tensor([loss.item() for loss in losses]).unsqueeze(0).cuda()
            if not self.aux_parameter['backward_each']:
                losses_sum=sum(losses)
                losses_sum.backward()
            return results,loss_items
        else:
            return results

    def get_wo_loss(self,output_wo,sample_batched):
        gt=sample_batched['gt'].cuda()
        wo_loss=F.binary_cross_entropy_with_logits(output_wo,gt)
        return wo_loss

    def get_hr_loss(self,output_hr,sample_batched):
        weight=sample_batched['gt_weight_hr'].cuda() if self.aux_parameter['weight_loss'] else None
        gt_hr=sample_batched['gt_hr'].cuda()
        hr_loss=F.binary_cross_entropy_with_logits(output_hr,gt_hr,weight=weight)
        return hr_loss

    def get_hr_params_lr(self, lr):
        train_params=[]
        ignore_modules=[self.backbone,self.my_aspp,self.my_decoder,self.pred_decoder]
        if self.aux_parameter['into_layer']!=-1: ignore_modules.append(self.encoder_anno)
        remain_modules=[ module for module in self.children() if module not in ignore_modules ]
        train_params.append({'params':self.get_params(remain_modules),'lr':lr})
        return train_params

    def freeze_main_bn(self):
        freeze_modules=[self.backbone,self.my_aspp,self.my_decoder,self.pred_decoder]
        if self.aux_parameter['into_layer']!=-1: freeze_modules.append(self.encoder_anno)
        for module in freeze_modules:
            freeze_bn(module)


