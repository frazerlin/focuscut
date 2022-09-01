import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

try:
    from  model.general.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
except ImportError:
    from  torch.nn import BatchNorm2d as SynchronizedBatchNorm2d

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, input_channel=3,if_bloom=True):
        super(ResNet, self).__init__()
        self.input_channel = input_channel
        self.inplanes = 64

        if output_stride==32:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 1]
        elif output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        if if_bloom :layers[3]=[1, 2, 4] if layers[3]==3 else [1,2]

        layer0=nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
            BatchNorm(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        layer1 = self._make_layer(block, 64,  layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.layers=nn.ModuleList([layer0,layer1,layer2,layer3,layer4])

        self._init_weight()


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        if not isinstance(blocks,list):blocks=[1]*blocks
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, x, return_idx=[1,2,3,4]):
        feats=[]
        for i in range(5):
            x = self.layers[i][:-1](x) if i==0 else self.layers[i](x)
            if i in return_idx: feats.append(x)
            if i==0:x=self.layers[i][-1](x)
        
        return feats[0] if len(return_idx)==1 else feats

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self, pretrain_url):
        pretrained_model = model_zoo.load_url(pretrain_url)

        src_names=['conv1.weight','bn1.weight','bn1.bias','bn1.running_mean','bn1.running_var']
        dst_names=['layer0.0.weight','layer0.1.weight','layer0.1.bias','layer0.1.running_mean','layer0.1.running_var']

        for src_name,dst_name in  zip(reversed(src_names),reversed(dst_names)):
            pretrained_model[dst_name] = pretrained_model.pop(src_name)
            pretrained_model.move_to_end(dst_name,last=False)

        pretrained_model.pop('fc.weight')
        pretrained_model.pop('fc.bias')

        for src_name in list(pretrained_model.keys()):
            pretrained_model['layers.'+ src_name[5:]] = pretrained_model.pop(src_name)
        
        init_conv=self.state_dict()['layers.0.0.weight'].clone()
        #init_conv[:,:,:,:]=0
        init_conv[:,:3,:,:]=pretrained_model['layers.0.0.weight'].clone()
        pretrained_model['layers.0.0.weight']=init_conv

        self.load_state_dict(pretrained_model)

def ResNet18(output_stride=16, BatchNorm=nn.BatchNorm2d, pretrained=True, input_channel=3,if_bloom=True,retain_layers=[0,4]):
    model = ResNet(BasicBlock, [2, 2, 2, 2], output_stride, BatchNorm, input_channel=input_channel,if_bloom=if_bloom)
    if pretrained:model._load_pretrained_model('https://download.pytorch.org/models/resnet18-5c106cde.pth')
    model.layers=model.layers[retain_layers[0]:retain_layers[1]+1]
    return model

def ResNet34(output_stride=16, BatchNorm=nn.BatchNorm2d, pretrained=True, input_channel=3,if_bloom=True,retain_layers=[0,4]):
    model = ResNet(BasicBlock, [3, 4, 6, 3], output_stride, BatchNorm, input_channel=input_channel,if_bloom=if_bloom)
    if pretrained:model._load_pretrained_model('https://download.pytorch.org/models/resnet34-333f7ec4.pth')
    model.layers=model.layers[retain_layers[0]:retain_layers[1]+1]
    return model

def ResNet50(output_stride=16, BatchNorm=nn.BatchNorm2d, pretrained=True, input_channel=3,if_bloom=True,retain_layers=[0,4]):
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, BatchNorm, input_channel=input_channel,if_bloom=if_bloom)
    if pretrained:model._load_pretrained_model('https://download.pytorch.org/models/resnet50-19c8e357.pth')
    model.layers=model.layers[retain_layers[0]:retain_layers[1]+1]
    return model

def ResNet101(output_stride=16, BatchNorm=nn.BatchNorm2d, pretrained=True, input_channel=3,if_bloom=True,retain_layers=[0,4]):
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, input_channel=input_channel,if_bloom=if_bloom)
    if pretrained:model._load_pretrained_model('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
    model.layers=model.layers[retain_layers[0]:retain_layers[1]+1]
    return model

def ResNet152(output_stride=16, BatchNorm=nn.BatchNorm2d, pretrained=True, input_channel=3,if_bloom=True,retain_layers=[0,4]):
    model = ResNet(Bottleneck, [3, 8, 36, 3], output_stride, BatchNorm, input_channel=input_channel,if_bloom=if_bloom)
    if pretrained:model._load_pretrained_model('https://download.pytorch.org/models/resnet152-b121ed2d.pth')
    model.layers=model.layers[retain_layers[0]:retain_layers[1]+1]
    return model

def get_resnet_backbone(name='resnet101',output_stride=16, BatchNorm=nn.BatchNorm2d, pretrained=True, input_channel=3,if_bloom=True,retain_layers=[0,4]):
    if name=='resnet18':
        return ResNet18(output_stride, BatchNorm, pretrained, input_channel,if_bloom,retain_layers)
    elif name=='resnet34':
        return ResNet34(output_stride, BatchNorm, pretrained, input_channel,if_bloom,retain_layers)
    elif name=='resnet50':
        return ResNet50(output_stride, BatchNorm, pretrained, input_channel,if_bloom,retain_layers)
    elif name=='resnet101':
        return ResNet101(output_stride, BatchNorm, pretrained, input_channel,if_bloom,retain_layers)
    elif name=='resnet152':
        return ResNet152(output_stride, BatchNorm, pretrained, input_channel,if_bloom,retain_layers)

    
if __name__ == "__main__":
    import torch
    import torchvision
    
    input=torch.randn([4,3,224,224])
   
    for backbone_name in ['resnet18','resnet34','resnet50','resnet101','resnet152'][0:]:
        model=get_resnet_backbone(backbone_name,output_stride=32,pretrained=True,input_channel=3,if_bloom=False)
        outputs=model(input,return_idx=[0,1,2,3,4])

        exec('model_src=torchvision.models.{}(pretrained=True)'.format(backbone_name))
        outputs_src=[]
        outputs_src.append(model_src.relu(model_src.bn1(model_src.conv1(input))))
        outputs_src.append(model_src.layer1(model_src.maxpool(outputs_src[-1])))
        outputs_src.append(model_src.layer2(outputs_src[-1]))
        outputs_src.append(model_src.layer3(outputs_src[-1]))
        outputs_src.append(model_src.layer4(outputs_src[-1]))

        print('-'*20+backbone_name+'-'*20)
        for output,output_src in zip(outputs,outputs_src):
            print(output.shape,output.sum().detach(),output_src.shape,output_src.sum().detach())
            assert((output.shape==output_src.shape)  and (output.sum().detach()==output_src.sum().detach()))
    
    






