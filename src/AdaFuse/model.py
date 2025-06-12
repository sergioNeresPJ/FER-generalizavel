import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.parameter import Parameter

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3,
                               stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3,
                               stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace = True)

        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1,
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):

        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x

class ResNet(nn.Module):
    def __init__(self, block, n_blocks, channels, output_dim):
        super().__init__()


        self.in_channels = channels[0]

        assert len(n_blocks) == len(channels) == 4

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride = 2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride = 2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, output_dim)

    def get_resnet_layer(self, block=BasicBlock, n_blocks=[2,2,2,2], channels=[64, 128, 256, 512], stride = 1):

        layers = []

        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False

        layers.append(block(self.in_channels, channels, stride, downsample))

        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)

        return x, h

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def Mask(nb_batch):
    bar = []
    for i in range(7):
        foo = [1] * 63 + [0] * 10
        if i == 6:
            foo = [1] * 64 + [0] * 10
        random.shuffle(foo)  # generate mask
        bar += foo
    bar = [bar for _ in range(nb_batch)]
    bar = np.array(bar).astype("float32")
    bar = bar.reshape(nb_batch, 512, 1, 1)
    bar = torch.from_numpy(bar)

    # Verifica se XPU está disponível
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bar = bar.to(device)
    bar = Variable(bar)
    return bar


###### channel separation and channel diverse loss
def supervisor(x, targets, cnum):
    branch = x
    branch = branch.reshape(branch.size(0),branch.size(1), 1, 1)
    branch = my_MaxPool2d(kernel_size=(1,cnum), stride=(1,cnum))(branch)
    branch = branch.reshape(branch.size(0),branch.size(1), branch.size(2) * branch.size(3))
    loss_2 = 1.0 - 1.0*torch.mean(torch.sum(branch,2))/cnum # set margin = 3.0

    mask = Mask(x.size(0))
    branch_1 = x.reshape(x.size(0),x.size(1), 1, 1) * mask
    branch_1 = my_MaxPool2d(kernel_size=(1,cnum), stride=(1,cnum))(branch_1)
    branch_1 = branch_1.view(branch_1.size(0), -1)
    loss_1 = nn.CrossEntropyLoss()(branch_1, targets)
    return [loss_1, loss_2]


class my_MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(my_MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        input = input.transpose(3,1)


        input = F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)
        input = input.transpose(3,1).contiguous()

        return input

    def __repr__(self):
        kh, kw = _pair(self.kernel_size)
        dh, dw = _pair(self.stride)
        padh, padw = _pair(self.padding)
        dilh, dilw = _pair(self.dilation)
        padding_str = ', padding=(' + str(padh) + ', ' + str(padw) + ')' \
            if padh != 0 or padw != 0 else ''
        dilation_str = (', dilation=(' + str(dilh) + ', ' + str(dilw) + ')'
                        if dilh != 0 and dilw != 0 else '')
        ceil_str = ', ceil_mode=' + str(self.ceil_mode)
        return self.__class__.__name__ + '(' \
            + 'kernel_size=(' + str(kh) + ', ' + str(kw) + ')' \
            + ', stride=(' + str(dh) + ', ' + str(dw) + ')' \
            + padding_str + dilation_str + ceil_str + ')'


class my_AvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super(my_AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        input = input.transpose(3,1)
        input = F.avg_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)
        input = input.transpose(3,1).contiguous()
        return input


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'kernel_size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) \
            + ', ceil_mode=' + str(self.ceil_mode) \
            + ', count_include_pad=' + str(self.count_include_pad) + ')'

class Model(nn.Module):
    def __init__(self, pretrained=True, num_classes=7, drop_rate=0, model_path=None):
        super(Model, self).__init__()

        res18 = ResNet(block = BasicBlock, n_blocks = [2,2,2,2], channels = [64, 128, 256, 512], output_dim=1000)
        msceleb_model = torch.load(model_path)
        state_dict = msceleb_model['state_dict']
        res18.load_state_dict(state_dict, strict=False)

        self.drop_rate = drop_rate
        self.features = nn.Sequential(*list(res18.children())[:-2])
        self.features2 = nn.Sequential(*list(res18.children())[-2:-1])

        fc_in_dim = list(res18.children())[-1].in_features  # original fc layer's in dimention 512
        self.fc = nn.Linear(fc_in_dim, num_classes)  # new fc layer 512x7
		self.gate_layer = nn.Linear(fc_in_dim, fc_in_dim)
        self.parm={}
        for name,parameters in self.fc.named_parameters():
            print(name,':',parameters.size())
            self.parm[name]=parameters

		

    def forward(self, x, clip_model, targets, phase='train'):
        with torch.no_grad():
            image_features = clip_model.encode_image(x)

        x = self.features2(x)
		x = x.view(x.size(0), -1)
		ecnn = x
		eclip = image_features

		# Gating: g = sigmoid(W * ecnn + b)
		g = torch.sigmoid(self.gate_layer(ecnn))

		# Fusão adaptativa: zfused = eclip * g + ecnn * (1 - g)
		zfused = eclip * g + ecnn * (1 - g)

		# Perda supervisionada auxiliar
		if phase == 'train':
			MC_loss = supervisor(zfused, targets, cnum=73)

		# Classificação final
		out = self.fc(zfused)

        if phase=='train':
            return out, MC_loss
        else:
            return out, out