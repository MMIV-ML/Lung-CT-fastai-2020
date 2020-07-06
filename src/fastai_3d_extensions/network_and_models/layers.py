from fastai.vision import *

def conv_layer3d(ni,nf, kernel_size=3, stride=1, padding=1, relu=True): 
    if relu: return nn.Sequential(nn.Conv3d(ni,nf,kernel_size=kernel_size, stride=stride, padding=padding), nn.BatchNorm3d(nf), nn.ReLU())
    else: return nn.Sequential(nn.Conv3d(ni,nf,kernel_size=kernel_size, stride=stride, padding=padding),nn.BatchNorm3d(nf))
    

class ResBlock(nn.Module): 
    def __init__(self, nf):
        super().__init__()
        self.conv1 = conv_layer3d(nf,nf)
        self.conv2 = conv_layer3d(nf,nf, relu=False)
    
    def forward(self, x):
        res = x + self.conv2(self.conv1(x))
        return res

def conv_and_res3d(ni,nf):
    return nn.Sequential(conv_layer3d(ni,nf), ResBlock(nf))

class AdaptiveConcatPool3d(nn.Module):
    "Layer that concats `AdaptiveAvgPool3d` and `AdaptiveMaxPool3d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool3d(self.output_size)
        self.mp = nn.AdaptiveMaxPool3d(self.output_size)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class ScaledSigmoid(nn.Module):
    def __init__(self, ymin, ymax):
        super().__init__()
        self.ymin = ymin
        self.ymax = ymax
        
    def forward(self, input): return torch.sigmoid(input) * (self.ymax - self.ymin) + self.ymin


