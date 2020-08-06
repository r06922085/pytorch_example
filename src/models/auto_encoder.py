import troch.nn as nn

'''
We show some basic knowledge of building neural network by pytorch,
Sometimes, it has several ways to achieve a goal,
so I try my best to use as much as method in different places.
For example, I use both of nn.Module and nn.Sequential, but they achieve the same goal.
'''

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_module('conv1', BlockDown(3, 10))
        self.add_module('conv2', BlockDown(10, 10))
        self.add_module('conv3', BlockDown(10, 10))

    def forward(self, x):
        x = _modules['conv1'](x)
        x = _modules['conv2'](x)
        x = _modules['conv3'](x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_module('deconv1', BlockUp(10, 10))
        self.add_module('deconv2', BlockUp(10, 10))
        self.add_module('deconv3', BlockUp(10, 3))

    def forward(self):
        x = _modules['deconv1'](x)
        x = _modules['deconv2'](x)
        x = _modules['deconv3'](x)
        return x

class BlockDown(nn.Sequential):
    def __init__(self, in_dim, out_dim, kernel_size=3, strides=1, padding=1, dilation_rate=1):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_dim, out_dim, kernel_size, strides, padding, dilation_rate))
        self.add_module('act', nn.Leaky_ReLU(0.1))
        self.add_module('space_to_depth', SpaceToDepth(2))
        

class BlockUp(nn.Sequential):
    def __init__(self, in_dim, out_dim, up_scale=2):
        super().__init__()
        self.add_module("conv", nn.Conv2d(in_dim, out_dim*up_scale*up_scale))
        self.add_module("act", nn.Leaky_ReLU(0.1))
        self.add_module('pixel_shuffle', nn.pixel_shuffle(up_scale))


class SpaceToDepth(nn.Module):
    '''
    This is the function that make feature map bigger,
    You can see this as a opposite function of pixel_shuffle,
    However, pytorch don't have this function, so we need to implement it by ourselves
    For more detail, check the diary document
    '''
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs**2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x
