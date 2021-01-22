import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(w):
    """Normalizes weight tensor over full filter."""
    return F.normalize(w.view(w.shape[0], -1)).view_as(w)

class DiracConv(nn.Module):

    def init_params(self, out_channels):
        self.alpha = nn.Parameter(torch.Tensor(out_channels).fill_(1))
        self.beta = nn.Parameter(torch.Tensor(out_channels).fill_(0.1))
        self.register_buffer('delta', dirac_(self.weight.data.clone()))
        assert self.delta.shape == self.weight.shape
        self.v = (-1,) + (1,) * (self.weight.dim() - 1)

    def transform_weight(self):
        return self.alpha.view(*self.v) * self.delta + self.beta.view(*self.v) * normalize(self.weight)


class DiracConv2d(nn.Conv2d, DiracConv):
    """Dirac parametrized convolutional layer.
    Works the same way as `nn.Conv2d`, but has additional weight parametrizatoin:
        :math:`\alpha\delta + \beta W`,
    where:
        :math:`\alpha` and :math:`\beta` are learnable scalars,
        :math:`\delta` is such a tensor so that `F.conv2d(x, delta) = x`, ie
            Kroneker delta
        `W` is weight tensor
    It is user's responsibility to set correcting padding. Only stride=1 supported.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)
        self.init_params(out_channels)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, input):
        o = self.bn(input)
        o = F.conv2d(input, self.transform_weight(), self.bias, self.stride, self.padding, self.dilation)
        o = F.relu(o)
        return o



class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups=1):
        super().__init__()

        self.w1x1 = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        self.w3x3 = nn.Parameter(torch.randn(out_channels, in_channels, 3, 3))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_normal_(self.w3x3.data)
        nn.init.kaiming_normal_(self.w1x1.data)

        self.s3 = nn.Parameter(torch.zeros((1,), dtype=torch.float32))

        self.stride = stride
        if in_channels == out_channels:
            weye = self._pad_1x1_to_3x3_tensor(torch.eye(out_channels, in_channels)[:,:,None,None])
            self.register_buffer('weye', weye)
            self.s1 = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
        else:
            self.weye = None

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _get_weight(self):
        w1 = normalize(self.w1x1)
        w3 = normalize(self.w3x3)

        w1 = self._pad_1x1_to_3x3_tensor(w1)
        weight = self.s3 * w3 
        if self.weye is None:
            weight += w1
        else:
            weight += self.s1 * w1 #add zero*w1 only if we also add identity
            weight += self.weye
       
        return weight

    def forward(self, x):
        weight = self._get_weight()
        return F.relu(F.conv2d(x, weight, self.bias, self.stride, 1))



class RepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None):
        super().__init__()

        assert len(width_multiplier) == 4

        self.override_groups_map = override_groups_map or dict()

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        # self.stage0 = nn.Sequential(RepVGGBlock(3, self.in_planes, 1),
        #                             RepVGGBlock(self.in_planes, self.in_planes, 2))
        self.stage0 = RepVGGBlock(3, self.in_planes, 2),
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)


    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, 
                                      stride=stride, groups=cur_groups))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}

def create_RepVGG_A0(num_classes=10):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None)


def test_block():
    layer = RepVGGBlock(16, 64, 1)
    x = torch.randn(4,16,64,64)
    y = layer(x)
    print(y.shape)

def test_vgg():
    net = create_RepVGG_A0(10)
    x = torch.randn(5, 3, 32, 32)

    y = net(x)
    print(y.shape)

if __name__ == '__main__':
    test_vgg()


