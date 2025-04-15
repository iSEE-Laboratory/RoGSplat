import torch.nn as nn
import spconv.pytorch as spconv

class SPConv(nn.Module):
    def __init__(self):
        super(SPConv, self).__init__()

        self.conv0 = double_conv(32, 32, 'subm0')
        self.down0 = stride_conv(32, 32, 'down0')

        self.conv1 = double_conv(32, 32, 'subm1')
        self.down1 = stride_conv(32, 64, 'down1')
        
        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 128, 'down2')

        self.conv3 = triple_conv(128, 128, 'subm3')
        self.down3 = stride_conv(128, 128, 'down3')

        self.conv4 = triple_conv(128, 128, 'subm4')

    def forward(self, x):
        net = self.conv0(x)
        net = self.down0(net)

        net = self.conv1(net)
        net1 = net.dense()
        net = self.down1(net)
        
        net = self.conv2(net)
        net2 = net.dense()
        net = self.down2(net)

        net = self.conv3(net)
        net3 = net.dense()
        net = self.down3(net)

        net = self.conv4(net)
        net4 = net.dense()

        volumes = [net1,net2,net3,net4]
        # volumes = [net4]
        return volumes

class SPConv2(nn.Module):
    def __init__(self):
        super(SPConv2, self).__init__()

        self.conv0 = double_conv(128, 128, 'subm0')
        self.down0 = stride_conv(128, 128, 'down0')

        self.conv1 = double_conv(128, 128, 'subm1')
        # self.down1 = stride_conv(32, 64, 'down1')
        
        # self.conv2 = triple_conv(64, 64, 'subm2')
        # self.down2 = stride_conv(64, 128, 'down2')

        # self.conv3 = triple_conv(128, 128, 'subm3')
        # self.down3 = stride_conv(128, 128, 'down3')

        # self.conv4 = triple_conv(128, 128, 'subm4')

    def forward(self, x):
        net = self.conv0(x)
        net = self.down0(net)

        net = self.conv1(net)
        net1 = net.dense()
        # net = self.down1(net)
        
        # net = self.conv2(net)
        # net2 = net.dense()
        # net = self.down2(net)

        # net = self.conv3(net)
        # net3 = net.dense()
        # net = self.down3(net)

        # net = self.conv4(net)
        # net4 = net.dense()

        volumes = [net1]
        # volumes = [net4]
        return volumes

class SparseConvNet(nn.Module):
    def __init__(self):
        super(SparseConvNet, self).__init__()

        self.conv0 = double_conv(32, 32, 'subm0')
        self.down0 = stride_conv(32, 32, 'down0')

        self.conv1 = double_conv(32, 32, 'subm1')
        # self.down1 = stride_conv(32, 64, 'down1')

        # self.conv2 = triple_conv(64, 64, 'subm2')
        # self.down2 = stride_conv(64, 128, 'down2')

        # self.conv3 = triple_conv(128, 128, 'subm3')
        # self.down3 = stride_conv(128, 128, 'down3')

        # self.conv4 = triple_conv(128, 128, 'subm4')

    def forward(self, x):
        net = self.conv0(x)
        net = self.down0(net)

        net = self.conv1(x)
        net1 = net.dense()
        # net = self.down1(net)

        # net = self.conv2(net)
        # net2 = net.dense()
        # net = self.down2(net)

        # net = self.conv3(net)
        # net3 = net.dense()
        # net = self.down3(net)

        # net = self.conv4(net)
        # net4 = net.dense()

        volumes = [net1]
        # volumes = [net4]
        return volumes


def single_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          1,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def double_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SparseConv3d(in_channels,
                            out_channels,
                            3,
                            2,
                            padding=1,
                            bias=False,
                            indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())
