import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class Unet(nn.Module):
    def __init__(self, in_channel=3, rgb_dim=3, head_dim=32, encoder_dim=[32, 48, 96], decoder_dim=[48, 64, 96],
                 norm_fn='group', predict_depth=False):
        super().__init__()
        self.predict_depth = predict_depth
        self.head_dim = head_dim
        self.d_out = head_dim
        self.in_ds = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True)
        )

        self.res1 = nn.Sequential(
            ResidualBlock(32, encoder_dim[0], norm_fn=norm_fn),
            ResidualBlock(encoder_dim[0], encoder_dim[0], norm_fn=norm_fn)
        )
        self.res2 = nn.Sequential(
            ResidualBlock(encoder_dim[0], encoder_dim[1], stride=2, norm_fn=norm_fn),
            ResidualBlock(encoder_dim[1], encoder_dim[1], norm_fn=norm_fn)
        )

        self.res3 = nn.Sequential(
            ResidualBlock(encoder_dim[1], encoder_dim[2], stride=2, norm_fn=norm_fn),
            ResidualBlock(encoder_dim[2], encoder_dim[2], norm_fn=norm_fn),
        )

        self.decoder3 = nn.Sequential(
            ResidualBlock(encoder_dim[2], decoder_dim[2], norm_fn=norm_fn),
            ResidualBlock(decoder_dim[2], decoder_dim[2], norm_fn=norm_fn)
        )

        self.decoder2 = nn.Sequential(
            ResidualBlock(encoder_dim[1] + decoder_dim[2], decoder_dim[1],
                          norm_fn=norm_fn),
            ResidualBlock(decoder_dim[1], decoder_dim[1], norm_fn=norm_fn)
        )

        self.decoder1 = nn.Sequential(
            ResidualBlock(encoder_dim[0] + decoder_dim[1], decoder_dim[0],
                          norm_fn=norm_fn),
            ResidualBlock(decoder_dim[0], decoder_dim[0], norm_fn=norm_fn)
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.out_conv1 = nn.Conv2d(decoder_dim[2], head_dim, kernel_size=3, padding=1)
        self.out_conv2 = nn.Conv2d(decoder_dim[0] + rgb_dim, head_dim, kernel_size=3, padding=1)

        self.out_relu = nn.ReLU(inplace=True)
        if predict_depth:
            self.depth_head = nn.Sequential(
                nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.head_dim, 1, kernel_size=1),
                # nn.Tanh()
            )


    def forward(self, img, depth):
        if depth is not None:
            x = torch.cat([img, depth], dim=1)
        else:
            x = img
        x = self.in_ds(x)  # (32,512,512)
        x1 = self.res1(x)  # (32,512,512)
        x2 = self.res2(x1)  # (48,256,256)
        x3 = self.res3(x2)  # (96,128,128)
        
        y3 = self.decoder3(x3)  # (96,128,128)
        out1 = self.out_conv1(y3)
        out1 = self.out_relu(out1)
        up3 = self.up(y3)  # (96,256,256)

        y2 = self.decoder2(torch.cat([up3, x2], dim=1))  # (64,256,256)

        up2 = self.up(y2)  # (64,512,512)
        y1 = self.decoder1(torch.cat([up2, x1], dim=1))  # (48,512,512)
        up1 = self.up(y1)  # (48,1024,1024)

        x = torch.cat([up1, img], dim=1)
        x = self.out_conv2(x)  # (32,1024,1024)
        x = self.out_relu(x)

        depth_out = None
        if self.predict_depth:
            depth_out = self.depth_head(x)
       
        return depth_out, out1, x
    

if __name__ == '__main__':
    data = torch.ones((1, 3, 1024, 1024))

    model = Unet(in_channel=3, encoder_dim=[64, 96, 128])

    feat = model(data)
    print(feat.shape)
