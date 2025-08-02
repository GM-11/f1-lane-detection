import torch.nn as nn
import torch
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class LaneDetectionUNet(nn.Module):
    def __init__(self):
        super(LaneDetectionUNet, self).__init__()

        # encoder
        self.down1 = DoubleConv(3, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.down2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.down3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.down4 = DoubleConv(64, 128)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.down5 = DoubleConv(128, 256)
        self.pool5 = nn.MaxPool2d(kernel_size=2)

        self.down6 = DoubleConv(256, 512)
        self.pool6 = nn.MaxPool2d(kernel_size=2)

        # bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # decoder
        self.up_conv6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_conv6 = DoubleConv(1024, 512)

        self.up_conv5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv5 = DoubleConv(512, 256)

        self.up_conv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv4 = DoubleConv(256, 128)

        self.up_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv3 = DoubleConv(128, 64)

        self.up_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv2 = DoubleConv(64, 32)

        self.up_conv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec_conv1 = DoubleConv(32, 16)

        # final output layer
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # encoder
        x1 = self.down1(x)
        p1 = self.pool1(x1)

        x2 = self.down2(p1)
        p2 = self.pool2(x2)

        x3 = self.down3(p2)
        p3 = self.pool3(x3)

        x4 = self.down4(p3)
        p4 = self.pool4(x4)

        x5 = self.down5(p4)
        p5 = self.pool5(x5)

        x6 = self.down6(p5)
        p6 = self.pool6(x6)

        # bottleneck
        bottleneck = self.bottleneck(p6)

        # decoder
        up6 = self.up_conv6(bottleneck)
        x6 = torch.cat([up6, x6], dim=1)
        x6 = self.dec_conv6(x6)

        up5 = self.up_conv5(x6)
        x5 = torch.cat([up5, x5], dim=1)
        x5 = self.dec_conv5(x5)

        up4 = self.up_conv4(x5)
        x4 = torch.cat([up4, x4], dim=1)
        x4 = self.dec_conv4(x4)

        up3 = self.up_conv3(x4)
        x3 = torch.cat([up3, x3], dim=1)
        x3 = self.dec_conv3(x3)

        up2 = self.up_conv2(x3)
        x2 = torch.cat([up2, x2], dim=1)
        x2 = self.dec_conv2(x2)

        up1 = self.up_conv1(x2)
        x1 = torch.cat([up1, x1], dim=1)
        x1 = self.dec_conv1(x1)

        # final output
        return self.final_conv(x1)
