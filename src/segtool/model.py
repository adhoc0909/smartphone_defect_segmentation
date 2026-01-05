from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import center_crop_like, make_bilinear_upsample_weight

class DoubleConvValid(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class UNetValid(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        self.d1 = DoubleConvValid(in_channels, base_channels) # 64
        self.p1 = nn.MaxPool2d(2)

        self.d2 = DoubleConvValid(base_channels, base_channels * 2) # 128
        self.p2 = nn.MaxPool2d(2)

        self.d3 = DoubleConvValid(base_channels * 2, base_channels * 4) # 256
        self.p3 = nn.MaxPool2d(2)

        self.d4 = DoubleConvValid(base_channels * 4, base_channels * 8) # 512
        self.p4 = nn.MaxPool2d(2)

        self.b = DoubleConvValid(base_channels * 8, base_channels * 16) # 1024

        self.u4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.c4 = DoubleConvValid(base_channels * 16, base_channels * 8)

        self.u3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.c3 = DoubleConvValid(base_channels * 8, base_channels * 4)

        self.u2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.c2 = DoubleConvValid(base_channels * 4, base_channels * 2)

        self.u1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.c1 = DoubleConvValid(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        in_hw = x.shape[-2:]

        x1 = self.d1(x)
        x2 = self.d2(self.p1(x1))
        x3 = self.d3(self.p2(x2))
        x4 = self.d4(self.p3(x3))
        xb = self.b(self.p4(x4))

        y4 = self.u4(xb)
        x4c = center_crop_like(x4, y4)
        y4 = self.c4(torch.cat([x4c, y4], dim=1))

        y3 = self.u3(y4)
        x3c = center_crop_like(x3, y3)
        y3 = self.c3(torch.cat([x3c, y3], dim=1))

        y2 = self.u2(y3)
        x2c = center_crop_like(x2, y2)
        y2 = self.c2(torch.cat([x2c, y2], dim=1))

        y1 = self.u1(y2)
        x1c = center_crop_like(x1, y1)
        y1 = self.c1(torch.cat([x1c, y1], dim=1))

        logits = self.out(y1)

        if logits.shape[-2:] != in_hw:
            logits = F.interpolate(
                logits,
                size=in_hw,
                mode="bilinear",
                align_corners=False
                )


        return logits
    


class DoubleConv(nn.Module):
    """(Conv3x3(pad=1) -> ReLU) * 2  => spatial size 유지"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """U-Net (same conv version): crop 없이 skip concat 가능"""
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        self.d1 = DoubleConv(in_channels, base_channels)           # 64
        self.p1 = nn.MaxPool2d(2)

        self.d2 = DoubleConv(base_channels, base_channels * 2)     # 128
        self.p2 = nn.MaxPool2d(2)

        self.d3 = DoubleConv(base_channels * 2, base_channels * 4) # 256
        self.p3 = nn.MaxPool2d(2)

        self.d4 = DoubleConv(base_channels * 4, base_channels * 8) # 512
        self.p4 = nn.MaxPool2d(2)

        self.b  = DoubleConv(base_channels * 8, base_channels * 16) # 1024

        self.u4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.c4 = DoubleConv(base_channels * 16, base_channels * 8)  # (skip 512 + up 512) = 1024

        self.u3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.c3 = DoubleConv(base_channels * 8, base_channels * 4)   # (256 + 256) = 512

        self.u2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.c2 = DoubleConv(base_channels * 4, base_channels * 2)   # (128 + 128) = 256

        self.u1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.c1 = DoubleConv(base_channels * 2, base_channels)       # (64 + 64) = 128

        self.out = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_hw = x.shape[-2:]

        x1 = self.d1(x)               # (B, 64, H,   W)
        x2 = self.d2(self.p1(x1))      # (B,128, H/2, W/2)
        x3 = self.d3(self.p2(x2))      # (B,256, H/4, W/4)
        x4 = self.d4(self.p3(x3))      # (B,512, H/8, W/8)
        xb = self.b(self.p4(x4))       # (B,1024,H/16,W/16)

        y4 = self.u4(xb)              # (B,512, H/8, W/8)
        if x4.shape[-2:] != y4.shape[-2:]:
            x4 = F.interpolate(x4, size=y4.shape[-2:], mode="bilinear", align_corners=False)
        y4 = self.c4(torch.cat([x4, y4], dim=1))  # (B,512, H/8, W/8)

        y3 = self.u3(y4)              # (B,256, H/4, W/4)
        if x3.shape[-2:] != y3.shape[-2:]:
            x3 = F.interpolate(x3, size=y3.shape[-2:], mode="bilinear", align_corners=False)
        y3 = self.c3(torch.cat([x3, y3], dim=1))  # (B,256, H/4, W/4)

        y2 = self.u2(y3)              # (B,128, H/2, W/2)
        if x2.shape[-2:] != y2.shape[-2:]:
            x2 = F.interpolate(x2, size=y2.shape[-2:], mode="bilinear", align_corners=False)
        y2 = self.c2(torch.cat([x2, y2], dim=1))  # (B,128, H/2, W/2)

        y1 = self.u1(y2)              # (B,64, H, W)
        if x1.shape[-2:] != y1.shape[-2:]:
            x1 = F.interpolate(x1, size=y1.shape[-2:], mode="bilinear", align_corners=False)
        y1 = self.c1(torch.cat([x1, y1], dim=1))  # (B,64, H, W)

        logits = self.out(y1)         # (B, num_classes, H, W)

        # 혹시라도 입력이 홀수 크기라 up/down 과정에서 1픽셀 어긋나면 마지막에 입력 크기로 맞춤
        if logits.shape[-2:] != in_hw:
            logits = F.interpolate(logits, size=in_hw, mode="bilinear", align_corners=False)

        return logits


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FCN8s(nn.Module):
    """
    FCN-8s 스타일 (원조 FCN의 skip + upsample 아이디어 유지)
    - 다운샘플: 4단계 (stride 2) => 최저 해상도: 1/16
    - skip: 1/16(score_pool4), 1/8(score_pool3)
    - 최종: 입력 크기(in_hw)로 bilinear 보정
    - 채널 스케일: base_channels 기준으로 U-Net과 유사하게 맞춤
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 64) -> None:
        super().__init__()

        c1 = base_channels          # 64
        c2 = base_channels * 2      # 128
        c3 = base_channels * 4      # 256
        c4 = base_channels * 8      # 512
        c5 = base_channels * 16     # 1024 (U-Net bottleneck와 규모 맞춤)

        # Encoder (VGG-like but scaled)
        # stage1: H,W
        self.conv1_1 = ConvBNReLU(in_channels, c1)
        self.conv1_2 = ConvBNReLU(c1, c1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 1/2

        # stage2: 1/2
        self.conv2_1 = ConvBNReLU(c1, c2)
        self.conv2_2 = ConvBNReLU(c2, c2)
        self.pool2 = nn.MaxPool2d(2, 2)  # 1/4

        # stage3: 1/4  (여기서 skip(=pool3 feature) 쓸 거라 stage3 output을 저장)
        self.conv3_1 = ConvBNReLU(c2, c3)
        self.conv3_2 = ConvBNReLU(c3, c3)
        self.conv3_3 = ConvBNReLU(c3, c3)
        self.pool3 = nn.MaxPool2d(2, 2)  # 1/8

        # stage4: 1/8  (여기서 skip(=pool4 feature) 쓸 거라 stage4 output을 저장)
        self.conv4_1 = ConvBNReLU(c3, c4)
        self.conv4_2 = ConvBNReLU(c4, c4)
        self.conv4_3 = ConvBNReLU(c4, c4)
        self.pool4 = nn.MaxPool2d(2, 2)  # 1/16

        # stage5: 1/16
        self.conv5_1 = ConvBNReLU(c4, c5)
        self.conv5_2 = ConvBNReLU(c5, c5)

        # "fc6/fc7를 conv로" (원조 FCN 핵심 아이디어)
        # 원조는 7x7 conv였지만 입력 크기 유연성을 위해 3x3 same으로 유지
        self.fc6 = ConvBNReLU(c5, c5, k=3, s=1, p=1)
        self.fc7 = ConvBNReLU(c5, c5, k=1, s=1, p=0)

        # score layers (1x1 conv): 채널 -> num_classes
        self.score_fr = nn.Conv2d(c5, num_classes, kernel_size=1)   # from fc7
        self.score_pool4 = nn.Conv2d(c4, num_classes, kernel_size=1) # skip from stage4 (1/16)
        self.score_pool3 = nn.Conv2d(c3, num_classes, kernel_size=1) # skip from stage3 (1/8)

        # upsample (deconv)
        # stride=2 두 번(1/16->1/8->1/4), 마지막은 bilinear로 in_hw까지 맞춤
        self.up2_1 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.up2_2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)

        # (선택) 가중치 bilinear 초기화: 학습 안정화에 도움
        self._init_upsample_bilinear()

    def _init_upsample_bilinear(self) -> None:
        # make_bilinear_upsample_weight를 이미 utils에 갖고 있으니 활용
        with torch.no_grad():
            w = make_bilinear_upsample_weight(self.up2_1.in_channels, self.up2_1.out_channels, self.up2_1.kernel_size[0])
            self.up2_1.weight.copy_(w)
            w = make_bilinear_upsample_weight(self.up2_2.in_channels, self.up2_2.out_channels, self.up2_2.kernel_size[0])
            self.up2_2.weight.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_hw = x.shape[-2:]

        # stage1
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)

        # stage2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        # stage3 (skip)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        pool3 = x
        x = self.pool3(x)

        # stage4 (skip)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        pool4 = x
        x = self.pool4(x)

        # stage5 + fc6/fc7
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.fc6(x)
        x = self.fc7(x)

        # score at 1/16
        score_fr = self.score_fr(x)            # (B, C, H/16, W/16)
        score_pool4 = self.score_pool4(pool4)  # (B, C, H/16, W/16) before pool4 was 1/8, but pool4 is stage4 output at 1/8
        # 주의: stage4 output(pool4)은 1/8이고 pool4 layer 후가 1/16임.
        # 우리는 skip을 stage4 output(1/8)에서 쓰고 싶으니, 아래에서 size로 맞춰줄 거임.

        # upscore 1: 1/16 -> 1/8
        upscore2 = self.up2_1(score_fr)  # (B, C, ~H/8, ~W/8)

        # stage4 skip은 1/8이므로 upscore2와 크기 정렬
        if score_pool4.shape[-2:] != upscore2.shape[-2:]:
            score_pool4 = F.interpolate(score_pool4, size=upscore2.shape[-2:], mode="bilinear", align_corners=False)

        fuse_pool4 = upscore2 + score_pool4  # FCN-16s fusion (add)

        # stage3 skip 준비 (pool3 = 1/4, 하지만 우리는 1/8 skip이 필요하니 pool3는 stage3 output at 1/4임)
        # FCN 원조는 pool3(1/8), pool4(1/16), pool5(1/32)였는데,
        # 우리는 다운샘플을 1/16까지만 하므로 "pool3=1/4, pool4=1/8" 구조가 됨.
        # 따라서 여기서는 pool3를 1/4 skip으로 쓰고, fuse를 1/4로 올린 후 더한다.

        # upscore 2: 1/8 -> 1/4
        upscore4 = self.up2_2(fuse_pool4)

        score_pool3 = self.score_pool3(pool3)  # (B, C, H/4, W/4)

        if score_pool3.shape[-2:] != upscore4.shape[-2:]:
            score_pool3 = F.interpolate(score_pool3, size=upscore4.shape[-2:], mode="bilinear", align_corners=False)

        fuse_pool3 = upscore4 + score_pool3   # FCN-8s fusion (add)

        # 최종 출력은 입력 크기로 보정 (네 파이프라인과 동일)
        if fuse_pool3.shape[-2:] != in_hw:
            fuse_pool3 = F.interpolate(fuse_pool3, size=in_hw, mode="bilinear", align_corners=False)

        return fuse_pool3
