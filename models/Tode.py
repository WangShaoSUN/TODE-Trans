import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import Concat
from .swin_transformer import SwinTransformer
import math

class UpSampleBN(nn.Cell):
    def __init__(self, input_features, output_features, res=True):
        super(UpSampleBN, self).__init__()
        self.res = res

        self._net = nn.SequentialCell([
            nn.Conv2d(input_features, input_features, kernel_size=3, stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(input_features),
            nn.LeakyReLU(),
            nn.Conv2d(input_features, input_features, kernel_size=3, stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(input_features),
            nn.LeakyReLU()
        ])

        self.up_net = nn.SequentialCell([
            nn.Conv2dTranspose(input_features, output_features, kernel_size=2, stride=2, pad_mode='pad'),
            nn.BatchNorm2d(output_features),
            nn.ReLU()
        ])

    def construct(self, x, concat_with):
        if concat_with is None:
            conv_x = self._net(x) + x if self.res else self._net(x)
        else:
            cat_x = Concat(1)((x, concat_with))
            conv_x = self._net(cat_x) + cat_x if self.res else self._net(cat_x)
        return self.up_net(conv_x)

class SELayer_down(nn.Cell):
    def __init__(self, H, W):
        super(SELayer_down, self).__init__()
        self.avg_pool_channel = nn.AdaptiveAvgPool1d(1)
        self.avg_pool_2d = nn.AdaptiveAvgPool2d((H // 2, W // 2))

    def construct(self, in_data, x):
        b, c, h, w = x.shape
        x = x.transpose(0, 2, 3, 1).reshape(b, -1, c)
        y = self.avg_pool_channel(x).view(b, h, w, 1)
        y = y.transpose(0, 3, 1, 2)
        y = self.avg_pool_2d(y)
        return in_data * y.expand_as(in_data)

class DecoderBN(nn.Cell):
    def __init__(self, num_features=128, lambda_val=1, res=True):
        super(DecoderBN, self).__init__()
        features = int(num_features)
        self.lambda_val = lambda_val

        self.se1_down = SELayer_down(120, 160)
        self.se2_down = SELayer_down(60, 80)
        self.se3_down = SELayer_down(30, 40)

        self.up1 = UpSampleBN(192, features, res)
        self.up2 = UpSampleBN(features + 96, features, res)
        self.up3 = UpSampleBN(features + 48, features, res)
        self.up4 = UpSampleBN(features + 24, features//2, res)


    def construct(self, features):
        x_block4, x_block3, x_block2, x_block1= features[3], features[2], features[1], features[0]

        x_block2_1 = self.lambda_val * self.se1_down(x_block2, x_block1) + (1-self.lambda_val) * x_block2
        x_block3_1 = self.lambda_val * self.se2_down(x_block3, x_block2_1) + (1-self.lambda_val) * x_block3
        x_block4_1 = self.lambda_val * self.se3_down(x_block4, x_block3_1) + (1-self.lambda_val) * x_block4

        x_d0 = self.up1(x_block4_1, None)
        x_d1 = self.up2(x_d0, x_block3_1)
        x_d2 = self.up3(x_d1, x_block2_1)
        x_d3 = self.up4(x_d2, x_block1)


        return x_d3

class Tode(nn.Cell):
    def __init__(self, lambda_val=1, res=True):
        super(Tode, self).__init__()
        self.encoder = SwinTransformer(patch_size=2, in_chans=4, embed_dim=24)
        self.decoder = DecoderBN(num_features=128, lambda_val=lambda_val, res=res)
        
        # Define final layers
        self.final_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, pad_mode='pad', padding=1)
        self.final_bn1 = nn.BatchNorm2d(64)
        self.final_relu1 = nn.ReLU()
        self.final_conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, pad_mode='pad', padding=1)
        self.final_relu2 = nn.ReLU()

    def construct(self, img, depth):
        shape_len = len(depth.shape)
        if shape_len == 3:
            n, h, w = depth.shape
        elif shape_len == 2:
            h, w = depth.shape
            n = 1
        else:
            # Handle error or unexpected shape
            raise ValueError(f"Expected depth to be 2 or 3 dimensions, but got {shape_len} dimensions")
        depth = ops.reshape(depth, (n, 1, h, w))
        img = ops.reshape(img, (n, 3, h, w))
        encoder_x = self.encoder(ops.concat((img, depth), axis=1))
        decoder_x = self.decoder(encoder_x)

        out = self.final_conv1(decoder_x)
        out = self.final_bn1(out)
        out = self.final_relu1(out)
        out = self.final_conv2(out)
        out = self.final_relu2(out)

        return out
