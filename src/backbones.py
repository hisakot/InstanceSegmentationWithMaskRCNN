import timm
import torchvision
import torch
from torch import nn
from typing import OrderedDict
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

class ResNetBackBoneWithFPN(nn.Module):
    """
    output_channel
    resnet18,34  -> [64, 128, 256, 512] = [out1,out2,out3,out4]
    resnetxx(xx>=50) -> [256, 512, 1024, 2048]
    """
    def __init__(self, resnet_type="resnet50", pretrained=False): # NOTE: resnet101, True are origin
        super().__init__()
        if resnet_type == "resnet18":
            resnet_model = 'resnet18'
            in_channels_list = [64, 128, 256, 512]
        elif resnet_type == "resnet34":
            resnet_model = 'resnet34'
            in_channels_list = [64, 128, 256, 512]
        elif resnet_type == "resnet50":
            resnet_model = 'resnet50'
            in_channels_list = [256, 512, 1024, 2048]
        elif resnet_type == "resnet101":
            resnet_model = 'resnet101'
            in_channels_list = [256, 512, 1024, 2048]
        elif resnet_type == "resnet152":
            resnet_model = 'resnet152'
            in_channels_list = [256, 512, 1024, 2048]
        self.resnet_model = timm.create_model(
                resnet_model,
                pretrained=False, # NOTE: True,
                num_classes=0,
                global_pool='',
                drop_path_rate=0.1)
        self.out_channels = 256 
        '''
        # to protect pretrained weights
        layers_to_train = ["layer4", "layer3", "layer2"]
        for name, parameter in self.resnet_model.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)
        '''
        self.fpn = FeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=self.out_channels,
                extra_blocks=None,
# norm_layer=nn.BatchNorm2d # Note: koregaarutougokanai
        )
    def forward(self, x):
        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.act1(x)
        x = self.resnet_model.maxpool(x)
        out1 = self.resnet_model.layer1(x)
        out2 = self.resnet_model.layer2(out1)
        out3 = self.resnet_model.layer3(out2)
        out4 = self.resnet_model.layer4(out3)
        x = OrderedDict()
        x['0'] = out1
        x['1'] = out2
        x['2'] = out3
        x['3'] = out4
        output = self.fpn(x)
        return output

class ViTBackboneWithFPN(nn.Module):
    def __init__(self, vit_type='large') -> None:
        super().__init__()
        if vit_type == "base":
            vit_model = 'vit_base_patch16_224_in21k'
            embed_dim = 768
            patch = 16
            block_list = [1, 4, 7, 10]
        elif vit_type == "large":
            vit_model = 'vit_large_patch16_224_in21k'
            embed_dim = 1024
            patch = 16
            block_list = [4, 10, 16, 22]
        elif vit_type == "huge":
            vit_model = 'vit_huge_patch14_224_in21k'
            embed_dim = 1280
            patch = 14
            block_list = [6, 14, 22, 30]
        self.model = timm.create_model(
                vit_model,
                pretrained=True,
                num_classes=0,
                global_pool='',
                drop_path_rate=0.3)
        img_wh = 224
        self.ch = 3
        self.patch = patch
        self.embed_dim = embed_dim
        self.num_patch = img_wh // self.patch
        self.sec1 = nn.Sequential(
            *self.model.blocks[0:block_list[0]]
        )
        self.block1d4 = self.model.blocks[block_list[0]]
        self.up4x = nn.Sequential(
                nn.ConvTranspose2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(2,2), stride=2),
                nn.GroupNorm(num_groups=embed_dim//16, num_channels=embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(2,2), stride=2),
        )
        self.sec2 = nn.Sequential(
            *self.model.blocks[block_list[0]+1:block_list[1]]
        )
        self.block2d4 = self.model.blocks[block_list[1]]
        self.up2x = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(2,2), stride=2)
        self.sec3 = nn.Sequential(
            *self.model.blocks[block_list[1]+1:block_list[2]]
        )
        self.block3d4 = self.model.blocks[block_list[2]]
        self.identity = nn.Identity()
        self.sec4 = nn.Sequential(
            *self.model.blocks[block_list[2]+1:block_list[3]]
        )
        self.block4d4 = self.model.blocks[block_list[3]]
        self.down2x = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.out_channels = 256
        '''
        layers_to_train = ["down2x", "block4d4", "sec4", "identity", "block3d4", "sec3", "up2x", "layer2d4", "sec2", "up4x"]
        for name, parameter in self.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)
        '''
        self.fpn = FeaturePyramidNetwork(
                in_channels_list=[embed_dim, embed_dim, embed_dim, embed_dim],
                out_channels=self.out_channels,
                extra_blocks=None,
                norm_layer=nn.BatchNorm2d
        )

    def forward(self, x):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.norm_pre(x)
        x = self.sec1(x)
        block1d4 = self.block1d4(x)
        x = self.sec2(block1d4)
        block2d4 = self.block2d4(x)
        x = self.sec3(block2d4)
        block3d4 = self.block3d4(x)
        x = self.sec4(block3d4)
        block4d4 = self.block4d4(x)
        
        out1 = block1d4[:, 1:].clone().reshape(-1, self.num_patch, self.num_patch, self.embed_dim).permute((0, 3, 1, 2))
        out2 = block2d4[:, 1:].clone().reshape(-1, self.num_patch, self.num_patch, self.embed_dim).permute((0, 3, 1, 2))
        out3 = block3d4[:, 1:].clone().reshape(-1, self.num_patch, self.num_patch, self.embed_dim).permute((0, 3, 1, 2))
        out4 = block4d4[:, 1:].clone().reshape(-1, self.num_patch, self.num_patch, self.embed_dim).permute((0, 3, 1, 2))
        x4 = self.up4x(out1)
        x2 = self.up2x(out2)
        x1 = self.identity(out3)
        x05 = self.down2x(out4)
        
        x = OrderedDict()
        x['0'] = x4
        x['1'] = x2
        x['2'] = x1
        x['3'] = x05
        output = self.fpn(x)
        return output

