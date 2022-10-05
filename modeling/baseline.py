# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
from .backbones.swin import ft_net_swin
from .backbones.swin_transformer import SwinTransformer
from .backbones.swin_transformerv2 import SwinTransformerV2
import torch.nn.functional as F
from functools import partial
import math
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)  #64,129,3072
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)  #64,129,768
        x = self.drop(x)
        return x
class DEAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, tokens):
        B, N, C = x.shape #64,32,768
        _, n, _ = tokens.shape #1
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) #2,64,12,32,64
        k, v = kv[0], kv[1]  #64,12,32,64 # make torchscript happy (cannot use tensor as tuple)
        q = self.q(tokens).reshape(B, n, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).squeeze(0) #64,8,1,256

        attn = (q @ k.transpose(-2, -1)) * self.scale  # 64,12,1,32
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        tokens = (attn @ v).transpose(1, 2).reshape(B, n, C) #64,1,768
        tokens = self.proj(tokens)
        tokens = self.proj_drop(tokens)
        return tokens


class DEBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, local_size=[4, 8]):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)

        self.attn = DEAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=0) 

        self.local_size = local_size

    def forward(self, xtoken, tokens): # 64,128,768  64,4,768 
        B, m, C = tokens.size()
        self_token = tokens.permute(1,0,2) #4, 64, 768
        norm_token = self.norm4(self_token) #4, 64, 768
        tokens = self_token + self.drop_path(self.self_attn(norm_token, norm_token, norm_token)[0]) #4, 64, 768
        tokens = tokens.permute(1,0,2) #64,4,768 

        tokens = tokens.reshape(B * m, -1, C) #64*m,1,768 
        # xtoken = xtoken.reshape(B*4, N//4, C)
        xtoken = self.global_to_local(xtoken)  # 512,16,768 #64*m,16,2048

        #multihead attention
        tokens = tokens + self.drop_path(self.attn(xtoken, self.norm1(tokens))) #64*m,1,2048
        tokens = tokens + self.drop_path(self.mlp(self.norm2(tokens))) #64*m,1,2048

        tokens = tokens.reshape(B, -1, C)# 64, m, 2048
        # tokens = self.local_to_global(tokens) # 64, m, 2048

        return tokens

    def global_to_local(self, x: torch.tensor) -> torch.tensor:
        """8*16 -> 8*4*4"""
        h, w = self.local_size
        H, W = [16, 8]
        b, N, C = x.size()
        m = (H // h)*(W // w)
        try:
            x = x.reshape(b, H, W, C)  # 64,16,8,768
            B = b * (H // h)
            x = x.reshape(B, h, W, C)  # 256,4,8,768
            B = B * (W // w)
            x = x.transpose(1, 2).reshape(B, w, h, C).transpose(1, 2)  # 512,4,4,768
            x =  x.reshape(B, h * w, C)  # 512,16,768
            return x  #.reshape(b, m, h * w, C)  # 512,16,768
        except RuntimeError:
            raise Exception("Wrong Size!")

    def local_to_global(self, x: torch.tensor) -> torch.tensor:
        """8*4*4 -> 8*16"""
        h, w = self.local_size
        H, W = [16, 8]
        B, N, C = x.size() # 512,16,2048
        m = (H // h)*(W // w)
        try:
            x = x.reshape(B, h, w, C) # 512,4,4,2048
            b = (B * w)// W 
            x = x.transpose(1, 2).reshape(b, W, h, C).transpose(1, 2) # 256,4,8,2048
            b = (b * h) // H 
            x = x.reshape(b, H * W, C)  #64,128,2048
            return x#.reshape(b, m, h * w, C)
        except RuntimeError:
            raise Exception("Wrong Size!")

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange, local_size=[4, 8], shift_size=[2, 4]):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.model_name = cfg.MODEL.NAME
        if self.model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=cfg.MODEL.LAST_STRIDE,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
            
        elif self.model_name == 'resnet50_ibn_a':
            self.in_planes = 2048
            self.base = resnet50_ibn_a(cfg.MODEL.LAST_STRIDE)
            print('using resnet50_ibn_a as a backbone')

        elif self.model_name == 'transformer':
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
            print('using transformer as a backbone')
        elif self.model_name == 'swinTransformer':
            # self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
            self.in_planes=768 #1024
            self.base = SwinTransformer(img_size=[256,128],#config.DATA.IMG_SIZE
                                patch_size=4,#config.MODEL.SWIN.PATCH_SIZE
                                in_chans=3, #config.MODEL.SWIN.IN_CHANS
                                #num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=96, #config.MODEL.SWIN.EMBED_DIM
                                depths=[2,2,18,2],#config.MODEL.SWIN.DEPTHS
                                num_heads=[3, 6, 12, 24],#config.MODEL.SWIN.NUM_HEADS
                                window_size=[16,8],#config.MODEL.SWIN.WINDOW_SIZE 7
                                mlp_ratio=4,#config.MODEL.SWIN.MLP_RATIO
                                qkv_bias=True,#config.MODEL.SWIN.QKV_BIAS
                                qk_scale=None,#config.MODEL.SWIN.QK_SCALE
                                drop_rate=cfg.MODEL.DROP_OUT,#config.MODEL.DROP_RATE
                                drop_path_rate=cfg.MODEL.DROP_PATH,#config.MODEL.DROP_PATH_RATE
                                ape=False,#config.MODEL.SWIN.APE
                                patch_norm=True,#config.MODEL.SWIN.QK_SCALE
                                use_checkpoint=False)#config.TRAIN.USE_CHECKPOINT
            print('using Swin transformer as a backbone')
        
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))


        self.num_classes = num_classes
        # self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        # if self.ID_LOSS_TYPE == 'arcface':
        #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
        #     self.classifier = Arcface(self.in_planes, self.num_classes,
        #                               s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        # elif self.ID_LOSS_TYPE == 'cosface':
        #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
        #     self.classifier = Cosface(self.in_planes, self.num_classes,
        #                               s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        # elif self.ID_LOSS_TYPE == 'amsoftmax':
        #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
        #     self.classifier = AMSoftmax(self.in_planes, self.num_classes,
        #                                 s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        # elif self.ID_LOSS_TYPE == 'circle':
        #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
        #     self.classifier = CircleLoss(self.in_planes, self.num_classes,
        #                                 s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        # else:
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier0 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier0.apply(weights_init_classifier)
        self.classifier1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier1.apply(weights_init_classifier)
        self.classifier2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier2.apply(weights_init_classifier)
        self.classifier3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier3.apply(weights_init_classifier)
        self.classifier4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck0 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck0.bias.requires_grad_(False)
        self.bottleneck0.apply(weights_init_kaiming)
        self.bottleneck1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck1.bias.requires_grad_(False)
        self.bottleneck1.apply(weights_init_kaiming)
        self.bottleneck2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck2.bias.requires_grad_(False)
        self.bottleneck2.apply(weights_init_kaiming)
        self.bottleneck3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck3.bias.requires_grad_(False)
        self.bottleneck3.apply(weights_init_kaiming)
        self.bottleneck4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck4.bias.requires_grad_(False)
        self.bottleneck4.apply(weights_init_kaiming)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.tokens = nn.Parameter(torch.zeros(1, 4, self.in_planes))
        trunc_normal_(self.tokens, std=.02)

        from functools import partial
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        self.denorm1 = norm_layer(self.in_planes)
        self.denorm2 = norm_layer(self.in_planes)
        self.denorm3 = norm_layer(self.in_planes)
        self.denorm4 = norm_layer(self.in_planes)

        dpr = [x.item() for x in torch.linspace(0, 0.05, 6)]  # stochastic depth decay rule

        self.num_heads = 8
        self.deblocks = nn.ModuleList([
            DEBlock(
                dim=self.in_planes, num_heads=self.num_heads, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(2)])

        self.self_attn = nn.MultiheadAttention(self.in_planes, self.num_heads, dropout=0)

        self.norm1 = norm_layer(self.in_planes)
        self.norm2 = norm_layer(self.in_planes)
        self.mlp =Mlp(in_features=self.in_planes, hidden_features=self.in_planes * 4, act_layer=nn.GELU, drop=0)

        self.shift_size = shift_size
        self.local_size = local_size

    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
        B = x.shape[0] #64
        if self.model_name == 'resnet50_ibn_a' or self.model_name == 'resnet50':
            x = self.base(x) #64,2048,16,8
            token0 = nn.functional.avg_pool2d(x, x.shape[2:4]).view(x.shape[0], -1) #64,2048
            xtoken = x.view(x.shape[0],x.shape[1],-1).permute(0,2,1) #64,128,2048

            self_token = xtoken.permute(1,0,2)
            norm_token = self.norm1(self_token)
            attn_token, attn_map = self.self_attn(norm_token, norm_token, norm_token) #128, 64, 768
            xtoken = self_token + attn_token
            xtoken = xtoken.permute(1,0,2) #64,128,768

            xtoken = xtoken + self.mlp(self.norm2(xtoken)) #64,1,2048
            xtoken0 = xtoken.mean(1)

        if self.model_name == 'transformer':
            # x = self.base(x, cam_label=cam_label, view_label=view_label) #64,129,768
            x=self.base(x)
            token0 = x[:, 0]  #64,768
            xtoken = x[:, 1:]#.clone().detach() #64,128,768
        if self.model_name == 'swinTransformer':
            # x = self.base(x, cam_label=cam_label, view_label=view_label) #64,129,768
            x=self.base(x)
            token0 =torch.flatten(self.gap(x.transpose(1, 2)),1) #64,768
            # token0=self.avgpool(x.transpose(1, 2)) 
            xtoken = x #.clone().detach() #64,128,768
            self_token = xtoken.permute(1,0,2)
            norm_token = self.norm1(self_token)
            attn_token, attn_map = self.self_attn(norm_token, norm_token, norm_token) #128, 64, 768
            xtoken = self_token + attn_token
            xtoken = xtoken.permute(1,0,2) #64,128,768

            xtoken = xtoken + self.mlp(self.norm2(xtoken)) #64,1,2048
            xtoken0 = xtoken.mean(1)

        tokens = self.tokens.expand(B, -1, -1) #64,m,2048
        
        L = 0
        for deblk in self.deblocks:
            if L == 0:
                tokens = deblk(xtoken, tokens) #64,m,2048
            else:
                xtoken = xtoken.view(B,16,8,-1)
                xtoken = torch.roll(xtoken, shifts = (self.shift_size[0], self.shift_size[1]), dims = (1,2))
                xtoken = xtoken.flatten(1,2)
                tokens = deblk(xtoken, tokens) #64,4,2048
            L = L + 1
            
        tokens1 = tokens[:, 0]
        tokens2 = tokens[:, 1]
        tokens3 = tokens[:, 2]
        tokens4 = tokens[:, 3]
        
        t0_bn = self.bottleneck(token0) #64,768
        tx0_bn = self.bottleneck0(xtoken0) #64,768
        t1_bn = self.bottleneck1(tokens1) #64,768
        t2_bn = self.bottleneck2(tokens2) #64,768
        t3_bn = self.bottleneck3(tokens3) #64,768
        t4_bn = self.bottleneck4(tokens4) #64,768

        if self.training:
            cls_score = self.classifier(t0_bn)
            cls_score_0 = self.classifier0(tx0_bn)
            cls_score_1 = self.classifier1(t1_bn)
            cls_score_2 = self.classifier2(t2_bn)
            cls_score_3 = self.classifier3(t3_bn)
            cls_score_4 = self.classifier4(t4_bn)

            return [cls_score, cls_score_0, cls_score_1, cls_score_2, cls_score_3, cls_score_4], \
                   [token0, xtoken0, tokens1, tokens2, tokens3, tokens4]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat([t0_bn, t1_bn / 4, t2_bn / 4, t3_bn / 4, t4_bn / 4], dim=1)
            else: #####
                return torch.cat([token0, xtoken0, tokens1, tokens2, tokens3, tokens4], dim=1)

                # return features


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))



class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        self.model_name=model_name
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)
        elif model_name=="swin224":
            self.in_planes=1024
            self.base=ft_net_swin()
        elif model_name== 'swinTransformer':
            self.in_planes=768 #1024
            self.base = SwinTransformer(img_size=[256,128],#config.DATA.IMG_SIZE
                                patch_size=4,#config.MODEL.SWIN.PATCH_SIZE
                                in_chans=3, #config.MODEL.SWIN.IN_CHANS
                                #num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=96, #config.MODEL.SWIN.EMBED_DIM
                                depths=[2,2,6,2],#config.MODEL.SWIN.DEPTHS
                                num_heads=[3, 6, 12, 24],#config.MODEL.SWIN.NUM_HEADS
                                window_size=[8,8],#config.MODEL.SWIN.WINDOW_SIZE 7
                                mlp_ratio=4,#config.MODEL.SWIN.MLP_RATIO
                                qkv_bias=True,#config.MODEL.SWIN.QKV_BIAS
                                qk_scale=None,#config.MODEL.SWIN.QK_SCALE
                                drop_rate=0.0,#config.MODEL.DROP_RATE
                                drop_path_rate=0.1,#config.MODEL.DROP_PATH_RATE
                                ape=False,#config.MODEL.SWIN.APE
                                patch_norm=True,#config.MODEL.SWIN.QK_SCALE
                                use_checkpoint=False)#config.TRAIN.USE_CHECKPOINT
        elif self.model_name == 'swinTransformerV2':
            # self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
            self.in_planes=768 #1024
            self.base = SwinTransformerV2(img_size=[256,128],#config.DATA.IMG_SIZE
                                patch_size=4,#config.MODEL.SWIN.PATCH_SIZE
                                in_chans=3, #config.MODEL.SWIN.IN_CHANS
                                #num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=96, #config.MODEL.SWIN.EMBED_DIM
                                depths=[2,2,6,2],#config.MODEL.SWIN.DEPTHS
                                num_heads=[3, 6, 12, 24],#config.MODEL.SWIN.NUM_HEADS
                                window_size=[8,8],#config.MODEL.SWIN.WINDOW_SIZE 7
                                mlp_ratio=4,#config.MODEL.SWIN.MLP_RATIO
                                qkv_bias=True,#config.MODEL.SWIN.QKV_BIAS
                                qk_scale=None,#config.MODEL.SWIN.QK_SCALE
                                drop_rate=0.0,#config.MODEL.DROP_RATE
                                drop_path_rate=0.1,#config.MODEL.DROP_PATH_RATE
                                ape=False,#config.MODEL.SWIN.APE
                                patch_norm=True,#config.MODEL.SWIN.QK_SCALE
                                use_checkpoint=False)#config.TRAIN.USE_CHECKPOINT
            print('using Swin transformer as a backbone')
        if pretrain_choice == 'imagenet' and model_name!="swin224":
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap1 = nn.AdaptiveAvgPool1d(1)
        self.gap2=nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            # self.classifier = nn.Linear(self.in_planes, self.num_classes)
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            self.classifier.apply(weights_init_classifier)  # new add by luo
        # elif self.neck == 'bnneck':
        #     self.bottleneck = nn.BatchNorm1d(self.in_planes)
        #     self.bottleneck.bias.requires_grad_(False)  # no shift
        #     self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        #
        #     self.bottleneck.apply(weights_init_kaiming)
        #     self.classifier.apply(weights_init_classifier)

    def forward(self, x,label=None, cam_label= None, view_label=None):
        # global_feat = (self.base(x))  # (b, 2048, 1, 1)
        if self.model_name=="resnet50":
            global_feat = self.gap2(self.base(x))  # (b, 2048, 1, 1)
        else:
            global_feat = self.gap1(self.base(x).transpose(1,2))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path).state_dict()
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape #64,129,768
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) #3,64,12,129,64
        q, k, v = qkv[0], qkv[1], qkv[2]  #64,12,129,64 # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale  #64,12,129,129
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) #64,129,768
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) #64,129,768
        x = x + self.drop_path(self.mlp(self.norm2(x))) #64,129,768
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x) #64,768,16,8

        x = x.flatten(2).transpose(1, 2) # [64, 128, 768]
        return x


class TransReID(nn.Module):
    """ Transformer-based Object Re-Identification
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., camera=0, view=0,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, local_feature=False, sie_xishu =1.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other backbone
        self.local_feature = local_feature
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed_overlap(
                img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
                embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cam_num = camera
        self.view_num = view
        self.sie_xishu = sie_xishu

        if camera > 1 and view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {} and viewpoint number is : {}'.format(camera, view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {}'.format(camera))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('viewpoint number is : {}'.format(view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))

        print('using drop_out rate is : {}'.format(drop_rate))
        print('using attn_drop_out rate is : {}'.format(attn_drop_rate))
        print('using drop_path rate is : {}'.format(drop_path_rate))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, camera_id, view_id):
        B = x.shape[0] #64
        x = self.patch_embed(x) #64,128,768

        cls_tokens = self.cls_token.expand(B, -1, -1) #64,1,768 # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1) #64,129,768

        if self.cam_num > 0 and self.view_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id * self.view_num + view_id]
        elif self.cam_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
        elif self.view_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[view_id]
        else: ##
            x = x + self.pos_embed

        x = self.pos_drop(x) #64,129,768

        for blk in self.blocks:
            x = blk(x) #64,129,768

        x = self.norm(x)

        # return x[:, 0]    #64,768
        return x    #64,768


    def forward(self, x, cam_label=None, view_label=None):
        x = self.forward_features(x, cam_label, view_label)
        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old backbone that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)  ##
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape,self.state_dict()[k].shape))



def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape) #1,197,768 -> 1,129,768
    ntok_new = posemb_new.shape[1]  #129

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:] #1,1,768 ,196,768
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid))) #14
    print('Position embedding resize to height:{} width: {}'.format(hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2) #1,768,14,14
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear') #1,768,16,8
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1) #1,128,768
    posemb = torch.cat([posemb_token, posemb_grid], dim=1) #1,129,768
    return posemb


def vit_base_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, camera=0, view=0,local_feature=False,sie_xishu=1.5, **kwargs):
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,\
        camera=camera, view=view, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  sie_xishu=sie_xishu, local_feature=local_feature, **kwargs)

    return model

def vit_small_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_rate=0., attn_drop_rate=0.,drop_path_rate=0.1, camera=0, view=0, local_feature=False, sie_xishu=1.5, **kwargs):
    kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=8, num_heads=8,  mlp_ratio=3., qkv_bias=False, drop_path_rate = drop_path_rate,\
        camera=camera, view=view,  drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  sie_xishu=sie_xishu, local_feature=local_feature, **kwargs)

    return model

def deit_small_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0, camera=0, view=0, local_feature=False, sie_xishu=1.5, **kwargs):
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, camera=camera, view=view, sie_xishu=sie_xishu, local_feature=local_feature,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model

