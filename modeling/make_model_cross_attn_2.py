import torch
import torch.nn as nn
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a
from .backbones.resnet import ResNet, Bottleneck

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

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))


        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
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
        self.bottleneck4.apply(weights_init_kaiming)#[xtt] bias not train

        self.gap = nn.AdaptiveAvgPool2d(1)

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
            x = self.base(x, cam_label=cam_label, view_label=view_label) #64,129,768
            token0 = x[:, 0]  #64,768
            xtoken = x[:, 1:]#.clone().detach() #64,128,768

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


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
    return model
