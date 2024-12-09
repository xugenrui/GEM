import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from modules import (ConvSC, ConvNeXtSubBlock, ConvMixerSubBlock, GASubBlock, gInception_ST,
                             HorNetSubBlock, MLPMixerSubBlock, MogaSubBlock, PoolFormerSubBlock,
                             SwinSubBlock, UniformerSubBlock, VANSubBlock, ViTSubBlock, TAUSubBlock)

from models.img_model.simvp_model import SimVP_Model

def pair(t):
    return t if isinstance(t, tuple) else (t, t)



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class enc_Model(nn.Module):
    def __init__(self,configs):
        super(enc_Model, self).__init__()

        # print(configs.img_size)
        
        self.pretrain_param = configs.pretrain_param
        if self.pretrain_param is not None:
            device = torch.device(f'cuda:{configs.cuda}' if torch.cuda.is_available() else 'cpu')
            cpt = torch.load(self.pretrain_param,map_location=device)

            self.inference_model = SimVP_Model(**cpt['hyper_parameters'])
            self.channels = cpt['hyper_parameters']['hid_S']
            model_dict = {key.replace('model.', ''): value for key, value in cpt['state_dict'].items() if 'model.' in key}

            self.inference_model.load_state_dict(model_dict)
        else:
            self.inference_model = SimVP_Model(
                (configs.img_len, configs.channels, configs.img_size, configs.img_size),
                hid_S = 32,
                hid_T = 128,
                N_S = 2,
                N_T = 8,
                model_type = 'gSTA',
                mlp_ratio = 8.,
                drop = 0.0,
                drop_path = 0.1
            )
            self.channels = 32
        self.frozen = configs.frozen

        self.img_h, self.img_w = pair(configs.img_size)
        self.p_h, self.p_w = pair(configs.patch_size)
        assert self.img_h % self.p_h == 0 and self.img_w % self.p_w == 0, 'Image size must be divisible by the patch size.'
        self.num_patches = (self.img_h // self.p_h) * (self.img_w // self.p_w)
        self.patch_dim = self.channels * self.p_h * self.p_w
        self.img_len = configs.img_len
        self.pred_len = configs.pred_len
        self.dim = configs.img_d_model
        self.img_pool = configs.img_pool
        self.img_otp = configs.img_otp
        self.conv1 = ConvSC(self.channels, self.channels)


        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.p_h, p2 = self.p_w),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, self.dim),
            nn.LayerNorm(self.dim),
        )   


        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, configs.img_d_model))
        self.cls_token = nn.Parameter(torch.randn(1,1,self.dim))
        self.dropout = nn.Dropout(configs.dropout)

        self.transformer = Transformer(dim=configs.img_d_model,depth=configs.img_e_layers,heads=configs.img_n_heads,
                                       dim_head=configs.img_dim_head,mlp_dim=configs.img_d_ff,dropout=configs.dropout)
        self.to_latent = nn.Identity()

    def forward(self, x):
        B, T, _, H, W = x.shape
        C = self.channels
        if self.pretrain_param is not None and self.frozen is True:
            with torch.no_grad():
                x = self.inference_model(x,inference=True)
        else:
            x = self.inference_model(x,inference=True)
        x = self.conv1(x)
        x = x.reshape(B, T, C, H, W) 
        x = x[:,:self.img_otp,:,:,:]

        x = rearrange(x,'b t c h w -> (b t) c h w')
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b = B*self.img_otp)
        x = self.to_patch_embedding(x)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embedding[:, :(self.num_patches + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.img_pool == 'mean' else x[:, 0]
        x = rearrange(x, '(b t) d -> b t d', t=self.img_otp)

        return x
