import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer, Cross_EncoderLayer, Cross_Encoder
from layers.SelfAttention_Family import FullAttention, AttentionLayer, FusionAttentionLayer
from layers.Embed import DataEmbedding
from einops import rearrange, repeat
from einops.layers.torch import Rearrange



class Adaptive_fusion(nn.Module):
    def __init__(self, configs):
        super(Adaptive_fusion, self).__init__()
        self.exps = configs.exps
        self.pred_len = configs.pred_len

        self.dec_embedding = DataEmbedding(configs.ts_dec_in, configs.ts_d_model, configs.ts_embed, configs.freq,
                                           configs.dropout)
        
        self.gate = nn.Linear(configs.ts_d_model * configs.pred_len, configs.ts_d_model * configs.pred_len, bias=True)
        self.sigmoid = nn.Sigmoid()
        if configs.ts_distil:
            self.reg = nn.Linear(configs.ts_d_model * (configs.seq_len // 2 + 1), configs.ts_d_model * configs.pred_len, bias=True)
            self.reg_cs = nn.Linear(configs.ts_d_model * (configs.seq_len // 2 + 1), configs.ts_d_model * configs.pred_len, bias=True)
        else:
            self.reg = nn.Linear(configs.ts_d_model * configs.seq_len, configs.ts_d_model * configs.pred_len, bias=True)
            self.reg_cs = nn.Linear(configs.ts_d_model * configs.seq_len, configs.ts_d_model * configs.pred_len, bias=True)
        

    def forward(self, ts_enc_x, img_enc_x, dec_inp, dec_inp_mark):
        _, L, _ = ts_enc_x.shape

        ts_enc_x = rearrange(ts_enc_x, 'b t d -> b (t d)')

        unit = ts_enc_x.detach()
        unit = self.reg_cs(unit)
        
        ts_enc_x = self.reg(ts_enc_x)

        ts_enc_x_gate = self.sigmoid(self.gate(unit))
        ts_enc_x_gate = rearrange(ts_enc_x_gate, 'b (t d) -> b t d', t=self.pred_len)
        ts_enc_x = rearrange(ts_enc_x, 'b (t d) -> b t d', t=self.pred_len)

        dec_info = self.dec_embedding(dec_inp, dec_inp_mark)
        dec_info = dec_info[:, -self.pred_len:, :]

        ts_enc_x = ts_enc_x + ts_enc_x_gate * dec_info

        if self.exps == True:
            return ts_enc_x, ts_enc_x_gate
        else:
            return ts_enc_x