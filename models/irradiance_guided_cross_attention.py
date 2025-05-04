import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer, Cross_EncoderLayer, Cross_Encoder
from layers.SelfAttention_Family import FullAttention, AttentionLayer, FusionAttentionLayer
from layers.Embed import DataEmbedding

class irradiance_guided_cross_attn(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(irradiance_guided_cross_attn, self).__init__()
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, y, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, y, y,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        z = x = self.norm1(x)


        return z, attn

class dec_Model(nn.Module):
    def __init__(self, args):
        super(dec_Model, self).__init__()
        self.pred_len = args.pred_len

        if args.img_model == 'Simvpv2':
            self.img_enc_len = args.img_otp
        else:
            self.img_enc_len = args.pred_len

        self.dec_embedding = DataEmbedding(args.ts_dec_in, args.ts_d_model, args.ts_embed, args.freq,
                                           args.dropout)

        self.cross_dec = Cross_Encoder(
            [
                irradiance_guided_cross_attn(
                    AttentionLayer(
                        FullAttention(False, args.ts_factor, attention_dropout=args.dropout,
                                      ), args.ts_d_model, args.ts_n_heads),
                    args.ts_d_model,
                    args.ts_d_ff,
                    dropout=args.dropout,
                    activation=args.crs_activation
                ) for l in range(args.crs_d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.ts_d_model),
        )
        
        self.dec = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, args.ts_factor, attention_dropout=args.dropout,
                                      ), args.ts_d_model, args.ts_n_heads),
                    args.ts_d_model,
                    args.ts_d_ff,
                    dropout=args.dropout,
                    activation=args.crs_activation
                ) for l in range(args.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.ts_d_model),
        )

        self.projection = nn.Linear(args.ts_d_model, args.c_out, bias=True)

    def forward(self, ts_enc_x, img_enc_x, dec_inp, dec_inp_mark):

        dec_out,_ = self.cross_dec(ts_enc_x, img_enc_x)
        
        dec_out,_ = self.dec(dec_out)

        dec_out = self.projection(dec_out)
      
        return dec_out
