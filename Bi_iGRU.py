import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat



class Configs:
    def __init__(self, seq_len, pred_len, enc_in):
        self.seq_len = seq_len               # input sequence length
        self.pred_len = pred_len             # forecast horizon
        self.enc_in = enc_in                 # number of input channels (variates)

        self.d_model = 128                    # model dimension
        self.d_ff = 128                      # feed-forward dimension
        self.e_layers = 3                    # number of encoder layers
        self.dropout = 0.2                   # dropout rate
        self.activation = 'relu'             # activation function
        self.output_attention = False        # no attention output
        self.use_norm = True                 # use normalization
        self.embed = 'fixed'                 # embedding type (unused here)
        self.freq = 'h'     


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        return x

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        #if x_mark is None:
        x = self.value_embedding(x)
        # else:
        #     x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self, attention, attention_r, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.attention_r = attention_r
        self.conv1 = nn.Linear(d_model, d_ff)
        self.conv2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.sorted_indices = None
        self.unsorted_indices = None

    def set_sorted_indices(self, sorted_indices, unsorted_indices):
        self.sorted_indices = sorted_indices
        self.unsorted_indices = unsorted_indices

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        B, N, E = x.shape
        new_x, _ = self.attention(x)
        new_x_back, _ = self.attention_r(x.flip(dims=[1]))
        x = x + new_x + new_x_back.flip(dims=[1])
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        return self.norm2(x + y), None

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns

class CustomModel(nn.Module):
    def __init__(self, configs):
        super(CustomModel, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.stride1 = 8
        self.patch_len1 = 8
        self.stride2 = 8
        self.patch_len2 = 16
        self.stride3 = 7
        self.patch_len3 = 24
        self.stride4 = 6
        self.patch_len4 = 32
        self.e_layers = configs.e_layers
        self.decompsition = series_decomp(13)
        self.patch_num1 = int((self.seq_len - self.patch_len2) // self.stride2) + 2
        self.padding_patch_layer1 = nn.ReplicationPad1d((0, self.stride1))
        self.padding_patch_layer2 = nn.ReplicationPad1d((0, self.stride2))
        self.padding_patch_layer3 = nn.ReplicationPad1d((0, self.stride3))
        self.padding_patch_layer4 = nn.ReplicationPad1d((0, self.stride4))
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, 
                                                    configs.embed, 
                                                    configs.freq,
                                                    configs.dropout)
        
        self.conv = torch.nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_model, 
                                    kernel_size=11, padding=5,
                                    padding_mode='reflect')
        self.encoder = Encoder(
            [
                EncoderLayer(
                    torch.nn.GRU(input_size=configs.d_model, hidden_size=configs.d_model,
                                 batch_first=True, num_layers=1, bidirectional=False),
                    torch.nn.GRU(input_size=configs.d_model, hidden_size=configs.d_model,
                                 batch_first=True, num_layers=1, bidirectional=False),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.downsample_layers = nn.ModuleList([
            nn.AvgPool1d(kernel_size=2**i, stride=2**i) for i in range(1, 4)
        ])
        self.down_pool = nn.Conv1d(in_channels=configs.enc_in, out_channels=configs.enc_in,
                                   kernel_size=3, padding=1,
                                   stride=2,
                                   padding_mode='circular',
                                   bias=False)
        self.embedding_map = nn.Linear(configs.d_model * self.patch_num1, configs.d_model)



    def forecast(self, x_enc):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        x_enc = self.enc_embedding(x_enc)
        enc_out, _ = self.encoder(x_enc, attn_mask=None)
        
        dec_out = self.projector(enc_out).permute(0, 2, 1)
        #print(dec_out.shape)
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc) #, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]