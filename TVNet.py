import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
from torch.nn.modules.utils import _triple


class Configs:
    def __init__(self, seq_len, pred_len, enc_in):
        self.seq_len = seq_len               # input sequence length
        self.pred_len = pred_len             # forecast horizon
        self.enc_in = enc_in                 # number of input channels (variates)
        self.c_in = enc_in                   # input dimension
        self.c_out = enc_in                  # output dimension (same as input for univariate time
        self.patch_length = 24
        self.stride = 12                     # stride for adaptive embedding
        self.d_model = 32                    # model dimension
        self.d_ff = 64                      # feed-forward dimension
        self.layers = 3                    # number of encoder layers
        self.dropout = 0.2                   # dropout rate
        self.activation = 'relu'             # activation function
        self.output_attention = False        # no attention output
        self.use_norm = True                 # use normalization
        self.embed = 'fixed'                 # embedding type (unused here)
        self.freq = 'h' 
        self.task_name = 'long_term_forecast'  # task name


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        
        """
        d_model: means output dim
        x=torch.randn(32,96,1)
        model=PositionalEmbedding(256)
        y=model(x)==>(1,96,d_model=256)
        """
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        
        """
        c_in: means input dim
        d_model: means output dim
        model=TokenEmbedding(8,256)
        x=torch.randn(32,96,8)
        y==>(32,96,256)
        """
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
        c_in: means input dim
        d_model: means output dim
        model=DataEmbedding(8,256)(x)
        x=torch.randn(32,96,8)
        y==>(32,96,256)
        """
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
    
class Embedding(nn.Module):
    def __init__(self, patch_length=24, stride=12):
        #patch_length:24
        #stride :12 
        super(Embedding, self).__init__()
        self.p = patch_length
        self.s = stride
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=patch_length,
            kernel_size=patch_length,
            stride=stride,
            padding=0
        )
        

    def forward(self, x):
        # （B,L,M）
        B,L,M = x.shape
        print("shape of x at the beginning", x.shape)
        x = x.reshape(B*M,L,1) #(B,L,M)==>(B*M,L,1)
        x = x.permute(0, 2, 1) #(B*M,1,L)

        print("shape of x before conv", x.shape)
        x_pad = F.pad(
            x,
            pad=(0, self.p-self.s),
            mode='replicate'
            )  #(B*M,1,L-S+1)
        print("shape of x after padding", x_pad.shape) #(B*M,1,L-P+1)
        x_emb = self.conv1d(x_pad) 

        print("shape of x after conv", x_emb.shape)  #(B*M,P/2,L-P+1)
        
        x_emb = x_emb.permute(0, 2, 1).unsqueeze(-1)
        x_emb = x_emb.reshape(B,x_emb.shape[1],-1,M)
        
        x_emb_odd = x_emb[:, :, 0::2, :] 
        x_emb_even = x_emb[:, :, 1::2, :]
        
        x_emb = torch.stack([x_emb_odd,x_emb_even],axis=2) #(B,N,2,P/2,M)
        
        return x_emb

class TimesBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, num_period=4):
    #def __init__(self,configs,stride=1, padding=0, dilation=1, groups=1, bias=True, num_period=4):
        super(TimesBlock, self).__init__()
        
        kernel_size = kernel_size #(1,3,3)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        
        assert stride[0] == 1
        assert padding[0] == 0
        assert dilation[0] == 1

#         self.in_channels = configs.d_model
#         self.out_channels = configs.d_model
#         self.kernel_size = configs.kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_period = num_period

        # Dynamic weights
        self.Wi = nn.Parameter(torch.Tensor(1, out_channels, in_channels, kernel_size[1], kernel_size[2]))
        nn.init.kaiming_uniform_(self.Wi, a=math.sqrt(5))

        # Inter Period modelling
        self.inter_period_pool = nn.AdaptiveAvgPool3d((None, 1, 1)) # C x T x 1 x 1 T= num_period
        self.period_conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.intra_period_pool = nn.AdaptiveAvgPool1d(1)
        # 1D conv to generate alpha
        self.alpha_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.Wi)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        b, c, t, h, w = x.size()  #(B,C,T,2,12)  odd_even split (B,C,T,2,P/2)
        res = x
        # Inter period modelling
        peri_pool = self.inter_period_pool(x).view(b, c, t) #(B,C,T,2,12)==>(B,C,T)
        peri_pool = self.period_conv(peri_pool)#(B,C,T)
        peri_pool = self.bn_relu(peri_pool)#(B,C,T)
        
       
        # Combine spatial and temporal pools
        pooled = peri_pool #(B,C,T)
        pooled = self.intra_period_pool(pooled) #(B,C,T)==>(B,C,1)
        
        # Generate alpha
        alpha = self.alpha_conv(pooled).view(b, self.out_channels, 1, 1, 1)+1
        # Generate dynamic weights
        W_dynamic = self.Wi * alpha
        
        
        
        # Reshape x for 2D convolution
        x = x.view(b, c, t,h*w) #(B,C,T,2*L/2)
        # Perform convolution
        W_dynamic = W_dynamic.view(b * self.out_channels, c, self.kernel_size[1], self.kernel_size[2])
        output = F.conv2d(x, weight=W_dynamic, stride=self.stride[1:], padding=self.padding[1:], dilation=self.dilation[1:], groups=self.groups)
        output = F.conv_transpose2d(output, weight=W_dynamic, stride=self.stride[1:], padding=self.padding[1:], dilation=self.dilation[1:], groups=self.groups)
        output = output.view(b, self.out_channels, t, h, w)
        
        output = output+res
        
        return output
    
class Model(nn.Module):
    def __init__(self, configs):
    #def __init__(self, configs):
        super(Model, self).__init__()
#         self.configs = configs
#         self.task_name = configs.task_name
#         self.seq_len = configs.seq_len
#         self.label_len = configs.label_len
#         self.pred_len = configs.pred_len

        """
parameter={
    "seq_len":96,
    "pred_len":32,
    "d_model":64,
    "task_name":'long_term_forecast',
    "c_in":7,
    "c_out":7,
    "layers":3,
    "patch_length":24,
    "stride":12
}
        """
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs.d_model,configs.d_model,(1,3,3))
                                    for _ in range(configs.layers)])  #Block stack
        self.task_name = configs.task_name
        self.layers = configs.layers
        self.enc_embedding = DataEmbedding(configs.c_in,configs.d_model)  #Embedding (B,L,C)==>(B,L,d_model)
        self.thd_reshape = Embedding(configs.patch_length,configs.stride)   #3D-Reshape #
        #self.layer_norm = nn.LayerNorm(parameter["d_model"])
        
        self.fc1 = nn.Linear(2*configs.seq_len,configs.pred_len)
        self.fc2 = nn.Linear(configs.d_model,configs.c_out)
        
    
    def forecast(self,x):
        
        #forecast
        x = self.enc_embedding(x)
        B,_,C = x.size()
        x = self.thd_reshape(x) 
        x = x.permute(0,4,1,2,3) #(B,N,2,P/2,M) -> (B,M,N,2,P/2)
        for i in range(self.layers):
            x = self.model[i](x)
            #print(x.shape)
            
        x = x.contiguous().view(B,-1,C)
        _,L,_ = x.size()
        
        x = self.fc1(x.permute(0,2,1))
        x = self.fc2(x.permute(0,2,1))
        
        return x
    
    
    
    def forward(self,x):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            out=self.forecast(x)
            return out 
        
        