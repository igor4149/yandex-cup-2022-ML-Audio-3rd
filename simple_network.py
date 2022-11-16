import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, :, None, :]
            self.sin_cached = emb.sin()[None, :, None, :]
        return self.cos_cached, self.sin_cached


# rotary pos emb helpers:

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in torch < 1.8.0


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    
    

class AttentionAggregation(nn.Module):
    def __init__(self, head=4, d_model=512):
        super().__init__()
        self.head = head
        self.d_model = d_model
        self.attention_transformation = nn.MultiheadAttention(self.d_model, self.head, batch_first=True)
        self.penc = Rotary(d_model // self.head)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.fc = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.SiLU(), nn.Linear(d_model * 4, d_model))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x_in = x
        pos_cos, pos_sin = self.penc(x)
        x_headed = x.reshape(x.shape[0], x.shape[1], self.head, x.shape[2] // self.head)
        xq, xk = apply_rotary_pos_emb(x_headed, x_headed, pos_cos, pos_sin)
        xq = xq.reshape(*x.shape); xk = xk.reshape(*x.shape)
        x = self.dropout(x)
        x, _ = self.attention_transformation(xq, xk, x, need_weights=False)
        x = self.dropout(x)
        x = x + x_in
        x = self.ln1(x)
        x = self.dropout(self.fc(x)) + x
        x = self.ln2(x)
        return x
        
        
        
class Net(nn.Module):
    def __init__(self, input_dim, emb_dim, n_att=2, n_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_att = n_att
        self.att = nn.ModuleList([AttentionAggregation(head=4, d_model=input_dim) for _ in range(self.n_att)])
        self.fc = nn.Sequential(nn.Linear(input_dim, emb_dim), nn.PReLU())
        self.sap_linear = nn.ModuleList([nn.Linear(emb_dim // self.n_heads, emb_dim // self.n_heads) for _ in range(self.n_heads)])
        self.attention = nn.ParameterList([self.new_parameter(emb_dim // self.n_heads, 1) for _ in range(self.n_heads)])
        self.out = nn.Linear(emb_dim, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)


    def forward(self, x):
        #x = (b, 512, 81)
        x = x.transpose(1, 2)
        for i in range(self.n_att):
            x = self.att[i](x)
        x = self.fc(x)
        x_chunks = torch.chunk(x, self.n_heads, dim=-1)
        heads_out = []
        for i in range(self.n_heads):
            h = torch.tanh(self.sap_linear[i](x_chunks[i]))
            w = torch.matmul(h, self.attention[i]).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x_chunks[i].size(0), x_chunks[i].size(1), 1)
            x_cur = torch.sum(x_chunks[i] * w, dim=1)
            heads_out.append(x_cur)
            
        x = torch.cat(heads_out, dim=-1)
        x = self.out(x)
        x = self.bn(x)
        return x

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

