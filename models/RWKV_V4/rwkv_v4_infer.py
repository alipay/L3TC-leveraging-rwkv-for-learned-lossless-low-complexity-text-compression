import torch
import math
import numbers
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from ..registry import MODULE_BUILD_FUNCS
from torch import Tensor, Size
from typing import Union, List
_shape_t = Union[int, List[int], Size]

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        u = torch.mean(input, dim=-1, keepdim=True)
        s = torch.mean(input * input, dim=-1, keepdim=True)
        s = torch.sqrt(s - u * u + self.eps)
        x_normalized = (input - u) / s
        output = x_normalized * self.weight + self.bias
        return output


class RWKV_ChannelMix(nn.Module):
    def __init__(self, n_embed, ffn_dim):
        super().__init__()

        self.time_mix_k = nn.Parameter(torch.ones(1, n_embed))
        self.time_mix_r = nn.Parameter(torch.ones(1, n_embed))

        self.key = nn.Linear(n_embed, ffn_dim, bias=False)
        self.receptance = nn.Linear(n_embed, n_embed, bias=False)
        self.value = nn.Linear(ffn_dim, n_embed, bias=False)

    def forward(self, x, state_ffn):
        xk = x * self.time_mix_k + state_ffn * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + state_ffn * (1 - self.time_mix_r)
        new_ffn = x

        r = torch.sigmoid(self.receptance(xr))
        k = torch.square(torch.relu(self.key(xk)))
        kv = self.value(k)
        
        rkv = r * kv
        return rkv, new_ffn


class RWKV_TimeMix(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.time_decay = nn.Parameter(torch.ones(n_embed))
        self.time_first = nn.Parameter(torch.ones(n_embed) * math.log(0.3))
        
        self.time_mix_k = nn.Parameter(torch.ones(1, n_embed))
        self.time_mix_v = nn.Parameter(torch.ones(1, n_embed))
        self.time_mix_r = nn.Parameter(torch.ones(1, n_embed))

        self.key = nn.Linear(n_embed, n_embed, bias=False)
        self.value = nn.Linear(n_embed, n_embed, bias=False)
        self.receptance = nn.Linear(n_embed, n_embed, bias=False)

        self.output = nn.Linear(n_embed, n_embed, bias=False)

    def forward(self, x, state_A, state_B, state_p, state_x):
        xk = x * self.time_mix_k + state_x * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + state_x * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + state_x * (1 - self.time_mix_r)
        new_x = x

        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))

        ww = self.time_first + k
        p = torch.maximum(state_p, ww)
        e1 = torch.exp(state_p - p)
        e2 = torch.exp(ww - p)
        a = e1 * state_A + e2 * v
        b = e1 * state_B + e2

        ww = state_p + -torch.exp(self.time_decay)
        p = torch.maximum(ww, k)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)
        new_A = e1 * state_A + e2 * v
        new_B = e1 * state_B + e2
        new_p = p

        rwkv = r * a / b
        rwkv = self.output(rwkv)
        return rwkv, new_A, new_B, new_p, new_x


class RWKV_TimeMix_ONNX(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.time_decay = nn.Parameter(torch.ones(n_embed))
        self.time_first = nn.Parameter(torch.ones(n_embed) * math.log(0.3))
        
        self.time_mix_k = nn.Parameter(torch.ones(1, n_embed))
        self.time_mix_v = nn.Parameter(torch.ones(1, n_embed))
        self.time_mix_r = nn.Parameter(torch.ones(1, n_embed))

        self.key = nn.Linear(n_embed, n_embed, bias=False)
        self.value = nn.Linear(n_embed, n_embed, bias=False)
        self.receptance = nn.Linear(n_embed, n_embed, bias=False)

        self.output = nn.Linear(n_embed, n_embed, bias=False)

    def forward(self, x, state_A, state_B, state_p, state_x):
        xk = x * self.time_mix_k + state_x * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + state_x * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + state_x * (1 - self.time_mix_r)
        new_x = x

        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))

        ww = self.time_first + k
        # p = torch.maximum(state_p, ww)
        p = torch.stack([state_p.flatten(), ww.flatten()]).max(dim=0)[0].view(state_p.shape)
        # p = torch.where(state_p > ww, state_p, ww)

        e1 = torch.exp(state_p - p)
        e2 = torch.exp(ww - p)
        a = e1 * state_A + e2 * v
        b = e1 * state_B + e2

        ww = state_p + -torch.exp(self.time_decay)
        # p = torch.maximum(ww, k)
        p = torch.stack([ww.flatten(), k.flatten()]).max(dim=0)[0].view(state_p.shape)
        # p = torch.where(ww > k, ww, k)

        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)
        new_A = e1 * state_A + e2 * v
        new_B = e1 * state_B + e2
        new_p = p

        rwkv = r * a / b
        rwkv = self.output(rwkv)
        return rwkv, new_A, new_B, new_p, new_x


class Block(nn.Module):
    def __init__(self, layer_id, n_embed, ffn_dim):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embed)
        
        self.att = RWKV_TimeMix(n_embed)
        self.ffn = RWKV_ChannelMix(n_embed, ffn_dim)

    def forward(self, x, state_A, state_B, state_p, state_x, state_ffn):
        if self.layer_id == 0:
            x = self.ln0(x)

        short_cut = x
        x = self.ln1(x)
        x, new_A, new_B, new_p, new_x = self.att(x, state_A, state_B, state_p, state_x)
        x = short_cut + x

        short_cut = x
        x = self.ln2(x)
        x, new_ffn = self.ffn(x, state_ffn)
        x = short_cut + x
        return x, new_A, new_B, new_p, new_x, new_ffn


class Block_ONNX(nn.Module):
    def __init__(self, layer_id, n_embed, ffn_dim):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embed)
        
        self.att = RWKV_TimeMix_ONNX(n_embed)
        self.ffn = RWKV_ChannelMix(n_embed, ffn_dim)

    def forward(self, x, state_A, state_B, state_p, state_x, state_ffn):
        if self.layer_id == 0:
            x = self.ln0(x)

        short_cut = x
        x, new_A, new_B, new_p, new_x = self.att(self.ln1(x), state_A, state_B, state_p, state_x)
        x = short_cut + x

        short_cut = x
        x, new_ffn = self.ffn(self.ln2(x), state_ffn)
        x = short_cut + x
        return x, new_A, new_B, new_p, new_x, new_ffn


class Block_Script(nn.Module):
    def __init__(self, n_embed, ffn_dim):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
        self.att = RWKV_TimeMix(n_embed)
        self.ffn = RWKV_ChannelMix(n_embed, ffn_dim)

    def forward(self, x, state_A, state_B, state_p, state_x, state_ffn):    
        short_cut = x
        x, new_A, new_B, new_p, new_x = self.att(self.ln1(x), state_A, state_B, state_p, state_x)
        x = short_cut + x

        short_cut = x
        x, new_ffn = self.ffn(self.ln2(x), state_ffn)
        x = short_cut + x
        return x, new_A, new_B, new_p, new_x, new_ffn


class RWKV_V4_Infer_For_CoreML(nn.Module):
    def __init__(self,
                 vocab_size=2000,
                 hidden_size=512,
                 num_hidden_layers=4,
                 intermediate_size=1024,
                 ):
        super(RWKV_V4_Infer_For_CoreML, self).__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList([Block(i, hidden_size, intermediate_size) for i in range(num_hidden_layers)])
        self.ln_out = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward_initialzation(self, batch_size, device):
        state_A = torch.zeros([self.num_hidden_layers, batch_size, self.hidden_size])
        state_B = torch.zeros([self.num_hidden_layers, batch_size, self.hidden_size])
        state_p = torch.zeros([self.num_hidden_layers, batch_size, self.hidden_size]) - 1e30
        state_x = torch.zeros([self.num_hidden_layers, batch_size, self.hidden_size])
        state_ffn = torch.zeros([self.num_hidden_layers, batch_size, self.hidden_size])
        hidden_state = torch.stack([state_A, state_B, state_p, state_x, state_ffn]).to(device)
        return hidden_state
        
    def forward(self, x, hidden_state):
        # x = self.emb(input_token)
        # x = torch.matmul(input_onehot, self.emb.weight)

        batch_size = x.size(0)
        state_A, state_B, state_p, state_x, state_ffn = hidden_state.split(1, dim=0)
        new_hidden_state = []

        for i, block in enumerate(self.blocks):
            x, new_A, new_B, new_p, new_x, new_ffn = \
                block(x, state_A[0, i], state_B[0, i], state_p[0, i], state_x[0, i], state_ffn[0, i])

            new_hidden_state.append(new_A)
            new_hidden_state.append(new_B)
            new_hidden_state.append(new_p)
            new_hidden_state.append(new_x)
            new_hidden_state.append(new_ffn)

        new_hidden_state = torch.cat(new_hidden_state)
        new_hidden_state = new_hidden_state.view([self.num_hidden_layers, 5, batch_size, self.hidden_size])
        new_hidden_state = new_hidden_state.transpose(0, 1)
        x = self.ln_out(x)
        x = self.head(x)
        return x, new_hidden_state


class RWKV_V4_Infer_For_ONNX(nn.Module):
    def __init__(self,
                 vocab_size=2000,
                 hidden_size=512,
                 num_hidden_layers=4,
                 intermediate_size=1024,
                 ):
        super(RWKV_V4_Infer_For_ONNX, self).__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList([Block_ONNX(i, hidden_size, intermediate_size) for i in range(num_hidden_layers)])
        self.ln_out = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward_initialzation(self, batch_size, device):
        state_A = torch.zeros([self.num_hidden_layers, batch_size, self.hidden_size])
        state_B = torch.zeros([self.num_hidden_layers, batch_size, self.hidden_size])
        state_p = torch.zeros([self.num_hidden_layers, batch_size, self.hidden_size]) - 1e30
        state_x = torch.zeros([self.num_hidden_layers, batch_size, self.hidden_size])
        state_ffn = torch.zeros([self.num_hidden_layers, batch_size, self.hidden_size])
        hidden_state = torch.stack([state_A, state_B, state_p, state_x, state_ffn]).to(device)
        return hidden_state

    def forward(self, x, hidden_state):
        # x = self.emb(input_token)
        batch_size = x.size(0)
        # x = torch.matmul(input_onehot, self.emb.weight)
        state_A, state_B, state_p, state_x, state_ffn = hidden_state.split(1, dim=0)
        new_hidden_state = []

        for i, block in enumerate(self.blocks):
            x, new_A, new_B, new_p, new_x, new_ffn = \
                block(x, state_A[0, i], state_B[0, i], state_p[0, i], state_x[0, i], state_ffn[0, i])

            new_hidden_state.append(new_A)
            new_hidden_state.append(new_B)
            new_hidden_state.append(new_p)
            new_hidden_state.append(new_x)
            new_hidden_state.append(new_ffn)

        new_hidden_state = torch.cat(new_hidden_state)
        new_hidden_state = new_hidden_state.view([self.num_hidden_layers, 5, batch_size, self.hidden_size])
        new_hidden_state = new_hidden_state.transpose(0, 1)
        x = self.ln_out(x)
        x = self.head(x)
        return x, new_hidden_state


class RWKV_V4_Infer_For_Script(nn.Module):
    def __init__(self,
                 vocab_size=2000,
                 hidden_size=512,
                 num_hidden_layers=4,
                 intermediate_size=1024,
                 ):
        super(RWKV_V4_Infer_For_Script, self).__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.ln0 = nn.LayerNorm(hidden_size)
        self.blocks = nn.ModuleList([Block_Script(hidden_size, intermediate_size) for i in range(num_hidden_layers)])
        self.ln_out = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward_initialzation(self, batch_size, device):
        state_A = torch.zeros([self.num_hidden_layers, batch_size, self.hidden_size])
        state_B = torch.zeros([self.num_hidden_layers, batch_size, self.hidden_size])
        state_p = torch.zeros([self.num_hidden_layers, batch_size, self.hidden_size]) - 1e30
        state_x = torch.zeros([self.num_hidden_layers, batch_size, self.hidden_size])
        state_ffn = torch.zeros([self.num_hidden_layers, batch_size, self.hidden_size])
        hidden_state = torch.stack([state_A, state_B, state_p, state_x, state_ffn]).to(device)
        return hidden_state

    def forward(self, input_token, hidden_state):
        x = self.emb(input_token)
        batch_size = input_token.size(0)
        # x = torch.matmul(input_onehot, self.emb.weight)
        state_A, state_B, state_p, state_x, state_ffn = hidden_state.split(1, dim=0)
        new_hidden_state = []

        x = self.ln0(x)
        for i, block in enumerate(self.blocks):
            x, new_A, new_B, new_p, new_x, new_ffn = \
                block(x, state_A[0, i], state_B[0, i], state_p[0, i], state_x[0, i], state_ffn[0, i])

            new_hidden_state.append(new_A)
            new_hidden_state.append(new_B)
            new_hidden_state.append(new_p)
            new_hidden_state.append(new_x)
            new_hidden_state.append(new_ffn)

        new_hidden_state = torch.cat(new_hidden_state)
        new_hidden_state = new_hidden_state.view([self.num_hidden_layers, 5, batch_size, self.hidden_size])
        new_hidden_state = new_hidden_state.transpose(0, 1)
        x = self.ln_out(x)
        x = self.head(x)
        return x, new_hidden_state


@MODULE_BUILD_FUNCS.registe_with_name(module_name='rwkv_v4_infer_for_coreml')
def build_rwkv_v4_infer_for_coreml(args):
    model = RWKV_V4_Infer_For_CoreML(
        vocab_size = args.vocab_size,
        hidden_size = args.hidden_size,
        num_hidden_layers = args.num_hidden_layer,
        intermediate_size = args.intermediate_size
    )
    criterion = nn.CrossEntropyLoss(reduction='none')
    return model, criterion


@MODULE_BUILD_FUNCS.registe_with_name(module_name='rwkv_v4_infer_for_onnx')
def build_rwkv_v4_infer_for_onnx(args):
    model = RWKV_V4_Infer_For_ONNX(
        vocab_size = args.vocab_size,
        hidden_size = args.hidden_size,
        num_hidden_layers = args.num_hidden_layer,
        intermediate_size = args.intermediate_size
    )
    criterion = nn.CrossEntropyLoss(reduction='none')
    return model, criterion


@MODULE_BUILD_FUNCS.registe_with_name(module_name='rwkv_v4_infer_for_script')
def build_rwkv_v4_infer_for_script(args):
    model = RWKV_V4_Infer_For_Script(
        vocab_size = args.vocab_size,
        hidden_size = args.hidden_size,
        num_hidden_layers = args.num_hidden_layer,
        intermediate_size = args.intermediate_size
    )
    criterion = nn.CrossEntropyLoss(reduction='none')
    return model, criterion