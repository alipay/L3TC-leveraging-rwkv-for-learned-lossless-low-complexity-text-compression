########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import math, os
import numpy as np
import logging
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.nn import functional as F
from deepspeed.ops.adam import FusedAdam
from ..registry import MODULE_BUILD_FUNCS

logger = logging.getLogger(__name__)


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss
    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)

# 'bf16' (fast & stable)
# 'fp16' (fast & will overflow after training a large model for very long. can be solved in the future)
# 'tf32' (decent speed & stable)
# 'fp32' (!!!very slow!!! only for verification)
os.environ['RWKV_FLOAT_MODE'] = 'fp16'

########################################################################################################
# CUDA Kernel
########################################################################################################

T_MAX = 2048 # increase this if your ctx_len is long [NOTE: TAKES LOTS OF VRAM!]
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

if 'USE_WKV_CUDA_FOR_RWKV' in os.environ and os.environ['USE_WKV_CUDA_FOR_RWKV'] == 'True':  # if get error caused by wkv_cuda, set False
    from torch.utils.cpp_extension import load
    wkv_cuda = load(name="wkv", sources=["models/RWKV_V4/cuda/wkv_op.cpp", "models/RWKV_V4/cuda/wkv_cuda.cu"],
                    verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 2048) == 0
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            w = -torch.exp(w.contiguous())
            u = u.contiguous()
            k = k.contiguous()
            v = v.contiguous()
        else:
            w = -torch.exp(w.float().contiguous())
            u = u.float().contiguous()
            k = k.float().contiguous()
            v = v.float().contiguous()
        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            return y
        elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
            return y.half()
        elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
            return y.bfloat16()

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 2048) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            wkv_cuda.backward(B, T, C, w, u, k, v, gy.contiguous(), gw, gu, gk, gv)
        else:
            wkv_cuda.backward(B, T, C, w, u, k, v, gy.float().contiguous(), gw, gu, gk, gv)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            return (None, None, None, gw, gu, gk, gv)
        elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())

def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())

########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################

def RWKV_Init(model, vocab_size, n_embed):  # fancy initialization of all lin & emb layer in the model
    print("\n[--> first run, init model params (very slow for large models) <--]")
    print("[so you shall only do it for 1 single GPU and save the checkpt and load it when using multiple GPU]\n")

    for mm in model.modules():
        if "RecursiveScriptModule" in str(type(mm)):
            if mm.original_name not in ["Linear"]:
                continue
            ww = None
            for name, param in mm.named_parameters():
                if name == "weight":
                    ww = param
        else:
            m = mm
            if not isinstance(m, (nn.Linear, nn.Embedding)):
                continue
            ww = m.weight
        with torch.no_grad():
            name = "[unknown weight]"
            for name, parameter in model.named_parameters():  # find the name of the weight
                if id(ww) == id(parameter):
                    break

            shape = ww.shape
            gain = 1.0
            scale = 1.0  # extra scale for gain

            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                if shape[0] == vocab_size and shape[1] == n_embed:  # token emb?
                    scale = 1e-4
                else:
                    scale = 0

            if isinstance(m, nn.Linear):
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                if shape[0] == vocab_size and shape[1] == n_embed:  # final projection?
                    scale = 0.5

            if hasattr(m, "scale_init"):
                scale = m.scale_init

            # print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {name}")

            gain *= scale
            if scale == -999:
                nn.init.eye_(ww)
            elif gain == 0:
                # zero init is great for some RWKV matrices
                nn.init.zeros_(ww)
            elif gain > 0:
                nn.init.orthogonal_(ww, gain=gain)
            else:
                nn.init.normal_(ww, mean=0.0, std=-scale)


class RWKV_TimeMix(torch.jit.ScriptModule):
    def __init__(self, n_embed, ctx_len, n_layer, layer_id, dropout_prob):
        super().__init__()
        self.layer_id = layer_id
        self.ctx_len = ctx_len
        self.n_embed = n_embed
        self.n_layer = n_layer
        self._states = {}

        attn_sz = n_embed

        with torch.no_grad(): # fancy init
            ratio_0_to_1 = (layer_id / (n_layer - 1)) # 0 to 1
            ratio_1_to_almost0 = (1.0 - (layer_id / n_layer)) # 1 to ~0
            
            # fancy time_decay
            decay_speed = torch.ones(attn_sz)
            for h in range(attn_sz):
                decay_speed[h] = -5 + 8 * (h / (attn_sz-1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(attn_sz)]) * 0.5)
            self.time_first = nn.Parameter(torch.ones(attn_sz) * math.log(0.3) + zigzag)
            
            # fancy time_mix
            x = torch.ones(1, 1, n_embed)
            for i in range(n_embed):
                x[0, 0, i] = i / n_embed
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
            
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(n_embed, attn_sz, bias=False)
        self.value = nn.Linear(n_embed, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embed, attn_sz, bias=False)

        self.dropout = nn.Dropout(dropout_prob)
        self.output = nn.Linear(attn_sz, n_embed, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

        self.infer_state_init()

    @torch.jit.script_method
    def jit_func(self, x):
        # Mix x with the previous timestep to produce xk, xv, xr
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v
    
    def infer_state_init(self):
        self._states.clear()

    def forward(self, x, train=True):
        if train:
            return self.forward_train(x)
        else:
            return self.forward_test(x)

    def forward_train(self, x):
        B, T, C = x.size() # x = (Batch,Time,Channel)
        sr, k, v = self.jit_func(x)

        rwkv = sr * RUN_CUDA(B, T, C, self.time_decay, self.time_first, k, v)
        rwkv = self.dropout(rwkv)
        rwkv = self.output(rwkv)
        return rwkv

    def forward_test(self, x):
        batch_size = x.size(0)

        if 'state_A' not in self._states:
            self._states['state_A'] = torch.zeros([batch_size, self.n_embed], device=x.device)
        if 'state_B' not in self._states:
            self._states['state_B'] = torch.zeros([batch_size, self.n_embed], device=x.device)
        if 'state_p' not in self._states:
            self._states['state_p'] = torch.zeros([batch_size, self.n_embed], device=x.device) - 1e30
        if 'state_x' not in self._states:
            self._states['state_x'] = torch.zeros([batch_size, self.n_embed], device=x.device)

        xk = x * self.time_mix_k.squeeze() + self._states['state_x'] * (1 - self.time_mix_k.squeeze())
        xv = x * self.time_mix_v.squeeze() + self._states['state_x'] * (1 - self.time_mix_v.squeeze())
        xr = x * self.time_mix_r.squeeze() + self._states['state_x'] * (1 - self.time_mix_r.squeeze())
        self._states['state_x'] = x

        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))

        ww = self.time_first + k
        p = torch.maximum(self._states['state_p'], ww)
        e1 = torch.exp(self._states['state_p'] - p)
        e2 = torch.exp(ww - p)
        a = e1 * self._states['state_A'] + e2 * v
        b = e1 * self._states['state_B'] + e2

        ww = self._states['state_p'] + -torch.exp(self.time_decay)
        p = torch.maximum(ww, k)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)
        self._states['state_A'] = e1 * self._states['state_A'] + e2 * v
        self._states['state_B'] = e1 * self._states['state_B'] + e2
        self._states['state_p'] = p

        rwkv = r * a / b
        rwkv = self.output(rwkv)
        return rwkv


class RWKV_ChannelMix(torch.jit.ScriptModule):
    def __init__(self, n_embed, ffn_dim, n_layer, layer_id, dropout_prob):
        super().__init__()
        self.n_embed = n_embed
        self.ffn_dim = ffn_dim
        self.n_layer = n_layer
        self.layer_id = layer_id
        self._states = {}

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad(): # fancy init of time_mix
            ratio_1_to_almost0 = (1.0 - (layer_id / n_layer)) # 1 to ~0

            x = torch.ones(1, 1, n_embed)
            for i in range(n_embed):
                x[0, 0, i] = i / n_embed

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        self.key = nn.Linear(n_embed, ffn_dim, bias=False)
        self.receptance = nn.Linear(n_embed, n_embed, bias=False)
        self.value = nn.Linear(ffn_dim, n_embed, bias=False)
        self.dropout = nn.Dropout(dropout_prob)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

        self.infer_state_init()

    def infer_state_init(self):
        self._states.clear()

    def forward(self, x, train=True):
        if train:
            x = self.forward_train(x)
        else:
            x = self.forward_test(x)
        return x

    @torch.jit.script_method
    def forward_train(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)

        rkv = torch.sigmoid(self.receptance(xr)) * kv
        rkv = self.dropout(rkv)
        return rkv

    def forward_test(self, x):
        batch_size = x.size(0)
        if 'state_ffn' not in self._states:
            self._states['state_ffn'] = torch.zeros([batch_size, self.n_embed], device=x.device)

        xk = x * self.time_mix_k.squeeze() + self._states['state_ffn'] * (1 - self.time_mix_k.squeeze())
        xr = x * self.time_mix_r.squeeze() + self._states['state_ffn'] * (1 - self.time_mix_r.squeeze())
        self._states['state_ffn'] = x

        r = torch.sigmoid(self.receptance(xr))
        k = torch.square(torch.relu(self.key(xk)))
        kv = self.value(k)
        
        rkv = r * kv
        return rkv


########################################################################################################
# The GPT Model with our blocks
########################################################################################################

class Block(nn.Module):
    def __init__(self, n_embed, ffn_dim, n_layer, ctx_len, layer_id, dropout_prob):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embed)

        self.att = RWKV_TimeMix(n_embed, ctx_len, n_layer, layer_id, dropout_prob)
        self.ffn = RWKV_ChannelMix(n_embed, ffn_dim, n_layer, layer_id, dropout_prob)
        self.short = nn.Linear(n_embed, n_embed, bias=False)

        self.infer_state_init()

    def infer_state_init(self):
        self.att.infer_state_init()
        self.ffn.infer_state_init()

    def forward(self, x, train=True):
        if train:
            x = self.forward_train(x)
        else:
            x = self.forward_test(x)
        return x

    def forward_train(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)

        short = F.relu(self.short(x))
        x = x + self.att(self.ln1(x), train=True)
        x = x + self.ffn(self.ln2(x), train=True)
        x = x + short
        return x

    def forward_test(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)
        
        short = F.relu(self.short(x))
        x = x + self.att(self.ln1(x), train=False)
        x = x + self.ffn(self.ln2(x), train=False)
        x = x + short
        return x


class Point_RWKV(nn.Module):
    def __init__(self,
                 vocab_size = 2000,
                 hidden_size = 512,
                 num_hidden_layers = 4,
                 intermediate_size = 1024,
                 ctx_len = 1024,
                 dropout_prob = 0.0
                ):
        super(Point_RWKV, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.ctx_len = ctx_len
        self.dropout_prob = dropout_prob

        self.emb = nn.Embedding(self.vocab_size, self.hidden_size)
        self.blocks = nn.ModuleList([Block(hidden_size, intermediate_size, num_hidden_layers, ctx_len, i, dropout_prob)
                                    for i in range(num_hidden_layers)])
        self.ln_out = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
   
        RWKV_Init(self, vocab_size, hidden_size)
        logger.info("number of parameters: %e", sum(p.numel()
                    for p in self.parameters()))
        
        self.infer_state_init()

    def get_ctx_len(self):
        return self.ctx_len

    def configure_optimizers(self, args):
        no_decay = set()

        for mn, m in self.named_modules():  # here we disable weight_decay
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        try:
            optimizer = FusedAdam(optim_groups, lr=args.learning_rate, betas=args.betas, eps=args.eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        except:
            print('\n\nDeepSpeed not found. Using torch optimizer instead (probably slower)\n\n')
            optimizer = torch.optim.Adam(optim_groups, lr=args.learning_rate, betas=args.betas, eps=args.eps)

        return optimizer

    def infer_state_init(self):
        for block in self.blocks:
            block.infer_state_init()

    def forward(self, input_token, input_types, train=True, output_token=None, output_types=None, criterion=None):
        if train:
            return self.forward_train(input_token)
        else:
            return self.forward_test(input_token, output_token, output_types, criterion)
    
    def forward_train(self, input_token):
        input_token = input_token.squeeze(dim=1)
        B, T = input_token.size()
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."

        x = self.emb(input_token)
        for block_id, block in enumerate(self.blocks):
            x = block(x, train=True)
        x = self.ln_out(x)
        x = self.head(x)
        return x

    def forward_test(self, input_token, output_token, output_types, criterion):
        self.infer_state_init()

        sum_cross_entropy = 0
        hit_count = 0
        token_count = 0
        output = dict()

        input_token = input_token.flatten(1, 2)
        output_token = output_token.flatten(1, 2)
        output_types = output_types.flatten(1, 2)
        B, T = input_token.size()

        pbar = tqdm(total=T)
        for token_id in range(T):
            idx = input_token[:, token_id]

            x = self.emb(idx)
            for block_id, block in enumerate(self.blocks):
                x = block(x, train=False)
            x = self.ln_out(x)
            x = self.head(x)

            x = x.view([-1, x.size(-1)])
            gt = output_token[:, token_id]
            pred = torch.argmax(x, dim=-1)

            cross_entropy = criterion(x, gt)
            token_types = output_types[:, token_id]
            cross_entropy = cross_entropy * token_types
            sum_cross_entropy += cross_entropy.sum().cpu().numpy()

            hit_count += ((pred == gt) * token_types).sum().cpu().numpy()
            token_count += token_types.sum().cpu().numpy()

            pbar.update()
            pbar.set_description("Avg_cross_entropy: {}".format(sum_cross_entropy / token_count))
        pbar.close()
        
        output['cross_entropy'] = sum_cross_entropy
        output['hit_count'] = hit_count
        output['token_count'] = token_count
        return output


@MODULE_BUILD_FUNCS.registe_with_name(module_name='prwkv')
def build_point_rwkv(args):
    model = Point_RWKV(
        vocab_size = args.vocab_size,
        hidden_size = args.hidden_size,
        num_hidden_layers = args.num_hidden_layer,
        intermediate_size = args.intermediate_size,
        ctx_len = args.ctx_len,
        dropout_prob = args.dropout,
    )
    criterion = nn.CrossEntropyLoss(reduction='none')
    return model, criterion