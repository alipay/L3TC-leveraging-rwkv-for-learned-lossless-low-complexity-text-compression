import torch
import math
import torch.nn as nn
from ..registry import MODULE_BUILD_FUNCS
from .ptq import QIntLayerNorm, QLinear, QAct, QConv2d, QIntSoftmax
from .ptq.bit_type import BIT_TYPE_DICT


class Config:

    def __init__(self, ptf=True, lis=True, quant_method='minmax'):
        '''
        ptf stands for Power-of-Two Factor activation quantization for Integer Layernorm.
        lis stands for Log-Int-Softmax.
        These two are proposed in our "FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer".
        '''
        self.BIT_TYPE_W = BIT_TYPE_DICT['int8']
        self.BIT_TYPE_A = BIT_TYPE_DICT['uint8']

        self.OBSERVER_W = 'minmax'
        self.OBSERVER_A = quant_method

        self.QUANTIZER_W = 'uniform'
        self.QUANTIZER_A = 'uniform'
        self.QUANTIZER_A_LN = 'uniform'

        self.CALIBRATION_MODE_W = 'channel_wise'
        self.CALIBRATION_MODE_A = 'layer_wise'
        self.CALIBRATION_MODE_S = 'layer_wise'

        if lis:
            self.INT_SOFTMAX = True
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint4']
            self.OBSERVER_S = 'minmax'
            self.QUANTIZER_S = 'log2'
        else:
            self.INT_SOFTMAX = False
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint8']
            self.OBSERVER_S = self.OBSERVER_A
            self.QUANTIZER_S = self.QUANTIZER_A
        if ptf:
            self.INT_NORM = True
            self.OBSERVER_A_LN = 'ptf'
            self.CALIBRATION_MODE_A_LN = 'channel_wise'
        else:
            self.INT_NORM = False
            self.OBSERVER_A_LN = self.OBSERVER_A
            self.CALIBRATION_MODE_A_LN = self.CALIBRATION_MODE_A

cfg = Config(ptf=True, lis=True, quant_method='minmax')


class RWKV_ChannelMix_For_Quant(nn.Module):
    def __init__(self, 
                 layer_id, 
                 n_embed, 
                 ffn_dim,
                 quant = False,
                 calibrate = False,
                ):
        super().__init__()
        self.layer_id = layer_id

        self.time_mix_k = nn.Parameter(torch.ones(1, n_embed))
        self.qact_time_mix_k = QAct(quant=quant,
                                    calibrate=calibrate,
                                    bit_type=cfg.BIT_TYPE_A,
                                    calibration_mode=cfg.CALIBRATION_MODE_A,
                                    observer_str=cfg.OBSERVER_A,
                                    quantizer_str=cfg.QUANTIZER_A)

        self.time_mix_r = nn.Parameter(torch.ones(1, n_embed))
        self.qact_time_mix_r = QAct(quant=quant,
                                    calibrate=calibrate,
                                    bit_type=cfg.BIT_TYPE_A,
                                    calibration_mode=cfg.CALIBRATION_MODE_A,
                                    observer_str=cfg.OBSERVER_A,
                                    quantizer_str=cfg.QUANTIZER_A)

        self.qact_xk = QAct(quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_A,
                            calibration_mode=cfg.CALIBRATION_MODE_A,
                            observer_str=cfg.OBSERVER_A,
                            quantizer_str=cfg.QUANTIZER_A)
        self.qact_xr = QAct(quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_A,
                            calibration_mode=cfg.CALIBRATION_MODE_A,
                            observer_str=cfg.OBSERVER_A,
                            quantizer_str=cfg.QUANTIZER_A)

        self.hard_sigmoid = nn.Hardsigmoid()
        self.qact_sigmoid = QAct(quant=quant,
                                 calibrate=calibrate,
                                 bit_type=cfg.BIT_TYPE_A,
                                 calibration_mode=cfg.CALIBRATION_MODE_A,
                                 observer_str=cfg.OBSERVER_A,
                                 quantizer_str=cfg.QUANTIZER_A)

        self.key = QLinear(n_embed, ffn_dim, bias=False)
        self.qact_key = QAct(quant=quant,
                             calibrate=calibrate,
                             bit_type=cfg.BIT_TYPE_A,
                             calibration_mode=cfg.CALIBRATION_MODE_A,
                             observer_str=cfg.OBSERVER_A,
                             quantizer_str=cfg.QUANTIZER_A)
        
        self.qact_square_key = QAct(quant=quant,
                                    calibrate=calibrate,
                                    bit_type=cfg.BIT_TYPE_A,
                                    calibration_mode=cfg.CALIBRATION_MODE_A,
                                    observer_str=cfg.OBSERVER_A,
                                    quantizer_str=cfg.QUANTIZER_A)

        self.receptance = QLinear(n_embed, n_embed, bias=False)
        self.qact_recept = QAct(quant=quant,
                                calibrate=calibrate,
                                bit_type=cfg.BIT_TYPE_A,
                                calibration_mode=cfg.CALIBRATION_MODE_A,
                                observer_str=cfg.OBSERVER_A,
                                quantizer_str=cfg.QUANTIZER_A)

        self.value = QLinear(ffn_dim, n_embed, bias=False)
        self.qact_val = QAct(quant=quant,
                             calibrate=calibrate,
                             bit_type=cfg.BIT_TYPE_A,
                             calibration_mode=cfg.CALIBRATION_MODE_A,
                             observer_str=cfg.OBSERVER_A,
                             quantizer_str=cfg.QUANTIZER_A)

        self.qact_rkv = QAct(quant=quant,
                             calibrate=calibrate,
                             bit_type=cfg.BIT_TYPE_A,
                             calibration_mode=cfg.CALIBRATION_MODE_A,
                             observer_str=cfg.OBSERVER_A,
                             quantizer_str=cfg.QUANTIZER_A)

    def forward(self, x, state_ffn):
        time_mix_k = self.qact_time_mix_k(self.time_mix_k)
        time_mix_r = self.qact_time_mix_r(self.time_mix_r)

        xk = x * time_mix_k + state_ffn * (1 - time_mix_k)
        xk = self.qact_xk(xk)
        xr = x * time_mix_r + state_ffn * (1 - time_mix_r)
        xr = self.qact_xr(xr)
        new_ffn = x

        r = self.receptance(xr)
        r = self.qact_recept(r)
        
        r = self.hard_sigmoid(r)
        r = self.qact_sigmoid(r)

        k = self.key(xk)
        k = self.qact_key(k)

        k = torch.square(torch.relu(k))
        k = self.qact_square_key(k)

        kv = self.value(k)
        kv = self.qact_val(kv)
        
        rkv = r * kv
        rkv = self.qact_rkv(rkv)
        return rkv, new_ffn


class RWKV_TimeMix_For_Quant(nn.Module):
    def __init__(self, 
                 layer_id, 
                 n_embed,
                 quant = False,
                 calibrate = False,
                ):
        super().__init__()
        self.layer_id = layer_id
        self.time_decay = nn.Parameter(torch.ones(n_embed))
        self.qact_time_decay = QAct(quant=quant,
                                    calibrate=calibrate,
                                    bit_type=cfg.BIT_TYPE_A,
                                    calibration_mode=cfg.CALIBRATION_MODE_A,
                                    observer_str=cfg.OBSERVER_A,
                                    quantizer_str=cfg.QUANTIZER_A)
        self.time_first = nn.Parameter(torch.ones(n_embed) * math.log(0.3))
        self.qact_time_first = QAct(quant=quant,
                                    calibrate=calibrate,
                                    bit_type=cfg.BIT_TYPE_A,
                                    calibration_mode=cfg.CALIBRATION_MODE_A,
                                    observer_str=cfg.OBSERVER_A,
                                    quantizer_str=cfg.QUANTIZER_A)
        
        self.time_mix_k = nn.Parameter(torch.ones(1, n_embed))
        self.qact_time_mix_k = QAct(quant=quant,
                                    calibrate=calibrate,
                                    bit_type=cfg.BIT_TYPE_A,
                                    calibration_mode=cfg.CALIBRATION_MODE_A,
                                    observer_str=cfg.OBSERVER_A,
                                    quantizer_str=cfg.QUANTIZER_A)
        self.time_mix_v = nn.Parameter(torch.ones(1, n_embed))
        self.qact_time_mix_v = QAct(quant=quant,
                                    calibrate=calibrate,
                                    bit_type=cfg.BIT_TYPE_A,
                                    calibration_mode=cfg.CALIBRATION_MODE_A,
                                    observer_str=cfg.OBSERVER_A,
                                    quantizer_str=cfg.QUANTIZER_A)
        self.time_mix_r = nn.Parameter(torch.ones(1, n_embed))
        self.qact_time_mix_r = QAct(quant=quant,
                                    calibrate=calibrate,
                                    bit_type=cfg.BIT_TYPE_A,
                                    calibration_mode=cfg.CALIBRATION_MODE_A,
                                    observer_str=cfg.OBSERVER_A,
                                    quantizer_str=cfg.QUANTIZER_A)
        
        self.qact_xk = QAct(quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_A,
                            calibration_mode=cfg.CALIBRATION_MODE_A,
                            observer_str=cfg.OBSERVER_A,
                            quantizer_str=cfg.QUANTIZER_A)
        self.qact_xv = QAct(quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_A,
                            calibration_mode=cfg.CALIBRATION_MODE_A,
                            observer_str=cfg.OBSERVER_A,
                            quantizer_str=cfg.QUANTIZER_A)
        self.qact_xr = QAct(quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_A,
                            calibration_mode=cfg.CALIBRATION_MODE_A,
                            observer_str=cfg.OBSERVER_A,
                            quantizer_str=cfg.QUANTIZER_A)

        self.key = QLinear(n_embed, n_embed, bias=False)
        self.qact_key = QAct(quant=quant,
                             calibrate=calibrate,
                             bit_type=cfg.BIT_TYPE_A,
                             calibration_mode=cfg.CALIBRATION_MODE_A,
                             observer_str=cfg.OBSERVER_A,
                             quantizer_str=cfg.QUANTIZER_A)

        self.value = QLinear(n_embed, n_embed, bias=False)
        self.qact_val = QAct(quant=quant,
                             calibrate=calibrate,
                             bit_type=cfg.BIT_TYPE_A,
                             calibration_mode=cfg.CALIBRATION_MODE_A,
                             observer_str=cfg.OBSERVER_A,
                             quantizer_str=cfg.QUANTIZER_A)

        self.receptance = QLinear(n_embed, n_embed, bias=False)
        self.qact_recept = QAct(quant=quant,
                                calibrate=calibrate,
                                bit_type=cfg.BIT_TYPE_A,
                                calibration_mode=cfg.CALIBRATION_MODE_A,
                                observer_str=cfg.OBSERVER_A,
                                quantizer_str=cfg.QUANTIZER_A)

        self.hard_sigmoid = nn.Hardsigmoid()
        self.qact_sigmoid = QAct(quant=quant,
                                 calibrate=calibrate,
                                 bit_type=cfg.BIT_TYPE_A,
                                 calibration_mode=cfg.CALIBRATION_MODE_A,
                                 observer_str=cfg.OBSERVER_A,
                                 quantizer_str=cfg.QUANTIZER_A)
        
        self.qact_p1 = QAct(quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_A,
                            calibration_mode=cfg.CALIBRATION_MODE_A,
                            observer_str=cfg.OBSERVER_A,
                            quantizer_str=cfg.QUANTIZER_A)
        
        self.qact_p2 = QAct(quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_A,
                            calibration_mode=cfg.CALIBRATION_MODE_A,
                            observer_str=cfg.OBSERVER_A,
                            quantizer_str=cfg.QUANTIZER_A)

        self.qact_rwkv = QAct(quant=quant,
                              calibrate=calibrate,
                              bit_type=cfg.BIT_TYPE_A,
                              calibration_mode=cfg.CALIBRATION_MODE_A,
                              observer_str=cfg.OBSERVER_A,
                              quantizer_str=cfg.QUANTIZER_A)

        self.output = QLinear(n_embed, n_embed, bias=False)
        self.qact_out = QAct(quant=quant,
                             calibrate=calibrate,
                             bit_type=cfg.BIT_TYPE_A,
                             calibration_mode=cfg.CALIBRATION_MODE_A,
                             observer_str=cfg.OBSERVER_A,
                             quantizer_str=cfg.QUANTIZER_A)

        self.qact_new_A = QAct(quant=quant,
                               calibrate=calibrate,
                               bit_type=cfg.BIT_TYPE_A,
                               calibration_mode=cfg.CALIBRATION_MODE_A,
                               observer_str=cfg.OBSERVER_A,
                               quantizer_str=cfg.QUANTIZER_A)
        self.qact_new_B = QAct(quant=quant,
                               calibrate=calibrate,
                               bit_type=cfg.BIT_TYPE_A,
                               calibration_mode=cfg.CALIBRATION_MODE_A,
                               observer_str=cfg.OBSERVER_A,
                               quantizer_str=cfg.QUANTIZER_A)   
        self.qact_new_p = QAct(quant=quant,
                               calibrate=calibrate,
                               bit_type=cfg.BIT_TYPE_A,
                               calibration_mode=cfg.CALIBRATION_MODE_A,
                               observer_str=cfg.OBSERVER_A,
                               quantizer_str=cfg.QUANTIZER_A)   

    def forward(self, x, state_A, state_B, state_p, state_x):
        time_mix_k = self.qact_time_mix_k(self.time_mix_k)
        time_mix_v = self.qact_time_mix_v(self.time_mix_v)
        time_mix_r = self.qact_time_mix_r(self.time_mix_r)
        
        xk = x * time_mix_k + state_x * (1 - time_mix_k)
        xk = self.qact_xk(xk)
        xv = x * time_mix_v + state_x * (1 - time_mix_v)
        xv = self.qact_xv(xv)
        xr = x * time_mix_r + state_x * (1 - time_mix_r)
        xr = self.qact_xr(xr)
        new_x = x
        
        k = self.key(xk)
        k = self.qact_key(k)

        v = self.value(xv)
        v = self.qact_val(v)

        r = self.receptance(xr)
        r = self.qact_recept(r)

        r = self.hard_sigmoid(r)
        r = self.qact_sigmoid(r)

        time_first = self.qact_time_first(self.time_first)
        ww = time_first + k
        p = torch.maximum(state_p, ww)
        p = self.qact_p1(p)

        e1 = torch.exp(state_p - p)
        e2 = torch.exp(ww - p)
        a = e1 * state_A + e2 * v
        b = e1 * state_B + e2

        time_decay = self.qact_time_decay(-torch.exp(self.time_decay))
        ww = state_p + time_decay
        p = torch.maximum(ww, k)
        p = self.qact_p1(p)

        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)
        new_A = e1 * state_A + e2 * v
        new_B = e1 * state_B + e2
        new_p = p
        rwkv = r * a / b
        new_A = self.qact_new_A(new_A)
        new_B = self.qact_new_B(new_B)
        new_p = self.qact_new_p(new_p)
        rwkv = self.qact_rwkv(rwkv)

        rwkv = self.output(rwkv)
        rwkv = self.qact_out(rwkv)
        return rwkv, new_A, new_B, new_p, new_x


class Block_For_Quant(nn.Module):
    def __init__(self, 
                 layer_id, 
                 n_embed, 
                 ffn_dim,
                 quant = False,
                 calibrate = False
                ):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = QIntLayerNorm(n_embed)
        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        self.ln2 = QIntLayerNorm(n_embed)
        self.qact2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        if self.layer_id == 0:
            self.ln0 = QIntLayerNorm(n_embed)
            self.qact0 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        
        self.att = RWKV_TimeMix_For_Quant(layer_id, n_embed, quant=quant, calibrate=calibrate)
        self.qact_att = QAct(quant=quant,
                             calibrate=calibrate,
                             bit_type=cfg.BIT_TYPE_A,
                             calibration_mode=cfg.CALIBRATION_MODE_A,
                             observer_str=cfg.OBSERVER_A,
                             quantizer_str=cfg.QUANTIZER_A)
        self.qact_new_A = QAct(quant=quant,
                               calibrate=calibrate,
                               bit_type=cfg.BIT_TYPE_A,
                               calibration_mode=cfg.CALIBRATION_MODE_A,
                               observer_str=cfg.OBSERVER_A,
                               quantizer_str=cfg.QUANTIZER_A)
        self.qact_new_B = QAct(quant=quant,
                               calibrate=calibrate,
                               bit_type=cfg.BIT_TYPE_A,
                               calibration_mode=cfg.CALIBRATION_MODE_A,
                               observer_str=cfg.OBSERVER_A,
                               quantizer_str=cfg.QUANTIZER_A)
        self.qact_new_p = QAct(quant=quant,
                               calibrate=calibrate,
                               bit_type=cfg.BIT_TYPE_A,
                               calibration_mode=cfg.CALIBRATION_MODE_A,
                               observer_str=cfg.OBSERVER_A,
                               quantizer_str=cfg.QUANTIZER_A)
        self.qact_new_x = QAct(quant=quant,
                               calibrate=calibrate,
                               bit_type=cfg.BIT_TYPE_A,
                               calibration_mode=cfg.CALIBRATION_MODE_A,
                               observer_str=cfg.OBSERVER_A,
                               quantizer_str=cfg.QUANTIZER_A)
        self.ffn = RWKV_ChannelMix_For_Quant(layer_id, n_embed, ffn_dim, quant=quant, calibrate=calibrate)
        self.qact_ffn = QAct(quant=quant,
                             calibrate=calibrate,
                             bit_type=cfg.BIT_TYPE_A,
                             calibration_mode=cfg.CALIBRATION_MODE_A,
                             observer_str=cfg.OBSERVER_A,
                             quantizer_str=cfg.QUANTIZER_A)
        self.qact_new_ffn = QAct(quant=quant,
                                 calibrate=calibrate,
                                 bit_type=cfg.BIT_TYPE_A,
                                 calibration_mode=cfg.CALIBRATION_MODE_A,
                                 observer_str=cfg.OBSERVER_A,
                                 quantizer_str=cfg.QUANTIZER_A)

    def forward(self, x, state_A, state_B, state_p, state_x, state_ffn, last_quantizer=None):
        if self.layer_id == 0:
            x = self.ln0(x, last_quantizer, self.qact0.quantizer)
            x = self.qact0(x)
            last_quantizer = self.qact0.quantizer
        
        
        short_cut = x
        x = self.qact1(self.ln1(x, last_quantizer, self.qact1.quantizer))
        x, new_A, new_B, new_p, new_x = self.att(x, state_A, state_B, state_p, state_x)

        new_A = self.qact_new_A(new_A)
        new_B = self.qact_new_B(new_B)
        new_p = self.qact_new_p(new_p)
        new_x = self.qact_new_x(new_x)
        x = self.qact_att(short_cut + x)

        short_cut = x
        x = self.qact2(self.ln2(x, self.qact1.quantizer, self.qact2.quantizer))
        x, new_ffn = self.ffn(x, state_ffn)
        new_ffn = self.qact_new_ffn(new_ffn)
        x = self.qact_ffn(short_cut + x)
        return x, new_A, new_B, new_p, new_x, new_ffn


class RWKV_V4_Quant_Infer(nn.Module):
    def __init__(self,
                 vocab_size=2000,
                 hidden_size=512,
                 num_hidden_layers=4,
                 intermediate_size=1024,
                 input_hidden_quant = True,
                 quant = False,
                 calibrate = False,
                 ):
        super(RWKV_V4_Quant_Infer, self).__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.input_hidden_quant = input_hidden_quant

        if input_hidden_quant:
            self.qact_state_A = QAct(quant=quant,
                                   calibrate=calibrate,
                                   bit_type=cfg.BIT_TYPE_A,
                                   calibration_mode=cfg.CALIBRATION_MODE_A,
                                   observer_str=cfg.OBSERVER_A,
                                   quantizer_str=cfg.QUANTIZER_A)
            self.qact_state_B = QAct(quant=quant,
                                   calibrate=calibrate,
                                   bit_type=cfg.BIT_TYPE_A,
                                   calibration_mode=cfg.CALIBRATION_MODE_A,
                                   observer_str=cfg.OBSERVER_A,
                                   quantizer_str=cfg.QUANTIZER_A)
            self.qact_state_p = QAct(quant=quant,
                                   calibrate=calibrate,
                                   bit_type=cfg.BIT_TYPE_A,
                                   calibration_mode=cfg.CALIBRATION_MODE_A,
                                   observer_str=cfg.OBSERVER_A,
                                   quantizer_str=cfg.QUANTIZER_A)
            self.qact_state_x = QAct(quant=quant,
                                   calibrate=calibrate,
                                   bit_type=cfg.BIT_TYPE_A,
                                   calibration_mode=cfg.CALIBRATION_MODE_A,
                                   observer_str=cfg.OBSERVER_A,
                                   quantizer_str=cfg.QUANTIZER_A)
            self.qact_state_ffn = QAct(quant=quant,
                                   calibrate=calibrate,
                                   bit_type=cfg.BIT_TYPE_A,
                                   calibration_mode=cfg.CALIBRATION_MODE_A,
                                   observer_str=cfg.OBSERVER_A,
                                   quantizer_str=cfg.QUANTIZER_A)

        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.qact_embed = QAct(quant=quant,
                               calibrate=calibrate,
                               bit_type=cfg.BIT_TYPE_A,
                               calibration_mode=cfg.CALIBRATION_MODE_A,
                               observer_str=cfg.OBSERVER_A,
                               quantizer_str=cfg.QUANTIZER_A)
        self.blocks = nn.ModuleList([Block_For_Quant(i, hidden_size, intermediate_size, quant=quant, calibrate=calibrate) for i in range(num_hidden_layers)])
        self.ln_out = QIntLayerNorm(hidden_size)
        self.qact_ln = QAct(quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_A,
                            calibration_mode=cfg.CALIBRATION_MODE_A,
                            observer_str=cfg.OBSERVER_A,
                            quantizer_str=cfg.QUANTIZER_A)
        self.head = QLinear(hidden_size, vocab_size, bias=False)
        self.qact_out = QAct(quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_A,
                            calibration_mode=cfg.CALIBRATION_MODE_A,
                            observer_str=cfg.OBSERVER_A,
                            quantizer_str=cfg.QUANTIZER_A)

    def model_quant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.quant = True
            if cfg.INT_NORM:
                if type(m) in [QIntLayerNorm]:
                    m.mode = 'int'

    def model_dequant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.quant = False

    def model_open_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.calibrate = True

    def model_open_last_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.last_calibrate = True

    def model_close_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.calibrate = False

    def forward_initialzation(self, batch_size, device):
        state_A = torch.zeros([self.num_hidden_layers, batch_size, self.hidden_size])
        state_B = torch.zeros([self.num_hidden_layers, batch_size, self.hidden_size])
        state_p = torch.zeros([self.num_hidden_layers, batch_size, self.hidden_size]) - 1e30
        state_x = torch.zeros([self.num_hidden_layers, batch_size, self.hidden_size])
        state_ffn = torch.zeros([self.num_hidden_layers, batch_size, self.hidden_size])
        hidden_state = torch.stack([state_A, state_B, state_p, state_x, state_ffn]).to(device)
        return hidden_state
        
    def forward(self, input_token, hidden_state):
        batch_size = input_token.size(0)
        state_A, state_B, state_p, state_x, state_ffn = hidden_state.split(1, dim=0)
        if self.input_hidden_quant:
            state_A = self.qact_state_A(state_A)
            state_B = self.qact_state_B(state_B)
            state_p = self.qact_state_p(state_p)
            state_x = self.qact_state_x(state_x)
            state_ffn = self.qact_state_ffn(state_ffn)

        x = self.emb(input_token)
        x = self.qact_embed(x)
        
        new_hidden_state = []
        for i, block in enumerate(self.blocks):
            last_quantizer = self.qact_embed.quantizer if i == 0 else self.blocks[
                i - 1].qact_ffn.quantizer
            x, new_A, new_B, new_p, new_x, new_ffn = \
                block(x, state_A[0, i], state_B[0, i], state_p[0, i], state_x[0, i], state_ffn[0, i], last_quantizer)

            new_hidden_state.append(new_A)
            new_hidden_state.append(new_B)
            new_hidden_state.append(new_p)
            new_hidden_state.append(new_x)
            new_hidden_state.append(new_ffn)

        new_hidden_state = torch.cat(new_hidden_state)
        new_hidden_state = new_hidden_state.view([self.num_hidden_layers, 5, batch_size, self.hidden_size])
        new_hidden_state = new_hidden_state.transpose(0, 1)
        x = self.ln_out(x, self.blocks[-1].qact_ffn.quantizer, self.qact_ln.quantizer)
        x = self.qact_ln(x)

        x = self.head(x)
        x = self.qact_out(x)
        return x, new_hidden_state


@MODULE_BUILD_FUNCS.registe_with_name(module_name='rwkv_v4_quant_infer')
def build_rwkv_v4_quant_infer(args):
    model = RWKV_V4_Quant_Infer(
        vocab_size = args.vocab_size,
        hidden_size = args.hidden_size,
        num_hidden_layers = args.num_hidden_layer,
        intermediate_size = args.intermediate_size
    )
    criterion = nn.CrossEntropyLoss(reduction='none')
    return model, criterion
