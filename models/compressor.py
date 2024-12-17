import torch
import torch.nn as nn


class XLCompressorForCoreML(nn.Module):
    def __init__(self, model, num_tokens):
        super(XLCompressorForCoreML, self).__init__()
        self.next_token_predictor = model
        self.num_tokens = num_tokens

    def forward(self, input_onehots, memories):
        outputs = []
        for token_id in range(self.num_tokens):
            input_onehot = input_onehots[:, token_id:token_id+1, :]
            output, memories = self.next_token_predictor(input_onehot, memories)
            outputs.append(output)

        if self.num_tokens > 1:
            outputs = torch.cat(outputs, dim=1)
        else:
            outputs = outputs[0]
        return outputs, memories


class XLCompressorCache(nn.Module):
    def __init__(self, model, num_tokens):
        super(XLCompressorCache, self).__init__()
        self.next_token_predictor = model
        self.num_tokens = num_tokens

    def forward(self, input_onehots, k_cache, v_cache):
        outputs = []
        for token_id in range(self.num_tokens):
            input_onehot = input_onehots[:, token_id:token_id + 1, :]
            output, k_cache, v_cache = self.next_token_predictor(input_onehot, k_cache, v_cache)
            outputs.append(output)

        if self.num_tokens > 1:
            outputs = torch.cat(outputs, dim=1)
        else:
            outputs = outputs[0]
        return outputs, k_cache, v_cache


class XLCompressorForXNN(nn.Module):
    def __init__(self, model, num_tokens):
        super(XLCompressorForXNN, self).__init__()
        self.next_token_predictor = model
        self.num_tokens = num_tokens

    def forward(self, input_onehots, memories):
        outputs = []
        for token_id in range(self.num_tokens):
            input_onehot = input_onehots[:, token_id:token_id + 1, :]
            output, memories = self.next_token_predictor(input_onehot, memories)
            outputs.append(output)

        if self.num_tokens > 1:
            outputs = torch.cat(outputs, dim=1)
        else:
            outputs = outputs[0]
        return outputs, memories


class RWKVCompressorForCoreML(nn.Module):
    def __init__(self, model, num_tokens):
        super(RWKVCompressorForCoreML, self).__init__()
        self.next_token_predictor = model
        self.num_tokens = num_tokens

    def forward(self, input_onehots, hidden_state):
        outputs = []
        for token_id in range(self.num_tokens):
            input_onehot = input_onehots[token_id, :, :]
            output, hidden_state = self.next_token_predictor(input_onehot, hidden_state)
            outputs.append(output)

        if self.num_tokens > 1:
            outputs = torch.stack(outputs, dim=0)
        else:
            outputs = outputs[0].unsqueeze(dim=0)
        return outputs, hidden_state

