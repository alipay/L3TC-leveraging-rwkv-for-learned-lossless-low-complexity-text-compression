import math
import torch
import random
import pkuseg
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset


class EnWikASCIITrainDataSet(Dataset):
    def __init__(self, args, corpus_path, word2id_dict):
        self.corpus_path = corpus_path
        self.descriptions = []
        self.segment_length = args.sentence_length
        self.word2id_dict = word2id_dict
        self.chunk_size = args.chunk_size
        self.epoch_length = args.epoch_length

        if args.debug:
            self.epoch_length = args.batch_size * 2

        self.start_token = word2id_dict['<s>']
        self.unknown_token = word2id_dict['<unk>']
        self.padding_token = word2id_dict['<pad>']

        with open(corpus_path, 'rb') as f:
            train_data = f.read()

        self.tokens = [int(x) + 3 for x in train_data]

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, item):
        output = {}
        
        input_token = []
        output_token = []
        input_types = []
        output_types = []

        start_token_id = np.random.randint(low=0, high=len(self.tokens) - self.chunk_size * self.segment_length)

        segments_count = 0
        while True:
            if start_token_id + segments_count * self.segment_length < len(self.tokens) and segments_count < self.chunk_size:
                cur_input_type = []
                cur_desc_type = []
                if segments_count == 0:
                    cur_input_segment = [self.start_token] + self.tokens[start_token_id + segments_count * self.segment_length: start_token_id + (segments_count+1) * self.segment_length][:-1]
                    cur_desc_segment = self.tokens[start_token_id + segments_count * self.segment_length: start_token_id + (segments_count+1) * self.segment_length]
                else:
                    cur_input_segment = self.tokens[start_token_id + segments_count * self.segment_length - 1: start_token_id + (segments_count+1) * self.segment_length - 1]
                    cur_desc_segment = self.tokens[start_token_id + segments_count * self.segment_length: start_token_id + (segments_count+1) * self.segment_length]
                    
                # 补全padding
                if len(cur_input_segment) < self.segment_length:
                    for i in range(0, self.segment_length - len(cur_input_segment)):
                        cur_input_segment.append(self.padding_token)

                if len(cur_desc_segment) < self.segment_length:
                    for i in range(0, self.segment_length - len(cur_desc_segment)):
                        cur_desc_segment.append(self.padding_token)

                input_token.append(cur_input_segment)
                output_token.append(cur_desc_segment)
                
                # 生成对应的type
                for i in cur_input_segment:
                    if i == self.padding_token:
                        cur_input_type.append(0)
                    else:
                        cur_input_type.append(1)
                
                # 生成对应的type
                for i in cur_desc_segment:
                    if i == self.padding_token:
                        cur_desc_type.append(0)
                    else:
                        cur_desc_type.append(1)

                input_types.append(cur_input_type)
                output_types.append(cur_desc_type)

                segments_count += 1
            else:
                break
        
        if len(input_token) < self.chunk_size:
            for _ in range(self.chunk_size - len(input_token)):
                input_token.append([self.padding_token] * self.segment_length)
                output_token.append([self.padding_token] * self.segment_length)
                input_types.append([0] * self.segment_length)
                output_types.append([0] * self.segment_length)

            output['input_token'] = input_token
            output['output_token'] = output_token
            output['input_types'] = input_types
            output['output_types'] = output_types
        else:
            output['input_token'] = input_token
            output['output_token'] = output_token
            output['input_types'] = input_types
            output['output_types'] = output_types
        
        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return instance


class EnWikASCIITestDataSet(Dataset):
    def __init__(self, args, corpus_path, word2id_dict):
        self.corpus_path = corpus_path
        self.descriptions = []
        self.segment_length = args.sentence_length
        self.model_name = args.model_name

        self.start_token = word2id_dict['<s>']
        self.unknown_token = word2id_dict['<unk>']
        self.padding_token = word2id_dict['<pad>']

        with open(corpus_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        test_lines = "".join(lines[1128023:1128023+5000])
        test_data = test_lines.encode()
        tokens = [int(x) + 3 for x in test_data]

        # 将tokens按照一定长度分割成多份，加速验证
        self.seg_tokens = []
        seg_data_len = 10000
        num_segs = math.ceil(len(tokens) / seg_data_len)
        for i in range(num_segs):
            self.seg_tokens.append(tokens[i * seg_data_len: (i+1) * seg_data_len])
        
        if 'rwkv' in self.model_name:
            self.num_segments = math.ceil(seg_data_len / self.segment_length)

        if args.debug:
            self.seg_tokens = self.seg_tokens[:5]

    def __len__(self):
        return len(self.seg_tokens)

    def __getitem__(self, item):
        output = {}

        descriptions_ids = self.seg_tokens[item]

        input_token = []
        output_token = []
        input_types = []
        output_types = []

        segments_count = 0
        while True:
            if segments_count * self.segment_length < len(descriptions_ids):
                cur_input_type = []
                cur_desc_type = []
                if segments_count == 0:
                    cur_input_segment = [self.start_token] + descriptions_ids[segments_count*self.segment_length: min((segments_count+1) * self.segment_length, len(descriptions_ids))][:-1]
                    cur_desc_segment = descriptions_ids[segments_count*self.segment_length: min((segments_count+1) * self.segment_length, len(descriptions_ids))]
                else:
                    cur_input_segment = descriptions_ids[segments_count*self.segment_length-1: min((segments_count+1) * self.segment_length, len(descriptions_ids))-1]
                    cur_desc_segment = descriptions_ids[segments_count*self.segment_length: min((segments_count+1) * self.segment_length, len(descriptions_ids))]

                # 补全padding
                if len(cur_input_segment) < self.segment_length:
                    for i in range(0, self.segment_length - len(cur_input_segment)):
                        cur_input_segment.append(self.padding_token)

                if len(cur_desc_segment) < self.segment_length:
                    for i in range(0, self.segment_length - len(cur_desc_segment)):
                        cur_desc_segment.append(self.padding_token)

                input_token.append(cur_input_segment)
                output_token.append(cur_desc_segment)

                # 生成对应的type
                for i in cur_input_segment:
                    if i == self.padding_token:
                        cur_input_type.append(0)
                    else:
                        cur_input_type.append(1)
                
                # 生成对应的type
                for i in cur_desc_segment:
                    if i == self.padding_token:
                        cur_desc_type.append(0)
                    else:
                        cur_desc_type.append(1)

                input_types.append(cur_input_type)
                output_types.append(cur_desc_type)

                segments_count += 1
            else:
                break
        
        if 'rwkv' in self.model_name:
            if segments_count < self.num_segments:
                padding_vals = [[0] * self.segment_length] * (self.num_segments - segments_count)
                input_token.extend(padding_vals)
                output_token.extend(padding_vals)
                input_types.extend(padding_vals)
                output_types.extend(padding_vals)
            else:
                input_token = input_token[:self.num_segments]
                output_token = output_token[:self.num_segments]
                input_types = input_types[:self.num_segments]
                output_types = output_types[:self.num_segments]

        output['input_token'] = input_token
        output['output_token'] = output_token
        output['input_types'] = input_types
        output['output_types'] = output_types
        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        
        return instance


if __name__ == '__main__':
    # dataloader = TransformerXLDataSet(CorpusPath)
    dataloader = TransformerXLTestSet(EvalPath)
    for data in dataloader:
        x = 1
        break
    print('加载完成')
