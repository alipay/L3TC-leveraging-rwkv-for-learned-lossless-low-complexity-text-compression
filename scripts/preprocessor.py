# coding=utf8
import random
import json
import os
import glob
import sentencepiece as spm
from collections import Counter, OrderedDict

from tqdm import tqdm
    

class PretrainProcess(object):
    def __init__(self):        
        self.TrainFile = "data/public_text_dataset/enwik8"
        self.ValFile = "data/public_text_dataset/enwik9"  # 只取中间5000lines For 快速评测验证
        self.vocab_size = 16384  # 4096 for enwik8 and 16384 for enwik9 in nncp
        self.model_type = 'bpe'  # 可选值: unigram (默认), bpe, char 或 word, 使用word类型时，必须对输入句子进行pretokenized。
        self.coverage = 0.999

    def get_file_infos(self):
        with open(self.TrainFile, "r", encoding="utf-8") as f:
            lines = f.readlines()

        max_length = max([len(line) for line in lines])
        print(f"There are {len(lines)} lines in enwik8, the length of the longest sentence is {max_length}")

    def do_spm_training(self):
        os.makedirs(f"dictionary/vocab_enwik8_{self.model_type}_{self.vocab_size}_{self.coverage}", exist_ok=True)
        # train_param = f'--input=./temp_process_spm_train_data.txt --pad_id=0 --unk_id=1 \
        #         --bos_id=2 --eos_id=-1 \
        #         --model_prefix=../dictionary/vocab_enwik8_{self.model_type}_{self.vocab_size}_{self.coverage}/spm_enwik8_{self.model_type}_{self.vocab_size}_{self.coverage} \
        #         --vocab_size={self.vocab_size} \
        #         --character_coverage={self.coverage} \
        #         --max_sentence_length=10000 \
        #         --add_dummy_prefix=0 \
        #         --remove_extra_whitespaces=0 \
        #         --user_defined_symbols=\n,\t \
        #         --model_type={self.model_type}'
        # spm.SentencePieceTrainer.Train(train_param)

        cmd = f'''bin/spm_train \
                --input={self.TrainFile} --pad_id=0 --unk_id=1 \
                --bos_id=2 --eos_id=-1 \
                --model_prefix=dictionary/vocab_enwik8_{self.model_type}_{self.vocab_size}_{self.coverage}/spm_enwik8_{self.model_type}_{self.vocab_size}_{self.coverage} \
                --vocab_size={self.vocab_size} \
                --character_coverage={self.coverage} \
                --max_sentence_length=10000 \
                --add_dummy_prefix=0 \
                --remove_extra_whitespaces=0 \
                --user_defined_symbols="\n,\t" \
                --model_type={self.model_type}'''
        print(cmd)
        os.system(cmd)

        # rm_cmd = "rm -f ./temp_process_spm_train_data.txt"
        # os.system(rm_cmd)

    def generate_dictionary(self):
        spm_vocab_file = f"dictionary/vocab_enwik8_{self.model_type}_{self.vocab_size}_{self.coverage}/spm_enwik8_{self.model_type}_{self.vocab_size}_{self.coverage}.vocab"
        self.symb2id_dict = OrderedDict()
        with open(spm_vocab_file, 'r', encoding="utf-8") as f:
            for line in f:
                if line == "\t\t0\n":
                    self.symb2id_dict["\t"] = len(self.symb2id_dict)
                elif line == "\n":
                    self.symb2id_dict["\n"] = len(self.symb2id_dict)
                elif line == "\t0\n":
                    continue
                else:
                    symb = line.strip().split()[0]
                    self.symb2id_dict[symb] = len(self.symb2id_dict)
        
        with open(f"dictionary/vocab_enwik8_{self.model_type}_{self.vocab_size}_{self.coverage}/spm_enwik8_{self.model_type}_{self.vocab_size}_{self.coverage}_vocab.json", "w") as obj_f:
            json.dump(self.symb2id_dict, obj_f, indent=4, ensure_ascii=False)

    def convert_rawtext_into_labeltext(self):
        sp_processor = spm.SentencePieceProcessor()
        sp_processor.Load(f"dictionary/vocab_enwik8_{self.model_type}_{self.vocab_size}_{self.coverage}/spm_enwik8_{self.model_type}_{self.vocab_size}_{self.coverage}.model")

        os.makedirs("data/train_data", exist_ok=True)
        os.makedirs("data/test_data", exist_ok=True)

        # 生成训练数据
        if True:
            train_data_filename = f"data/train_data/train_enwik8_{self.model_type}_{self.vocab_size}_{self.coverage}.txt"

            with open(train_data_filename, "w") as output_file:
                pass
            
            # 将enwik8看成是一行数据
            with open(self.TrainFile, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            train_tokens = []
            for line in tqdm(lines):
                proto = sp_processor.encode(line, out_type='immutable_proto')
                tokens = []
                for n in proto.pieces:
                    if n.begin == n.end:
                        continue
                    tokens.append(str(n.id))
                
                train_tokens.extend(tokens)

            train_tokens = ",".join(train_tokens)
            with open(train_data_filename, "a") as output_file:
                output_file.write(train_tokens + "\n")

        # 生成测试数据
        if True:
            test_data_filename = f"data/test_data/test_enwik9_{self.model_type}_{self.vocab_size}_{self.coverage}.txt"
            with open(test_data_filename, "w") as output_file:
                pass
            
            # 将enwik9看成是一行数据
            with open(self.ValFile, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            test_lines = lines[1128023:1128023+5000]
            test_tokens = []
            for line in tqdm(test_lines):
                proto = sp_processor.encode(line, out_type='immutable_proto')
                tokens = []
                for n in proto.pieces:
                    if n.begin == n.end:
                        continue
                    tokens.append(str(n.id))
                
                test_tokens.extend(tokens)

            test_tokens = ",".join(test_tokens)
            with open(test_data_filename, "a") as output_file:
                output_file.write(test_tokens + "\n")


if __name__ == '__main__':
    pp = PretrainProcess()

    pp.get_file_infos()
    
    pp.do_spm_training()

    pp.generate_dictionary()
   
    pp.convert_rawtext_into_labeltext()
