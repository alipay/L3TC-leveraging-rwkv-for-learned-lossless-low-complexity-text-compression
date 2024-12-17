# coding: utf-8
import os
import glob
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import time
import json
import math
import argparse
import shutil
import pandas as pd
import sentencepiece as spm

import sys
sys.path.append('..')
from tqdm import tqdm
from util import misc
from util.slconfig import DictAction, SLConfig
# from util.nncp_arithmetic_coder import ArithmeticEncoder, ArithmeticDecoder, BitInputStream, BitOutputStream
from util.arithmeticcoding import *


def get_args_parser():
    parser = argparse.ArgumentParser('Pytorch Compressor', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # compression common setting
    parser.add_argument('--input_file', type=str, default='../data/public_text_dataset/enwik9', 
                        help='input file to be compressed')
    parser.add_argument('--output_file', type=str, default='../data/compressed_data/enwik9.bin', 
                        help='output compressed file')
    parser.add_argument('--pretrain_model_path', 
                        help='load from other checkpoint')
    parser.add_argument('--tokenizer', type=str, 
                        help='choose tokenizer')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--tmp_processed_dir', type=str, default="../data/tmp", 
                        help='path to processed data')
    parser.add_argument('--result_dir', type=str, default="../results", 
                        help='path to processed data')
    parser.add_argument('--azstd_bin_file', type=str, default="/home/usr/junxuan/codec/ResourceAnalyzer/src/bin_files/zstd", 
                        help='path to processed data')

    # compression specific setting
    parser.add_argument('--load_preprocess_cache', action='store_true')
    parser.add_argument('--segment_length', type=int, default=0, help="0: infinite(不推荐), >1: segment (unit: byte)")

    return parser


def get_word2id_dict(dict_path):
    with open(dict_path, "r") as obj_f:
        word2id_dict = json.load(obj_f)
    return word2id_dict


def get_ascii_word2id_dict():
    word2id_dict = {}
    word2id_dict['<s>'] = len(word2id_dict)
    word2id_dict['<unk>'] = len(word2id_dict)
    word2id_dict['<pad>'] = len(word2id_dict)
    
    byte_order = 'big'
    for i in range(256):
        byte_val = i.to_bytes(1, byte_order)
        word2id_dict[byte_val] = len(word2id_dict)
    return word2id_dict


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.model_name in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.model_name)
    model, criterion = build_func(args)
    return model, criterion


def check_keep(keyname, ignorekeywordlist):
    for keyword in ignorekeywordlist:
        if keyword in keyname:
            ignorelist.append(keyname)
            return False
    return True


def compression_preprocess_with_LLM(args, input_file, tokenizer):
    if args.segment_length > 0:
        segment_lines = []
        cache_line = ""
        with open(input_file, "rb") as f:
            test_data = f.read()

        num_segs = int(math.ceil(len(test_data) / args.segment_length))
        segments_tokens = []
        unk_words = []
        for i in tqdm(range(num_segs)):
            seg_data = test_data[i * args.segment_length: (i+1) * args.segment_length]
            if "rwkv" in args.model_name and "pile" in args.config_file:
                seg_data = "\n" + seg_data.decode('latin')
            else:
                seg_data = seg_data.decode('latin')
            seg_tokens = tokenizer.encode(seg_data)
            segments_tokens.append(seg_tokens)
    else:
        raise ValueError("Unsupport Now !")
    
    return segments_tokens, unk_words


def compression_preprocess_ascii(args, input_file, word2id_dict):
    """
        文件压缩前处理
    """
    start_token = word2id_dict['<s>']
    if args.segment_length > 0:
        segment_lines = []
        cache_line = ""
        with open(input_file, "rb") as f:
            test_data = f.read()

        test_tokens = [int(x) + 3 for x in test_data]
        num_segs = int(math.ceil(len(test_tokens) / args.segment_length))
        segments_tokens = []
        unk_words = []
        for i in range(num_segs):
            seg_tokens = [2] + test_tokens[i * args.segment_length: (i+1) * args.segment_length]
            segments_tokens.append(seg_tokens)
    else:
        raise ValueError("Unsupport Now !")

    return segments_tokens, unk_words


def compression_preprocess(args, input_file, word2id_dict):
    """
        文件压缩前处理
    """
    start_token = word2id_dict['<s>']
    sentence_piece_model = args.tokenizer
    sp_processor = spm.SentencePieceProcessor()
    sp_processor.Load(sentence_piece_model)

    if args.segment_length > 0:
        segment_lines = []
        cache_line = ""
        with open(input_file, "r", encoding='UTF-8') as f:
            lines = f.readlines()
        
        for line in tqdm(lines):
            line = cache_line + line
            cache_line = ""
            
            if len(line) > args.segment_length:
                if len(line) % args.segment_length == 0:
                    num_segs = len(line) // args.segment_length
                else:
                    num_segs = len(line) // args.segment_length + 1
                
                for i in range(num_segs - 1):
                    segment_line = line[i * args.segment_length: (i+1) * args.segment_length]
                    segment_lines.append(segment_line)
                
                i = num_segs - 1
                segment_line = line[i * args.segment_length: (i+1) * args.segment_length]
                if len(segment_line) < args.segment_length:
                    cache_line = segment_line
                else:
                    segment_lines.append(segment_line)

            else:
                if len(line) < args.segment_length:
                    cache_line = line
                else:
                    segment_lines.append(segment_line)
            
        if len(cache_line) > 0:
            segment_lines.append(cache_line)

        segments_tokens = []
        unk_words = []

        for line in tqdm(segment_lines):
            # line = line.replace('\n', "<next-line>")
            proto = sp_processor.encode(line, out_type='immutable_proto')
            tokens = [2] # 2 is start_token
            for n in proto.pieces:
                if n.begin == n.end:
                    continue
                if n.id == 1:
                    unk_words.append(n.surface)
                tokens.append(n.id)

            segments_tokens.append(tokens)
    else:
        raise ValueError("Unsupport Now !")
    
    return segments_tokens, unk_words


def compress_tokens_with_ArithmeticCoder(args, model, segments_tokens):
    """
        模型前向 + 算术编码
    """
    CHECK_CROSS_ENTROPY = True
    num_segments = len(segments_tokens)
    batch_size = 1024
    if "transformer_xl" in args.model_name and "169" in args.config_file:
        batch_size = 512
    num_batches = num_segments // batch_size if num_segments % batch_size == 0 else num_segments // batch_size + 1
    
    entropy_sum = 0.0
    token_count = 0
    total_bin_size = 0
    for i in range(num_batches):
        batch_seg_tokens = segments_tokens[i * batch_size: (i+1) * batch_size]
        # 对batch_seg_tokens进行padding
        max_length = 0
        for seg_tokens in batch_seg_tokens:
            if len(seg_tokens) > max_length:
                max_length = len(seg_tokens)

        # 对batch维度进行padding
        if len(batch_seg_tokens) < batch_size:
            for _ in range(batch_size - len(batch_seg_tokens)):
                batch_seg_tokens.append([0] * max_length)
        
        # 对token维度padding
        # sub_file_lens = []
        # sub_file_arith_encs = []
        for idx, seg_tokens in enumerate(batch_seg_tokens):
            # sub_file_lens.append(len(seg_tokens))
            # original_file_len = len(seg_tokens) - 1  # since tokens including start token, length = numtokens - 1

            # out_name = os.path.join(args.tmp_processed_dir, "enwik_%09d.bin" %(i * batch_size + idx))
            # out_file = open(out_name, "wb")
            # # build the output stream
            # bit_output = BitOutputStream(out_file)
            # # build arithmetic encoder
            # arith_enc = ArithmeticEncoder(bit_output)
            # # 将原始文件尺寸写入bin
            # out_file.write(original_file_len.to_bytes(2, byteorder='big'))
            # sub_file_arith_encs.append(arith_enc)

            for _ in range(max_length - len(seg_tokens)):
                seg_tokens.append(0)
        
        batch_seg_tokens_tensor = torch.tensor(batch_seg_tokens).cuda()
        pbar = tqdm(total=max_length-1)
        for token_id in range(max_length-1):
            with torch.no_grad():
                if token_id == 0:
                    hidden_state = model.forward_initialzation(batch_size, args.device)

                start_time = time.time()
                input_tokens_tensor = batch_seg_tokens_tensor[:, token_id]
                output_token_tensor = batch_seg_tokens_tensor[:, token_id+1]
                if "transformer_xl" in args.model_name:
                    input_tokens_tensor = input_tokens_tensor[:, None]
                logits, hidden_state = model(input_tokens_tensor, hidden_state)
                if "transformer_xl" in args.model_name:
                    logits = logits.squeeze(dim=1)
                model_time = time.time() - start_time

                # get freq table and do ac
                start_time = time.time()
                probs = torch.softmax(logits, dim=-1)               
                freqs = torch.round(probs * 10000000).int()
                freqs = torch.max(freqs, freqs.new_ones(freqs.size()))
                new_probs = freqs / freqs.sum(dim=1)[:, None]
                # freqs_list = freqs.cpu().numpy().tolist()
                probs_time = time.time() - start_time
                start_time = time.time()

                # for idx, freq in enumerate(freqs_list):
                #     if output_token_tensor[idx] != 0:
                #         freq = SimpleFrequencyTable(freq)
                #         sub_file_arith_encs[idx].write(freq, batch_seg_tokens[idx][token_id+1])

                if CHECK_CROSS_ENTROPY:
                    indices = [list(range(batch_size)), output_token_tensor.cpu().numpy().tolist()]
                    entropy = torch.log(new_probs[indices]) / -math.log(2)
                    entropy_sum += (entropy * (output_token_tensor != 0)).sum().item()
                    
                    # cross_entropy = criterion(logits.view(batch_size, -1), output_token_tensor.view(batch_size))
                    token_count += (output_token_tensor != 0).sum().item()

                    entropy_time = time.time() - start_time
                    pbar.set_description("Total Entropy: {:.4f}, Ave Entropy: {:.4f}, model_time: {:.6f}, probs_time: {:.6f}, entropy_time: {:.6f}".format(entropy_sum, entropy_sum / token_count, model_time, probs_time, entropy_time))

            pbar.update()
        
        pbar.close()

        # for arith_enc in sub_file_arith_encs:
        #     arith_enc.finish()
        #     arith_enc.output.close()

        print("[Compression Infos] Finish {} / {} batches.".format(i+1, num_batches))

        # # 计算目前bin的大小
        # for idx in range(batch_size):
        #     out_name = os.path.join(args.tmp_processed_dir, "enwik_%09d.bin" %(i * batch_size + idx))
        #     total_bin_size += os.path.getsize(out_name)
        
        #     rm_cmd = "rm -f {}".format(out_name)
        #     os.system(rm_cmd)

    total_bin_size_min = math.ceil(entropy_sum / 8)

    return total_bin_size_min


def compress_tokens_with_ArithmeticCoder_with_Transformer(args, model, segments_tokens):
    """
        模型前向 + 算术编码
    """
    CHECK_CROSS_ENTROPY = True
    num_segments = len(segments_tokens)
    if "llama" in args.model_name:
        batch_size = 4
    else:
        batch_size = 16
    num_batches = num_segments // batch_size if num_segments % batch_size == 0 else num_segments // batch_size + 1
    
    entropy_sum = 0.0
    token_count = 0
    total_bin_size = 0

    pbar = tqdm(total=num_batches)
    for i in range(num_batches):
        batch_seg_tokens = segments_tokens[i * batch_size: (i+1) * batch_size]
        # 对batch_seg_tokens进行padding
        max_length = 0
        for seg_tokens in batch_seg_tokens:
            if len(seg_tokens) > max_length:
                max_length = len(seg_tokens)

        # 对batch维度进行padding
        if len(batch_seg_tokens) < batch_size:
            for _ in range(batch_size - len(batch_seg_tokens)):
                batch_seg_tokens.append([0] * max_length)
            
        # 对token维度padding
        for idx, seg_tokens in enumerate(batch_seg_tokens):
            for _ in range(max_length - len(seg_tokens)):
                seg_tokens.append(0)
        
        batch_seg_tokens_tensor = torch.tensor(batch_seg_tokens).cuda()
        input_tokens_tensor_all = batch_seg_tokens_tensor[:, :-1]

        with torch.no_grad():
            start_time = time.time()
            output_logits_all = model(input_tokens_tensor_all)
            if 'llama' in args.model_name:
                output_logits_all = output_logits_all[0]
            model_time = time.time() - start_time
        
        for token_id in range(max_length-1):
            output_logits = output_logits_all[:, token_id]
            output_token_tensor = batch_seg_tokens_tensor[:, token_id+1]

            # get freq table and do ac
            start_time = time.time()
            probs = torch.softmax(output_logits, dim=-1)               
            freqs = torch.round(probs * 10000000).int()
            freqs = torch.max(freqs, freqs.new_ones(freqs.size()))
            new_probs = freqs / freqs.sum(dim=1)[:, None]
            # freqs_list = freqs.cpu().numpy().tolist()
            probs_time = time.time() - start_time

            if CHECK_CROSS_ENTROPY:
                start_time = time.time()
                indices = [list(range(batch_size)), output_token_tensor.cpu().numpy().tolist()]
                entropy = torch.log(new_probs[indices]) / -math.log(2)
                entropy_sum += (entropy * (output_token_tensor != 0)).sum().item()
                
                # cross_entropy = criterion(logits.view(batch_size, -1), output_token_tensor.view(batch_size))
                token_count += (output_token_tensor != 0).sum().item()

                entropy_time = time.time() - start_time
                pbar.set_description("Total Entropy: {:.4f}, Ave Entropy: {:.4f}, model_time: {:.6f}, probs_time: {:.6f}, entropy_time: {:.6f}".format(entropy_sum, entropy_sum / token_count, model_time, probs_time, entropy_time))

        pbar.update()
        print("[Compression Infos] Finish {} / {} batches.".format(i+1, num_batches))
    
    pbar.close()

    total_bin_size_min = math.ceil(entropy_sum / 8)

    return total_bin_size_min


def compress_unks_with_azstd(args, unk_seqs):
    unk_out_file_path = os.path.join(args.tmp_processed_dir, "unk.txt")
    unk_out_file = open(unk_out_file_path, "wb")
    for i in range(len(unk_seqs)):
        unk_out_file.write(unk_seqs[i].encode())
        if i != len(unk_seqs) - 1:
            unk_out_file.write("\n".encode())
    unk_out_file.close()
    unk_txt_size = os.path.getsize(unk_out_file_path)
    
    unk_zst_file_path = unk_out_file_path + ".zst"
    # azstd_cmd = "{} -z {} {}".format(args.azstd_bin_file, os.path.abspath(unk_out_file_path), os.path.abspath(unk_zst_file_path))
    zstd_cmd = "{} -19 {} -o {}".format(args.azstd_bin_file, os.path.abspath(unk_out_file_path), os.path.abspath(unk_zst_file_path))
    result = os.system(zstd_cmd)
    
    if result == 0:
        unk_zst_size = os.path.getsize(unk_zst_file_path)
        return unk_txt_size, unk_zst_size
    else:
        return unk_txt_size, 0


def do_text_compression_with_LLM(args, model, tokenizer):
    """
        使用Llama, 执行完整的文本压缩测试，并获取压缩结果
    """
    if args.segment_length == 0:
        forward_method = "infinite"
    else:
        forward_method = "segment_{}".format(args.segment_length)

    segment_tokens, unk_words = compression_preprocess_with_LLM(args, args.input_file, tokenizer)
    number_of_total_tokens = sum([len(seg) for seg in segment_tokens])
    print(f"[INFO] Total token number is {number_of_total_tokens}")

    # 通过模型去压缩segment_tokens
    if "rwkv" in args.model_name:
        total_bin_size = compress_tokens_with_ArithmeticCoder(args, model, segment_tokens)
    else:
        total_bin_size = compress_tokens_with_ArithmeticCoder_with_Transformer(args, model, segment_tokens)

    # 通过zstd压缩unks
    if not len(unk_words):
        unk_txt_size, unk_zst_size = 0, 0
    else:
        unk_txt_size, unk_zst_size = compress_unks_with_azstd(args, unk_words)
    with open(os.path.join(args.tmp_processed_dir, "results.txt"), "w") as f:
        f.write("total_bin_size: {}\n".format(total_bin_size))
        f.write("unk_txt_size: {}\n".format(unk_txt_size))
        f.write("unk_zst_size: {}\n".format(unk_zst_size))


def do_text_compression(args, model, word2dict):
    """
        执行完整的文本压缩测试，并获取压缩结果
    """
    if args.segment_length == 0:
        forward_method = "infinite"
    else:
        forward_method = "segment_{}".format(args.segment_length)

    # 对input_file进行预处理
    dataset_name = getattr(args, "dataset_name", "default")
    if dataset_name == "enwik_ascii":
        segment_tokens, unk_words = compression_preprocess_ascii(args, args.input_file, word2dict)
    else:
        segment_tokens, unk_words = compression_preprocess(args, args.input_file, word2dict)

    number_of_total_tokens = sum([len(seg) for seg in segment_tokens])
    print(f"[INFO] Total token number is {number_of_total_tokens}")
    # 通过模型去压缩segment_tokens
    if "transformer_infer" == args.model_name:
        total_bin_size = compress_tokens_with_ArithmeticCoder_with_Transformer(args, model, segment_tokens)
    else:
        total_bin_size = compress_tokens_with_ArithmeticCoder(args, model, segment_tokens)

    # 通过zstd压缩unks
    if not len(unk_words):
        unk_txt_size, unk_zst_size = 0, 0
    else:
        unk_txt_size, unk_zst_size = compress_unks_with_azstd(args, unk_words)

    with open(os.path.join(args.tmp_processed_dir, "results.txt"), "w") as f:
        f.write("total_bin_size: {}\n".format(total_bin_size))
        f.write("unk_txt_size: {}\n".format(unk_txt_size))
        f.write("unk_zst_size: {}\n".format(unk_zst_size))
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    device = torch.device(args.device)

    # 读取Config文件并且更新args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # 读取字典
    if "ascii" in args.dataset_name:
        word2id_dict = get_ascii_word2id_dict()
    elif "llama" in args.model_name:
        word2id_dict = None
    elif "rwkv" in args.model_name and "pile" in args.config_file:
        word2id_dict = None
    else:
        word2id_dict = get_word2id_dict(args.vocab_path)

    # 搭建TokenPredictor
    if "rwkv" in args.model_name:
        args.model_name = args.model_name + "_infer_for_script"
    elif "transformer_xl" in args.model_name:
        args.model_name = args.model_name + "_infer_for_coreml"
    elif "transformer" == args.model_name:
        args.model_name = args.model_name + "_infer"

    if "llama" in args.model_name:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt_dir)
        model = AutoModelForCausalLM.from_pretrained(args.model_ckpt_dir)
        model.to(device)

    elif "rwkv" in args.model_name and "pile" in args.config_file:
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
        args.vocab_size = len(tokenizer)
        model, criterion = build_model_main(args)
        model.to(device)
        model.eval()

        # Load Trained Checkpoint
        assert args.pretrain_model_path is not None, "pretrained model path cannot be None !"
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')

        from collections import OrderedDict
        _ignorekeywordlist = []
        ignorelist = []

        print("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in misc.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})
        if "rwkv" in args.model_name:
            for key, value in _tmp_st.items():
                if "att.time_mix" in key or "ffn.time_mix" in key:
                    _tmp_st[key] = value[0]
            
            _tmp_st["ln0.weight"] = _tmp_st["blocks.0.ln0.weight"]
            _tmp_st["ln0.bias"] = _tmp_st["blocks.0.ln0.bias"]
            _tmp_st.pop("blocks.0.ln0.weight")
            _tmp_st.pop("blocks.0.ln0.bias")
        _load_output = model.load_state_dict(_tmp_st, strict=False)
        print(str(_load_output))

    else:
        args.vocab_size = len(word2id_dict)
        model, criterion = build_model_main(args)
        model.to(device)
        model.eval()

        # 读取Trained Checkpoint
        assert args.pretrain_model_path is not None, "pretrained model path cannot be None !"
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = []
        ignorelist = []

        print("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in misc.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})
        if "rwkv" in args.model_name:
            for key, value in _tmp_st.items():
                if "att.time_mix" in key or "ffn.time_mix" in key:
                    _tmp_st[key] = value[0]

                if "rwkv_tc_hira_rbranch" in args.model_name:
                    if "key.weight" in key:
                        for branch_id in range(args.rwkv_rank_branch):
                            A_key = key.replace("key.weight", f"key_A_list.{branch_id}.weight")
                            B_key = key.replace("key.weight", f"key_B_list.{branch_id}.weight")
                            _tmp_st[key] = _tmp_st[key] + _tmp_st[B_key] @ _tmp_st[A_key]
                    elif "value.weight" in key:
                        for branch_id in range(args.rwkv_rank_branch):
                            A_key = key.replace("value.weight", f"value_A_list.{branch_id}.weight")
                            B_key = key.replace("value.weight", f"value_B_list.{branch_id}.weight")
                            _tmp_st[key] = _tmp_st[key] + _tmp_st[B_key] @ _tmp_st[A_key]
                    elif "receptance.weight" in key:
                        for branch_id in range(args.rwkv_rank_branch):
                            A_key = key.replace("receptance.weight", f"receptance_A_list.{branch_id}.weight")
                            B_key = key.replace("receptance.weight", f"receptance_B_list.{branch_id}.weight")
                            _tmp_st[key] = _tmp_st[key] + _tmp_st[B_key] @ _tmp_st[A_key]

                elif "rwkv_tc_hira" in args.model_name:
                    if "key.weight" in key:
                        A_key = key.replace("key.weight", "key_A.weight")
                        B_key = key.replace("key.weight", "key_B.weight")
                        _tmp_st[key] = _tmp_st[key] + _tmp_st[B_key] @ _tmp_st[A_key]
                    elif "value.weight" in key:
                        A_key = key.replace("value.weight", "value_A.weight")
                        B_key = key.replace("value.weight", "value_B.weight")
                        _tmp_st[key] = _tmp_st[key] + _tmp_st[B_key] @ _tmp_st[A_key]
                    elif "receptance.weight" in key:
                        A_key = key.replace("receptance.weight", "receptance_A.weight")
                        B_key = key.replace("receptance.weight", "receptance_B.weight")
                        _tmp_st[key] = _tmp_st[key] + _tmp_st[B_key] @ _tmp_st[A_key]

            _tmp_st["ln0.weight"] = _tmp_st["blocks.0.ln0.weight"]
            _tmp_st["ln0.bias"] = _tmp_st["blocks.0.ln0.bias"]
            _tmp_st.pop("blocks.0.ln0.weight")
            _tmp_st.pop("blocks.0.ln0.bias")

        _load_output = model.load_state_dict(_tmp_st, strict=False)
        print(str(_load_output))

    # start compression
    if not os.path.exists(args.tmp_processed_dir):
        os.makedirs(args.tmp_processed_dir)
    else:
        os.system(f"rm -rf {args.tmp_processed_dir}/*")

    if "llama" in args.model_name or ("rwkv" in args.model_name and "pile" in args.config_file):
        do_text_compression_with_LLM(args, model, tokenizer)
    else:
        do_text_compression(args, model, word2id_dict)


    
    
