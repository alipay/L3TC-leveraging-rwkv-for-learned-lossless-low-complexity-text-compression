import os
import glob
import sys
import torch
import json
import math
import time
import logging
import numpy as np
import random
import argparse
import torch.distributed as dist
from torch.nn import functional as F
os.environ['USE_WKV_CUDA_FOR_RWKV'] = 'True'

from pathlib import Path
from util.slconfig import DictAction, SLConfig
from util.logger import setup_logger
from util import misc
from util.utils import BestMetricHolder, SmoothedValue
from torch.optim import Adam
from models.RWKV_V4.rwkv_v4_train import L2Wrap
from torch.utils.data import DataLoader, DistributedSampler
from dataset.dataset import TransformerXLTrainDataSet, TransformerXLTestDataSet
from dataset.enwik_dataset import EnWikTrainDataSet, EnWikTestDataSet
from dataset.enwik_ascii_dataset import EnWikASCIITrainDataSet, EnWikASCIITestDataSet


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformerXL predictor', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # training parameters
    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1204, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument("--distributed", default=False, action='store_true')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    
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


def print_param_norm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def evaluate(model, data_iter, epoch=0):
    # test
    with torch.no_grad():
        model.eval()

        num_total_tokens = 0
        num_correct_tokens = 0
        sum_cross_entropy = 0.0

        for i, test_data in enumerate(data_iter):
            test_data = {k: v.to(device) for k, v in test_data.items()}

            input_token = test_data['input_token']
            input_types = test_data['input_types']
            output_token = test_data['output_token']
            output_types = test_data['output_types']

            output = model(input_token, input_types, train=False, output_token=output_token, output_types=output_types, criterion=criterion)
            cross_entropy = output['cross_entropy']
            hit_count = output['hit_count']
            token_count = output['token_count']
            
            num_correct_tokens += hit_count
            num_total_tokens += token_count
            sum_cross_entropy += cross_entropy

            print("Eval_Epoch {}: {} / {}, avg_cross_entropy: {}".format(epoch, i+1, len(data_iter), sum_cross_entropy / num_total_tokens))

        acc_rate = float(num_correct_tokens) / float(num_total_tokens)
        acc_rate = round(acc_rate, 2)

        avg_ce = sum_cross_entropy / num_total_tokens
        avg_ce = round(avg_ce, 4)
        
        print('accuracy：%s' % acc_rate)
        print('average ppl：%s' % avg_ce)

        test_stats = {
            "acc_rate": acc_rate,
            "avg_ce": avg_ce
        }
    
    return test_stats


def train_one_epoch(model, criterion, data_iter, optimizer, device, epoch, max_norm,
                    lr_scheduler=None, args=None, logger=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    model.train()
    criterion.train()
    
    metric_logger = misc.MetricLogger(delimiter="  ", weight_dict={})
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq

    _cnt = 0
    for data in metric_logger.log_every(data_iter, print_freq, header, logger=logger):
        data = {k: v.to(device) for k, v in data.items()}
        input_token = data['input_token']
        input_types = data['input_types']
        output_token = data['output_token']
        output_types = data['output_types']

        if "multi_pred" in args.model_name:
            output_token_2 = F.pad(output_token, (-1, 1, 0, 0, 0, 0), value=2)
            output_types_2 = F.pad(output_types, (-1, 1, 0, 0, 0, 0), value=0)

            output_token_3 = F.pad(output_token, (-2, 2, 0, 0, 0, 0), value=2)
            output_types_3 = F.pad(output_types, (-2, 2, 0, 0, 0, 0), value=0)

            output_token_4 = F.pad(output_token, (-3, 3, 0, 0, 0, 0), value=2)
            output_types_4 = F.pad(output_types, (-3, 3, 0, 0, 0, 0), value=0)
        
        with torch.cuda.amp.autocast(enabled=args.amp):
            if "gmm" in args.model_name:
                output = model(input_token, input_types, train=True, output_token=output_token)
            else:
                output = model(input_token, input_types, train=True)

            if "multi_pred" in args.model_name:
                out1, out2, out3, out4 = output
                output_view = out1.view([-1, out1.size(-1)])
                output_token = output_token.view([-1])
                output_types = output_types.view([-1])
                loss_1 = criterion(output_view, output_token)
                loss_1 = (loss_1 * output_types).sum() / max(output_types.sum(), 1)

                output_view_2 = out2.view([-1, out2.size(-1)])
                output_token_2 = output_token_2.view([-1])
                output_types_2 = output_types_2.view([-1])
                loss_2 = criterion(output_view_2, output_token_2)
                loss_2 = (loss_2 * output_types_2).sum() / max(output_types_2.sum(), 1)

                output_view_3 = out3.view([-1, out3.size(-1)])
                output_token_3 = output_token_3.view([-1])
                output_types_3 = output_types_3.view([-1])
                loss_3 = criterion(output_view_3, output_token_3)
                loss_3 = (loss_3 * output_types_3).sum() / max(output_types_3.sum(), 1)

                output_view_4 = out4.view([-1, out4.size(-1)])
                output_token_4 = output_token_4.view([-1])
                output_types_4 = output_types_4.view([-1])
                loss_4 = criterion(output_view_4, output_token_4)
                loss_4 = (loss_4 * output_types_4).sum() / max(output_types_4.sum(), 1)

                loss = loss_1 + loss_2 + loss_3 + loss_4
            elif "gmm" in args.model_name:
                output_view = output.view([-1, output.size(-1)])
                output_types = output_types.view([-1])
                loss = -torch.log(output_view).view([-1])
                loss = (loss * output_types).sum() / max(output_types.sum(), 1)
            else:
                output_view = output.view([-1, output.size(-1)])
                output_token = output_token.view([-1])
                output_types = output_types.view([-1])
                loss = criterion(output_view, output_token)
                loss = (loss * output_types).sum() / max(output_types.sum(), 1)

            if 'rwkv' in args.model_name:
                if "multi_pred" in args.model_name:
                    pass
                else:
                    loss = L2Wrap.apply(loss, output)

            if args.distributed:
                dist.barrier()
                dist.all_reduce(loss)
                loss = loss / args.world_size
            loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print("loss: ", loss)
            print("output_view: ", output_view)
            sys.exit(1)
        
        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            loss.backward()
            # param_norm = print_param_norm(model.parameters())
            # print("param_norm is {}".format(param_norm))
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        metric_logger.update(loss=(loss_value, output_types.sum()), **{})
        metric_logger.update(lr=(optimizer.param_groups[0]["lr"], 1))

        _cnt += 1
        if lr_scheduler is not None:
            lr_scheduler.step()

        if args.debug:
            if _cnt % 5000 == 0:
                print("BREAK!"*5)
                break
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    train_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    return train_stat


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TransformerXL training scripts', parents=[get_args_parser()])
    args = parser.parse_args()
    device = torch.device(args.device)

    # create log directory
    timestamp = time.strftime('_%Y%m%d_%H%M%S', time.localtime())
    args.output_dir = os.path.join(args.output_dir, os.path.basename(args.config_file)[:-3] + timestamp)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # whether to use distributed training
    if args.distributed:
        # os.environ['RANK'] = "0"
        # os.environ['WORLD_SIZE'] = "4"
        # os.environ['MASTER_ADDR'] = "127.0.0.1"
        # os.environ['MASTER_PORT'] = "3002"
        
        dist.init_process_group("nccl")
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        logging.info("=" * 100, args.rank, args.world_size)
        torch.distributed.barrier()
        misc.setup_for_distributed(args.rank == 0)
    else:
        args.distributed = False
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0

    # Load Config and update args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        try:
            cfg.dump(save_cfg_path)
        except Exception:
            pass
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # load dictionary
    if "ascii" in args.dataset_name:
        word2id_dict = get_ascii_word2id_dict()
    else:
        word2id_dict = get_word2id_dict(args.vocab_path)

    # set global random seed
    seed = args.random_seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if 'spm_image_bpe_16384' in args.vocab_path:
        args.vocab_size = 16384
    else:
        args.vocab_size = len(word2id_dict)

    # build l3tc model
    model, criterion = build_model_main(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module

    # set logger
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="transformer_text_compression")
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info("args: " + str(args) + '\n')

    n_parameters_without_embed_fc = 0
    for name, val in model.named_parameters():
        if "rwkv" in args.model_name:
            if "head" in name or "emb" in name:
                continue
        elif args.model_name == "transformer":
            if "token_embed" in name or "pos_embed" in name or "generator" in name:
                continue
        elif "transformer_xl" in args.model_name:
            if "token_emd" in name:
                continue

        n_parameters_without_embed_fc += val.numel()
    logger.info('number of params without embed && fc:'+str(n_parameters_without_embed_fc))
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    # logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    # build TrainTestDataset
    dataset_name = getattr(args, "dataset_name", "default")
    if dataset_name == "enwik":
        dataset_train = EnWikTrainDataSet(args, args.train_file, word2id_dict)
        dataset_val = EnWikTestDataSet(args, args.test_file, word2id_dict)
    elif dataset_name == "enwik_ascii":
        dataset_train = EnWikASCIITrainDataSet(args, args.train_file, word2id_dict)
        dataset_val = EnWikASCIITestDataSet(args, args.test_file, word2id_dict)
    else:
        dataset_train = TransformerXLTrainDataSet(args, args.train_file, word2id_dict)
        dataset_val = TransformerXLTestDataSet(args, args.test_file, word2id_dict)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   num_workers=args.num_workers)

    eval_batch_size = 1024 if 'rwkv' in args.model_name else 1
    data_loader_val = DataLoader(dataset_val, batch_size=eval_batch_size, sampler=sampler_val,
                                 drop_last=False, num_workers=args.num_workers)

    # build optimizer and loss function
    if 'rwkv' in args.model_name:
        optimizer = model_without_ddp.configure_optimizers(args)
    else:
        optimizer = Adam(model_without_ddp.parameters(), lr=args.learning_rate)

    if args.scheduler[0] == "multi_epoch":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler[1], gamma=args.scheduler[2])
    elif args.scheduler[0] == "step_lr":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler[1], gamma=args.scheduler[2])
    elif args.scheduler[0] == "exponential_lr":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler[1])
    else:
        lr_scheduler = None

    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])           

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    # load pretrained model
    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        # _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        _ignorekeywordlist = []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in misc.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))     
    
    if args.eval:
        test_stats = evaluate(model, data_loader_val)
        test_stats = evaluate(model, data_loader_val)
        sys.exit()

    # best result holder
    best_map_holder = BestMetricHolder(init_res=100.0, better='small', use_ema=False)

    # start training
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch,
                                      args.clip_max_norm, lr_scheduler=lr_scheduler if args.scheduler != "multi_epoch" else None, args=args, 
                                      logger=(logger if args.save_log else None) )
        
        if args.scheduler != "multi_epoch" and lr_scheduler is not None:
            lr_scheduler.step()

        if args.output_dir:
            # traverse current checkpoint
            all_exist_ckpts = glob.glob(os.path.join(args.output_dir, "*pth"))
            all_exist_ckpts = [ckptname for ckptname in all_exist_ckpts if os.path.basename(ckptname) not in ['checkpoint.pth', 'checkpoint_best.pth']]
            all_exist_ckpts = sorted(all_exist_ckpts)
            if len(all_exist_ckpts) >= 3:
                rm_cmd = "rm -f {}".format(all_exist_ckpts[0])
                os.system(rm_cmd)

            # checkpoint_paths = [os.path.join(args.output_dir, 'checkpoint.pth')]
            checkpoint_paths = []
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                    'epoch': epoch,
                    'args': args,
                }
                misc.save_on_master(weights, checkpoint_path)
        
        test_stats = evaluate(model, data_loader_val)
        avg_cross_entropy = test_stats['avg_ce']
        _isbest = best_map_holder.update(avg_cross_entropy, epoch, is_ema=False)
        if _isbest:
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
            misc.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
        }

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats, indent=4) + "\n")
        
        if args.debug:
            if epoch >= 1:
                print("BREAK!"*5)
                break