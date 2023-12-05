import argparse
import json
import logging
import os.path as osp
from typing import Union
import os

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    # common
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--device_id', type=str, default='0')
    parser.add_argument('--max_dialog_len', type=int, default=128)
    parser.add_argument('--rq_num', type=str, default='1')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        choices=['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-13b-hf',
                                 'meta-llama/Llama-2-7b-chat-hf', 'gpt-3.5-turbo', 'google/flan-t5-large', 't5-small',
                                 't5-large'])
    parser.add_argument('--dataset_path', type=str, default='data/redial')
    parser.add_argument('--stage', type=str, default='crs')  # crs or quiz
    parser.add_argument('--model_name', type=str, default='llama')
    parser.add_argument('--num_device', type=int, default=1)
    parser.add_argument("--write", action='store_true', help="Whether to write of results.")
    parser.add_argument("--lora_weights", type=str, default='lora-alpaca')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test', 'valid', 'train_test'])
    parser.add_argument('--log_name', type=str, default='')
    parser.add_argument('--prompt', type=str, default='withoutCoT')
    parser.add_argument('--data_type', type=str, default='default')
    parser.add_argument('--isNew', type=bool, default=False)
    parser.add_argument('--oversample_ratio', type=int, default=1)
    parser.add_argument('--train_response', type=bool, default=False)
    parser.add_argument('--num_reviews', type=int, default=1)
    parser.add_argument('--train_on_inputs', type=bool, default=True)
    parser.add_argument('--merge', type=bool, default=False)
    parser.add_argument('--quiz_merge', type=bool, default=False)

    parser.add_argument('--all_merge', type=bool, default=False)

    # ChatGPT
    parser.add_argument('--log_file', type=str, default='')
    parser.add_argument('--chatgpt_cnt', type=int, default=0)
    parser.add_argument('--chatgpt_hit', type=int, default=0)
    parser.add_argument('--chatgpt_key', type=str, default="")

    args = parser.parse_args()
    args.device_id = f'cuda:{args.device_id}' if args.device_id else "cpu"
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # torch.cuda.set_device(0)

    args.num_device = torch.cuda.device_count()
    print(f"NUM_DEVICE{args.num_device}")

    args.output_dir = 'result'
    if not os.path.exists(args.output_dir): os.mkdir(args.output_dir)

    args.score_dir = 'score'
    if not os.path.exists(args.score_dir): os.mkdir(args.score_dir)

    print(args)
    # logging.info(args)
    return args


def dir_init(default_args):
    from copy import deepcopy
    """ args 받은다음, device, Home directory, data_dir, log_dir, output_dir, 들 지정하고, Path들 체크해서  """
    args = deepcopy(default_args)
    from platform import system as sysChecker
    if sysChecker() == 'Linux':
        args.home = os.path.dirname(os.path.dirname(__file__))
    elif sysChecker() == "Windows":
        args.home = ''
        # args.batch_size, args.num_epochs = 4, 2
        # args.debug = True
        pass  # HJ local
    else:
        raise Exception("Check Your Platform Setting (Linux-Server or Windows)")

    return args
