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
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--max_input_length', type=int, default=200)
    parser.add_argument('--device_id', type=str, default='0')
    parser.add_argument('--rq_num', type=str, default='1')
    parser.add_argument('--base_model', type=str, default='gpt-3.5-turbo',
                        choices=['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-13b-hf', 'meta-llama/Llama-2-7b-chat-hf', 'gpt-3.5-turbo'])
    parser.add_argument('--model_name', type=str, default='llama')
    parser.add_argument('--log_file', type=str, default='')
    parser.add_argument('--chatgpt_cnt', type=int, default=0)
    parser.add_argument('--chatgpt_hit', type=int, default=0)
    parser.add_argument('--chatgpt_key', type=str, default="")
    parser.add_argument('--num_device', type=int, default=1)
    parser.add_argument('--log_name', type=str, default='MYTEST')
    parser.add_argument("--write", action='store_true', help="Whether to write of results.")

    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])

    args = parser.parse_args()
    args.device_id = f'cuda:{args.device_id}' if args.device_id else "cpu"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    args.num_device = torch.cuda.device_count()
    print(f"NUM_DEVICE{args.num_device}")
    args.wandb_project = "LLMCRS"
    args.wandb_run_name = args.log_name

    args.output_dir = 'result'
    if not os.path.exists(args.output_dir): os.mkdir(args.output_dir)

    print(args)
    # logging.info(args)
    return args
