import argparse
import json
import logging
import os.path as osp
from typing import Union
import os


def parse_args():
    parser = argparse.ArgumentParser()
    # common
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_input_length', type=int, default=200)
    parser.add_argument('--device_id', type=str, default='0')
    parser.add_argument('--rq_num', type=int, default=1)
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-hf',
                        choices=['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-13b-hf', 'meta-llama/Llama-2-7b-chat-hf'])

    args = parser.parse_args()
    args.device_id = f'cuda:{args.device_id}' if args.device_id else "cpu"

    args.output_dir = 'result'
    if not os.path.exists(args.output_dir): os.mkdir(args.output_dir)

    print(args)
    # logging.info(args)
    return args