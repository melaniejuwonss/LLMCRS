import os

import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, T5Tokenizer
import transformers
import torch
import json
import argparse
import logging
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from pytz import timezone

from chatgpt_test import chatgpt_test
from dataset_rec import CRSDatasetRec
from llama_finetune import llama_finetune
from llama_test import LLaMaEvaluator
from t5_finetune import t5_finetune
from t5_test import T5Evaluator
from utils.data import read_data
from utils.parser import parse_args, dir_init
from os.path import dirname, realpath


def convertIds2Names(id_list, id2name):
    return [id2name[item] for item in id_list]


if __name__ == '__main__':
    args = parse_args()
    args = dir_init(args)
    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))
    result_path = os.path.join(args.home, args.output_dir, args.base_model.replace('/', '-'))
    if not os.path.exists(result_path): os.mkdir(result_path)
    args.log_name = mdhm + '_' + args.base_model.replace('/', '-') + '_' + f'rq{args.rq_num}' + '_' + args.log_name
    if args.log_file == '':
        log_file = open(os.path.join(args.home, result_path, f'{mdhm}_rq{args.rq_num}.json'), 'a', buffering=1,
                        encoding='UTF-8')
    else:
        log_file = open(os.path.join(args.home, result_path, f'{args.log_file}.json'), 'a', buffering=1,
                        encoding='UTF-8')

    args.lora_weights = os.path.join(args.home, args.lora_weights)
    args.log_file = log_file

    args.wandb_project = "LLMCRS"
    args.wandb_run_name = args.log_name

    wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    if args.stage.lower() == "crs":
        ROOT_PATH = dirname(realpath(__file__))
        DATASET_PATH = os.path.join(ROOT_PATH, args.dataset_path)
        crs_dataset = CRSDatasetRec(args, DATASET_PATH)
        train_data = crs_dataset.train_data
        valid_data = crs_dataset.valid_data
        test_data = crs_dataset.test_data

        if 'train' in args.mode:
            instructions = [i['context_tokens'] for i in train_data]
            labels = [i['item'] for i in train_data]
            # labels = [crs_dataset.entityid2name[i['item']] for i in train_data]

        elif 'test' == args.mode:
            instructions = [i['context_tokens'] for i in test_data]
            # labels = [crs_dataset.entityid2name[i['item']] for i in test_data]
            labels = [i['item'] for i in test_data]
            # negItems = [convertIds2Names(i['negItems'], crs_dataset.entityid2name) for i in test_data]
            # for idx, data in enumerate(test_data):
            #     negItems = data['negItems']
            #     negItems = [crs_dataset.entityid2name[item] for item in negItems]
            #     test_data[idx]['negItems'] = negItems

    elif args.stage.lower() == "quiz":
        question_data = read_data(args)
        instructions = [i[0] for i in question_data]
        labels = [i[1] for i in question_data]

    if 'gpt' in args.base_model.lower():
        chatgpt_test(args=args, instructions=instructions, labels=labels)

    if 'llama' in args.base_model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(args.base_model)

        evaluator = LLaMaEvaluator(args=args, tokenizer=tokenizer, instructions=instructions, labels=labels,
                                   prompt_template_name=args.prompt)
        if 'train' in args.mode:
            llama_finetune(args=args, evaluator=evaluator, tokenizer=tokenizer, instructions=instructions,
                           labels=labels, num_epochs=args.epoch, prompt_template_name=args.prompt)
        if 'test' == args.mode:
            evaluator.test()

    if 't5' in args.base_model.lower():
        tokenizer = T5Tokenizer.from_pretrained(args.base_model)

        evaluator = T5Evaluator(args=args, tokenizer=tokenizer, instructions=instructions, labels=labels,
                                prompt_template_name=args.prompt)
        if 'train' in args.mode:
            t5_finetune(args=args, evaluator=evaluator, tokenizer=tokenizer, instructions=instructions,
                        labels=labels, num_epochs=args.epoch, prompt_template_name=args.prompt)
        if 'test' == args.mode:
            evaluator.test()
