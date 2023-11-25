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
    result_path = os.path.join(args.home, args.output_dir, args.base_model.replace('/', '-'), mdhm[:4])
    score_path = os.path.join(args.home, args.score_dir, args.base_model.replace('/', '-'), mdhm[:4])
    if not os.path.exists(result_path): os.mkdir(result_path)
    if not os.path.exists(score_path): os.mkdir(score_path)
    args.log_name = mdhm + '_' + args.base_model.replace('/', '-') + '_' + args.log_name
    if 'gpt' in args.base_model.lower() or args.lora_weights[-1].isdigit() is True:
        log_file = open(os.path.join(args.home, result_path, f'{args.log_name}.json'), 'a', buffering=1,
                        encoding='UTF-8')
        args.log_file = log_file
    args.result_path = os.path.join(args.home, result_path)
    score_file = open(os.path.join(args.home, score_path, f'{args.log_name}.json'), 'a', buffering=1, encoding='UTF-8')
    score_file.write('Overall\tMentioned\tNot-mentioned\tGen-mentioned\tGen-Not-mentioned\n')
    args.lora_weights = os.path.join(args.home, args.lora_weights)

    args.score_file = score_file

    args.wandb_project = "LLMCRS"
    args.wandb_run_name = args.log_name

    wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    if args.stage.lower() == "crs":
        ROOT_PATH = dirname(realpath(__file__))
        DATASET_PATH = os.path.join(ROOT_PATH, args.dataset_path)
        args.dataset_path = DATASET_PATH
        crs_dataset = CRSDatasetRec(args)
        if "cot" in args.data_type:
            cot_data_path = os.path.join(DATASET_PATH, 'cot')
            with open(os.path.join(cot_data_path, f'train_data_{args.data_type}.json'), 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            with open(os.path.join(cot_data_path, f'test_data_{args.data_type}.json'), 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            if args.oversampling:
                train_new_idx = json.load(open(f'data/redial/train_new_idx.json', 'r', encoding='utf-8'))
                train_old_idx = list(range(len(train_data)))
                train_old_idx = [item for item in train_old_idx if item not in train_new_idx]
                rec_train_data = [train_data[x] for x in train_new_idx]
                chat_train_data = [train_data[x] for x in train_old_idx]
                oversample_ratio = int(len(rec_train_data) / len(chat_train_data))
                chat_train_data = chat_train_data * oversample_ratio
                train_data = rec_train_data + chat_train_data
        elif "cot" not in args.data_type:
            train_data = crs_dataset.train_data
            valid_data = crs_dataset.valid_data
            test_data = crs_dataset.test_data

        new_idx = json.load(open(os.path.join(args.dataset_path, 'train_new_idx.json'), 'r', encoding='utf-8'))
        train_data = [{'context_tokens': data['context_tokens'], 'item': data['item'], 'isNew': idx in new_idx} for
                      idx, data in enumerate(train_data)]

        # if 'train' in args.mode:
        train_instructions = [i['context_tokens'] for i in train_data]
        if args.data_type == "augment":
            train_labels = [crs_dataset.entityid2name[i['item']] for i in train_data]
        else:
            train_labels = [i['item'] for i in train_data]
        train_new = [i['isNew'] for i in train_data]

        # if 'test' in args.mode:
        test_instructions = [i['context_tokens'] for i in test_data]
        if args.data_type == "augment":
            test_labels = [crs_dataset.entityid2name[i['item']] for i in test_data]
        else:
            test_labels = [i['item'] for i in test_data]

        if 'valid' == args.mode:
            valid_instructions = [i['context_tokens'] for i in valid_data]
            valid_labels = [crs_dataset.entityid2name[i['item']] for i in valid_data]


    elif args.stage.lower() == "quiz":
        question_data = read_data(args)
        instructions = [i[0] for i in question_data]
        labels = [i[1] for i in question_data]

    if 'gpt' in args.base_model.lower():
        if args.mode == "train":
            chatgpt_test(args=args, instructions=train_instructions, labels=train_labels)
        else:
            chatgpt_test(args=args, instructions=test_instructions, labels=test_labels)

    if 'llama' in args.base_model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(args.base_model)

        evaluator = LLaMaEvaluator(args=args, tokenizer=tokenizer, instructions=test_instructions, labels=test_labels,
                                   prompt_template_name=args.prompt)
        if 'train' in args.mode:
            llama_finetune(args=args, evaluator=evaluator, tokenizer=tokenizer, instructions=train_instructions,
                           labels=train_labels, isNews=train_new, num_epochs=args.epoch,
                           prompt_template_name=args.prompt)
        if 'test' in args.mode:
            # 특정 weight 지정 없이, 모든 epoch 에 해당하는 weights test
            if args.lora_weights[args.lora_weights.rfind('/') + 1:] != "lora-alpaca" and args.lora_weights[
                -1].isdigit() is False:
                origin_lora_weights = args.lora_weights
                for e in range(args.epoch):
                    args.lora_weights = origin_lora_weights + '_E' + str(int(e + 1))
                    evaluator.test(epoch=e + 1)
            elif 'train' in args.mode:
                for e in range(args.epoch):
                    args.lora_weights = os.path.join("./lora-alpaca", args.log_name + '_E' + str(int(e + 1)))
                    evaluator.test(epoch=e + 1)
            else:
                if args.lora_weights[args.lora_weights.rfind(
                        '/') + 1:] == "lora-alpaca":  # default lora_weights (i.e., not-trained LLaMa)
                    evaluator.test()
                else:
                    evaluator.test()

    if 't5' in args.base_model.lower():
        tokenizer = T5Tokenizer.from_pretrained(args.base_model)

        evaluator = T5Evaluator(args=args, tokenizer=tokenizer, instructions=test_instructions, labels=test_labels,
                                prompt_template_name=args.prompt)
        if 'train' in args.mode:
            t5_finetune(args=args, evaluator=evaluator, tokenizer=tokenizer, instructions=train_instructions,
                        labels=train_labels, num_epochs=args.epoch, prompt_template_name=args.prompt)
        if 'test' == args.mode:
            evaluator.test()
