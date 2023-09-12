import os

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import json
import argparse
import logging
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from pytz import timezone

from llama_test import llama_test
from utils.data import read_data
from utils.parser import parse_args


class RQ(Dataset):
    def __init__(self, tokenizer, args):
        super(Dataset, self).__init__()
        self.data_samples = []
        self.tokenizer = tokenizer
        self.args = args
        self.read_data()

    def read_data(self):
        RQ_data = json.load((open('data/rq' + str(self.args.rq_num) + '.json', 'r', encoding='utf-8')))
        question, answer = [], []
        for data in RQ_data:
            question.append(data['Question'])
            answer.append(data['Answer'])

        # tokenized_input = self.tokenizer(question, return_tensors="pt", padding=True, return_token_type_ids=False).to(
        #     self.args.device_id)
        # tokenized_output = self.tokenizer(answer, return_tensors="pt", padding=True, return_token_type_ids=False).to(
        #     self.args.device_id)
        for t_input, t_output in zip(question, answer):
            self.data_samples.append((t_input, t_output))

    def __getitem__(self, idx):
        input = self.data_samples[idx][0]
        output = self.data_samples[idx][1]

        return input, output

    def __len__(self):
        return len(self.data_samples)


class RQCollator:
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, data_batch):
        question_batch, resp_batch, input_len_batch = [], [], []
        for data_input, data_output in data_batch:
            question_batch.append(data_input)
            input_len_batch.append(len(data_input))
            resp_batch.append(data_output)

        input_batch = {}
        tokenized_input = self.tokenizer(question_batch, return_tensors="pt", padding=True,
                                         return_token_type_ids=False).to(
            self.args.device_id)
        input_batch['answer'] = resp_batch
        input_batch['question_len'] = torch.sum(tokenized_input.attention_mask, dim=1)
        input_batch['question'] = tokenized_input

        return input_batch


def evaluate(gen_seq, answer, log_file):
    # gen_output, result_f = [], []
    # for seq, len in zip(gen_seq, input_len):
    #     gen_output.append(seq[len:])
    # decoded_output = tokenizer.batch_decode(gen_output, skip_special_tokens=True)
    for output, label in zip(gen_seq, answer):
        log_file.write(json.dumps({'GEN': output, 'ANSWER': label}, ensure_ascii=False) + '\n')
    #     result_f.append({'GEN': output, 'ANSWER': label})
    # with open('result/llama/' + str(rq_num) + '_result.json', 'w', encoding='utf-8') as f_write:
    #     f_write.write(json.dumps(result_f, indent=4))


class Textdataset(Dataset):
    def __init__(self, data_samples):
        self.data_samples = data_samples

    def __getitem__(self, idx):
        input = self.data_samples[idx][0]
        output = self.data_samples[idx][1]

        return input, output

    def __len__(self):
        return len(self.data_samples)


if __name__ == '__main__':
    args = parse_args()
    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))
    result_path = os.path.join(args.output_dir, args.base_model.replace('/', '-'))
    if not os.path.exists(result_path): os.mkdir(result_path)

    log_file = open(os.path.join(result_path, f'rq{args.rq_num}_{mdhm}.json'), 'a', buffering=1, encoding='UTF-8')
    question_data = read_data(args)
    instructions = [i[0] for i in question_data]

    # question_dataset = Textdataset(question_data)
    # dataloader = DataLoader(question_dataset, batch_size=args.batch_size, shuffle=False)

    # model_name = args.model_name
    # tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token_id = (
    #         0  # unk. we want this to be different from the eos token
    #     )
    #     tokenizer.padding_side = "left"  # Allow batched inference
    # rqDataset = RQ(tokenizer, args)
    # rqCollator = RQCollator(tokenizer, args)
    # dataloader = DataLoader(rqDataset, batch_size=args.batch_size, shuffle=False, collate_fn=rqCollator)
    # # CUDA_LAUNCH_BLOCKING = 1

    # inputs = tokenizer(question, return_tensors="pt", padding=True, return_token_type_ids=False).to(model.device)
    # print(inputs['input_ids'].shape)

    llama_test(args=args, instructions=instructions)
    # for batches in tqdm(dataloader, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
    #     with torch.no_grad():
    #         output_sequences = llama_test(args=args, instructions=batches[0])
    #         evaluate(output_sequences, batches['answer'], log_file)
