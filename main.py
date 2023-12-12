import os
from loguru import logger
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
from dataset_rec import CRSDatasetRec, ReviewDataset
from llama_finetune import llama_finetune
from llama_test import LLaMaEvaluator
from t5_finetune import t5_finetune
from t5_test import T5Evaluator
from utils.data import quiz_read_data, plot_read_data, meta_plot_review_read_data, review_read_data, crs_read_data, \
    synthetic_dialog_read_pretrain_data, review_read_pretrain_data, review_passage_read_pretrain_data, \
    synthetic_dialog_read_pretrain_data, meta_read_pretrain_data, refined_review_read_pretrain_data
from utils.parser import parse_args, dir_init
from os.path import dirname, realpath


def convertIds2Names(id_list, id2name):
    return [id2name[item] for item in id_list]


def createLogFile(args):
    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))
    result_path = os.path.join(args.home, args.output_dir, args.base_model.replace('/', '-'), mdhm[:4])
    score_path = os.path.join(args.home, args.score_dir, args.base_model.replace('/', '-'), mdhm[:4])
    if not os.path.exists(result_path): os.mkdir(result_path)
    if not os.path.exists(score_path): os.mkdir(score_path)
    args.log_name = mdhm + '_' + args.base_model.replace('/', '-') + '_' + args.log_name
    if 'gpt' in args.base_model.lower() or (args.lora_weights[-1].isdigit() is True and args.mode == "test"):
        log_file = open(os.path.join(args.home, result_path, f'{args.log_name}.json'), 'a', buffering=1,
                        encoding='UTF-8')
        args.log_file = log_file
    args.result_path = os.path.join(args.home, result_path)
    score_file = open(os.path.join(args.home, score_path, f'{args.log_name}.json'), 'a', buffering=1, encoding='UTF-8')
    for key, value in vars(args).items():
        score_file.write(f"{key}:{value}\n")
    score_file.write("\n=================================================\n")
    score_file.write('Overall\tMentioned\tNot-mentioned\tGen-mentioned\tGen-Not-mentioned\n')
    args.lora_weights = os.path.join(args.home, args.lora_weights)

    args.score_file = score_file

    return args


def cutoffInstruction(instructions, length, reverse=False):
    new_instructions = []
    for data in tqdm(instructions):
        if reverse:
            new_instructions.append(tokenizer.decode(tokenizer(data).input_ids[1:][-length:]))
        else:
            new_instructions.append(tokenizer.decode(tokenizer(data).input_ids[1:][:length]))
    logger.info('[Finish Cutting-off the instructions]')
    return new_instructions


# def slidingWindow(instructions, max_length, window_size=10):
#     new_instructions = []
#     for data in tqdm(instructions):
#         input_length = tokenizer(data).input_ids[1:]
#         if input_length < max_length:
#             new_instructions.append(data)
#         for idx in range(0, )
#         new_instructions.append(tokenizer.decode(tokenizer(data).input_ids[1:][:length]))
#     logger.info('[Finish Cutting-off the instructions]')
#     return new_instructions


if __name__ == '__main__':
    args = parse_args()
    args = dir_init(args)

    args = createLogFile(args)  # Create result and score files

    # Wandb initialize
    args.wandb_project = "LLMCRS"
    args.wandb_run_name = args.log_name
    wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    # data path setting
    ROOT_PATH = dirname(realpath(__file__))
    DATASET_PATH = os.path.join(ROOT_PATH, args.dataset_path)
    args.dataset_path = DATASET_PATH

    if 'gpt' not in args.base_model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(args.base_model)

    logger.info(f'[STAGE: {args.stage.lower()}]')
    if args.stage.lower() == "quiz" or args.quiz_merge is True:  # quiz -> onlyinstruction
        quiz_train_instructions, quiz_train_labels, quiz_train_new = quiz_read_data(args, 'train')
        quiz_test_instructions, quiz_test_labels, _ = quiz_read_data(args, 'test')
        if args.JW:
            instructions, choices = [], []
            for instruction in quiz_train_instructions:
                instructions.append(instruction[:instruction.rfind('\n Choices:')])
                choices.append(instruction[instruction.find('\n Choices:'):])
            instructions = cutoffInstruction(instructions, args.cutoff)
            quiz_train_instructions = []
            for instruction, choice in zip(instructions, choices):
                quiz_train_instructions.append(instruction + choice)

            instructions, choices = [], []
            for instruction in quiz_test_instructions:
                instructions.append(instruction[:instruction.rfind('\n Choices:')])
                choices.append(instruction[instruction.find('\n Choices:'):])
            instructions = cutoffInstruction(instructions, args.cutoff)
            quiz_test_instructions = []
            for instruction, choice in zip(instructions, choices):
                quiz_test_instructions.append(instruction + choice)

    if args.stage.lower() == "plot" or args.plot_merge is True:
        plot_train_instructions, plot_train_labels, plot_train_new = plot_read_data(args, 'train')
        plot_test_instructions, plot_test_labels, _ = plot_read_data(args, 'test')

    if args.stage.lower() == 'pretrain' or args.pretrain_merge is True:
        args.prompt = 'pretrain'
        if args.TH:
            pretrain_train_instructions, pretrain_train_labels, pretrain_train_new = meta_plot_review_read_data(
                args, 'train')
            pretrain_train_instructions = cutoffInstruction(pretrain_train_instructions, args.cutoff)

            pretrain_test_instructions, pretrain_test_labels, pretrain_test_new = meta_plot_review_read_data(
                args, 'test')
            pretrain_test_instructions = pretrain_test_instructions[:100]
            pretrain_test_labels = pretrain_test_labels[:100]
            pretrain_test_instructions = cutoffInstruction(pretrain_test_instructions, args.cutoff)
            # pretrain_train_instructions2, pretrain_train_labels2, pretrain_train_new2 = meta_read_pretrain_data(args)
            # pretrain_train_instructions.extend(pretrain_train_instructions2)
            # pretrain_train_labels.extend(pretrain_train_labels)
            # pretrain_train_new.extend(pretrain_train_new2)

        elif args.JW:
            args.prompt = 'pretrain'
            pretrain_train_instructions, pretrain_train_labels, pretrain_train_new = review_passage_read_pretrain_data(
                args)
            pretrain_train_instructions = cutoffInstruction(pretrain_train_instructions, args.cutoff)  # max: 412

        else:
            # pretrain_train_instructions, pretrain_train_labels, pretrain_train_new = review_read_pretrain_data(args)
            pretrain_train_instructions, pretrain_train_labels, pretrain_train_new = meta_read_pretrain_data(args)

        pretrain_test_instructions = pretrain_train_instructions[:100]
        pretrain_test_labels = pretrain_train_labels[:100]
        # pretrain_test_new = pretrain_train_new[:100]

    if args.stage.lower() == "review" or args.review_merge is True:
        review_train_instructions, review_train_labels, review_train_new = review_read_data(args, 'train')
        review_test_instructions, review_test_labels, _ = review_read_data(args, 'test')
        if 'gpt' not in args.base_model.lower():
            review_train_instructions = cutoffInstruction(review_train_instructions, args.cutoff)
            review_test_instructions = cutoffInstruction(review_test_instructions, args.cutoff)
            if args.TH is True:
                template = "I will give you a review of a movie.\nIn the review, the movie title is masked with [title].\nHere is the review:\n%s\nBased on the review, guess the movie [title] that the above review is discussing\n\n### Response:"
                review_train_instructions = [template % data for data in review_train_instructions]
                review_test_instructions = [template % data for data in review_test_instructions]
    if args.stage.lower() == "crs" or args.crs_merge is True:
        crs_dataset = CRSDatasetRec(args)
        train_data = crs_dataset.train_data
        valid_data = crs_dataset.valid_data
        test_data = crs_dataset.test_data

        if "cot" in args.data_type:
            if args.JW is False:
                temp = args.data_type
                args.data_type = "default"
                crs_train_instructions, crs_train_labels, crs_train_new = crs_read_data(train_data, "train", args)
                crs_valid_instructions, crs_valid_labels, _ = crs_read_data(valid_data, "valid", args)
                crs_test_instructions, crs_test_labels, _ = crs_read_data(test_data, "test", args)

                crs_train_instructions = cutoffInstruction(crs_train_instructions, args.cutoff, True)
                crs_valid_instructions = cutoffInstruction(crs_valid_instructions, args.cutoff, True)
                crs_test_instructions = cutoffInstruction(crs_test_instructions, args.cutoff, True)

                args.data_type = temp
                cot_train_instructions, _, _ = crs_read_data(train_data, "train", args)
                cot_test_instructions, _, _ = crs_read_data(test_data, "test", args)

                crs_train_instructions = [crs + cot for crs, cot in zip(crs_train_instructions, cot_train_instructions)]
                crs_test_instructions = [crs + cot for crs, cot in zip(crs_test_instructions, cot_test_instructions)]
            else:
                crs_train_instructions, crs_train_labels, crs_train_new = crs_read_data(train_data, "train", args)
                crs_test_instructions, crs_test_labels, _ = crs_read_data(test_data, "test", args)

                crs_train_instructions = cutoffInstruction(crs_train_instructions, args.cutoff, True)
                crs_test_instructions = cutoffInstruction(crs_test_instructions, args.cutoff, True)
        else:
            crs_train_instructions, crs_train_labels, crs_train_new = crs_read_data(train_data, "train", args)
            crs_valid_instructions, crs_valid_labels, _ = crs_read_data(valid_data, "valid", args)
            crs_test_instructions, crs_test_labels, _ = crs_read_data(test_data, "test", args)

            crs_train_instructions = cutoffInstruction(crs_train_instructions, args.cutoff, True)
            crs_valid_instructions = cutoffInstruction(crs_valid_instructions, args.cutoff, True)
            crs_test_instructions = cutoffInstruction(crs_test_instructions, args.cutoff, True)

        crs_train_instructions_addprompt, crs_test_instructions_addprompt, crs_valid_instructions_addprompt = [], [], []
        # if args.JW:
        #     args.template = "onlyinstruction"
        #     for data in tqdm(crs_train_instructions):
        #         crs_train_instructions_addprompt.append(
        #             f"Pretend you are a movie recommender system. I will give you a dialogue between a user and you (a recommender system).\n\nHere is the dialogue:\n{data}\n\nGuess which movie should be recommended to the user.")  # \nSystem: You should watch [BLANK]. Based on the conversation, guess the item for [BLANK]."
        #     for data in crs_valid_instructions:
        #         crs_valid_instructions_addprompt.append(
        #             f"Pretend you are a movie recommender system. I will give you a dialogue between a user and you (a recommender system).\n\nHere is the dialogue:\n{data}\n\nGuess which movie should be recommended to the user.")  # \nSystem: You should watch [BLANK]. Based on the conversation, guess the item for [BLANK]."
        #     for data in tqdm(crs_test_instructions):
        #         crs_test_instructions_addprompt.append(
        #             f"Pretend you are a movie recommender system. I will give you a dialogue between a user and you (a recommender system).\n\nHere is the dialogue:\n{data}\n\nGuess which movie should be recommended to the user.")
        if args.JW is False:
            for data in tqdm(crs_train_instructions):
                crs_train_instructions_addprompt.append(
                    f"{data}\n\nGuess which movie should be recommended to the user.")  # \nSystem: You should watch [BLANK]. Based on the conversation, guess the item for [BLANK]."
            for data in crs_valid_instructions:
                crs_valid_instructions_addprompt.append(
                    f"{data}\n\nGuess which movie should be recommended to the user.")  # \nSystem: You should watch [BLANK]. Based on the conversation, guess the item for [BLANK]."
            for data in tqdm(crs_test_instructions):
                crs_test_instructions_addprompt.append(
                    f"{data}\n\nGuess which movie should be recommended to the user.")  # \nSystem: You should watch [BLANK]. Based on the conversation, guess the item for [BLANK]."

            crs_train_instructions = crs_train_instructions_addprompt
            crs_valid_instructions = crs_valid_instructions_addprompt
            crs_test_instructions = crs_test_instructions_addprompt

    # Stage 에 따른 train, test 데이터셋 설정
    train_instructions = eval(f"{args.stage}_train_instructions")
    train_labels = eval(f"{args.stage}_train_labels")
    train_new = eval(f"{args.stage}_train_new")

    test_instructions = eval(f"{args.stage}_test_instructions")
    test_labels = eval(f"{args.stage}_test_labels")

    if args.review_merge is True:
        train_instructions.extend(review_train_instructions)
        train_labels.extend(review_train_labels)
        train_new.extend(review_train_new)
    if args.plot_merge is True:
        train_instructions.extend(plot_train_instructions)
        train_labels.extend(plot_train_labels)
        train_new.extend(plot_train_new)
    if args.quiz_merge is True:
        train_instructions.extend(quiz_train_instructions)
        train_labels.extend(quiz_train_labels)
        train_new.extend(quiz_train_new)
    if args.crs_merge is True:
        train_instructions.extend(crs_train_instructions)
        train_labels.extend(crs_train_labels)
        train_new.extend(crs_train_new)
    if args.pretrain_merge is True:
        train_instructions.extend(pretrain_train_instructions)
        train_labels.extend(pretrain_train_labels)
        train_new.extend(pretrain_train_new)

    logger.info('[Finish loading datasets]')
    logger.info(f'[Train Dataset Size: {len(train_instructions)}]')
    logger.info(f'[Test Dataset Size: {len(test_instructions)}]')

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
            if args.lora_weights[args.lora_weights.rfind('/') + 1:] != "lora-alpaca" \
                    and args.lora_weights[-1].isdigit() is False:
                origin_lora_weights = args.lora_weights
                for e in range(args.epoch):
                    args.lora_weights = origin_lora_weights + '_E' + str(int(e + 1))
                    evaluator.test(epoch=e + 1)
            elif 'train' in args.mode:
                for e in range(args.epoch):
                    args.lora_weights = os.path.join("./lora-alpaca", args.log_name + '_E' + str(int(e + 1)))
                    evaluator.test(epoch=e + 1)
            else:
                # default lora_weights (i.e., not-trained LLaMa)
                if args.lora_weights[args.lora_weights.rfind('/') + 1:] == "lora-alpaca":
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
