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
from utils.data import quiz_read_data, plot_read_data, meta_plot_review_read_data, synthetic_dialog_read_data
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

    args.wandb_project = "LLMCRS"
    args.wandb_run_name = args.log_name
    wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    ROOT_PATH = dirname(realpath(__file__))
    DATASET_PATH = os.path.join(ROOT_PATH, args.dataset_path)
    args.dataset_path = DATASET_PATH

    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)

    quiz_train_data = quiz_read_data(args, 'train')
    quiz_test_data = quiz_read_data(args, 'test')

    plot_train_data = plot_read_data(args, 'train')
    meta_plot_review_data = meta_plot_review_read_data(args, 'train')

    if args.stage.lower() == 'pretrain':
        train_data = []
        synthetic_dialog = synthetic_dialog_read_data(args, 'train')
        tokenized_synthetic_dialog = [tokenizer(data['OUTPUT']).input_ids[1:] for data in tqdm(synthetic_dialog)]

        for data in tqdm(tokenized_synthetic_dialog):
            if len(data) < args.cutoff:
                train_data.append(tokenizer.decode(data))
            else:
                remain_len = len(data) - args.cutoff
                for i in range(0, remain_len + 1, 10):
                    train_data.append(tokenizer.decode(data[i:i + args.cutoff]))

        # for data in meta_plot_review_data:
        #     meta = data['meta']
        #     plot = data['plot']
        #     review = data['review']
        #     title = data['item']
        #     if review == '':
        #         continue
        #     # if review != '' and plot != '':
        #     #     context_tokens = f"""I will give you information about a moive {title}.\nPlease read carefully and memorize all information.\n\nI will give you meta information of the movie {title}:\n{meta}\n\nI will give you a plot of the movie {title}:\n{plot}\n\nI will give you a review of the movie {title}:\n{review}"""
        #     # elif review != '':
        #     #     context_tokens = f"""I will give you information about a moive {title}.\nPlease read carefully and memorize all information.\n\nI will give you meta information of the movie {title}:\n{meta}\n\nI will give you a review of the movie {title}:\n{review}"""
        #     # elif plot != '':
        #     #     context_tokens = f"""I will give you information about a moive {title}.\nMeta information of the movie {title}:\n{meta}\nPlot of the movie {title}:\n{plot}"""
        #     # context_tokens = plot
        #     if args.TH:
        #         review = review.replace(title, '[title]')
        #         name = title.split('(')[0].strip()
        #         year = title.split('(')[-1][:-1].strip()
        #         if not year.isdigit():
        #             year = ''
        #         review = review.replace(f"\"{name}\" ({year})", '[title]')
        #         review = review.replace(f"\"{name.lower()}\" ({year})", '[title]')
        #         review = review.replace(name, '[title]')
        #         review = review.replace(name.lower(), '[title]')
        #         review = review.replace(f"({year})", '')
        #         context_tokens = f"""A review of the movie [title]:\n{review}"""
        #     else:
        #         context_tokens = f"""A review of the movie {title}:\n{review}"""
        #
        #     train_data.append({'context_tokens': context_tokens, 'item': '', 'isNew': True})
        # # test_data = train_data[:100]
        # for data in tqdm(train_data):
        #     data['context_tokens'] = tokenizer.decode(tokenizer(data['context_tokens']).input_ids)[1:][:args.cutoff]

        train_instructions = [i['context_tokens'] for i in train_data]
        train_labels = [i['item'] for i in train_data]
        test_instructions = train_instructions[:100]
        test_labels = train_labels[:100]
        train_new = [True for i in train_data]


    elif args.stage.lower() == "crs":
        crs_dataset = CRSDatasetRec(args)
        # if "cot" not in args.data_type:
        train_data = crs_dataset.train_data
        valid_data = crs_dataset.valid_data
        test_data = crs_dataset.test_data
        cnt = 0
        for data in tqdm(train_data):
            context_tokens = tokenizer.decode((tokenizer(data['context_tokens']).input_ids)[1:][-args.cutoff:])
            data['context_tokens'] = f"{context_tokens}\n\nGuess which movie should be recommended to the user."  # \nSystem: You should watch [BLANK]. Based on the conversation, guess the item for [BLANK]."
        for data in valid_data:
            data['context_tokens'] = f"{context_tokens}\n\nGuess which movie should be recommended to the user."  # \nSystem: You should watch [BLANK]. Based on the conversation, guess the item for [BLANK]."
        for data in tqdm(test_data):
            context_tokens = tokenizer.decode((tokenizer(data['context_tokens']).input_ids)[1:][-args.cutoff:])
            data['context_tokens'] = f"{context_tokens}\n\nGuess which movie should be recommended to the user."  # \nSystem: You should watch [BLANK]. Based on the conversation, guess the item for [BLANK]."

        new_idx = json.load(open(os.path.join(args.dataset_path, 'train_new_idx.json'), 'r', encoding='utf-8'))

        target = 'item'
        if args.train_response:  # 구분하기 위한 코드 (item으로 학습할 지? response?)
            target = 'response'
        train_data = [{'context_tokens': data['context_tokens'], 'item': data[target], 'isNew': idx in new_idx} for
                      idx, data in enumerate(train_data)]

        if 'synthetic' in args.data_type or args.all_merge:
            syn_data_path = os.path.join(DATASET_PATH, 'synthetic')
            if not os.path.exists(syn_data_path): os.mkdir(syn_data_path)

            temp = args.data_type
            if args.all_merge:
                args.data_type = 'synthetic_dialog_review'
            with open(os.path.join(syn_data_path, f'{args.data_type}.json'), 'r', encoding='utf-8') as f:
                syn_train_data = json.load(f)
            args.data_type = temp

            target_item_list = [data['INPUT'].split('I will give you a review of movie')[1].split('\n')[0].strip() for
                                data in syn_train_data]
            for idx, data in enumerate(syn_train_data):
                target_item_list[idx] = target_item_list[idx].replace('.', '')
                data['OUTPUT'] = data['OUTPUT'].replace(target_item_list[idx], '[BLANK]')
                title = target_item_list[idx].split('(')[0].strip()
                year = target_item_list[idx].split('(')[-1][:-1].strip()
                if not year.isdigit():
                    year = ''
                data['OUTPUT'] = data['OUTPUT'].replace(f"\"{title}\" ({year})", '[BLANK]')
                data['OUTPUT'] = data['OUTPUT'].replace(f"\"{title.lower()}\" ({year})", '[BLANK]')
                data['OUTPUT'] = data['OUTPUT'].replace(title, '[BLANK]')
                data['OUTPUT'] = data['OUTPUT'].replace(title.lower(), '[BLANK]')
                data['OUTPUT'] = data['OUTPUT'].replace(f"({year})", '')

                data['OUTPUT'] = f"{data['OUTPUT']}\n\n Based on the conversation, guess the item for [BLANK]."
            syn_train_data = [{'context_tokens': data['OUTPUT'], 'item': target_item_list[idx], 'isNew': True} for
                              idx, data in
                              enumerate(syn_train_data)]

            # test_data.extend(syn_train_data[:20])
            if args.merge:
                train_data.extend(syn_train_data)
            else:
                train_data = syn_train_data
                test_data = syn_train_data[:50]

        elif 'onlyReview' in args.data_type:
            review_data_path = os.path.join(DATASET_PATH, 'review')
            if not os.path.exists(review_data_path): os.mkdir(review_data_path)
            if args.merge is True:
                crs_test_data = test_data
                crs_train_data = train_data
                for crs_train in crs_train_data:
                    context_tokens = "Pretend you are a movie recommender system. I will give you a dialogue between a user and you (a recommender system). \n\nHere is the dialogue: \n" + \
                                     crs_train['context_tokens']
                    crs_train['context_tokens'] = context_tokens
                for crs_test in crs_test_data:
                    context_tokens = "Pretend you are a movie recommender system. I will give you a dialogue between a user and you (a recommender system). \n\nHere is the dialogue: \n" + \
                                     crs_test['context_tokens']
                    crs_test['context_tokens'] = context_tokens

            with open(os.path.join(review_data_path, f'{args.data_type}.json'), 'r', encoding='utf-8') as f:
                train_data = json.load(f)

            review_template = """I will give you a review of a movie.\nIn the review, the movie title is masked with %s.\nHere is the review:\n%s\n\nBased on the review, guess the movie title for [title] without extra explanations."""
            if args.TH:
                review_template = """I will give you a review of a movie.\nIn the review, the movie title is masked with %s.\nHere is the review:\n%s\n\nBased on the review, guess the movie title that the above review is discussing"""
            elif args.JW:
                review_template = """I will give you a review of a movie\nIn this review, the movie title is maksed with %s.\nAfter reading the review, guess the movie title for [title] by considering actor, genre, director, writer, and plot discussed in the review.\nHere is the review:\n%s"""
            origin_train_data = [
                {'context_tokens': review_template % (data['item'], data['context_tokens']), 'item': data['item'],
                 'isNew': True} for data in train_data]

            if args.pretrain:
                review_template = """I will give you a review of a movie %s\n%s"""
                origin_train_data = [
                    {'context_tokens': (data['item'], data['context_tokens']), 'item': '', 'isNew': True} for data in
                    train_data]

            target_item_list = [data['item'] for data in train_data]

            for idx, data in enumerate(train_data):
                data['context_tokens'] = data['context_tokens'].replace(target_item_list[idx], '[title]')
                title = target_item_list[idx].split('(')[0].strip()
                year = target_item_list[idx].split('(')[-1][:-1].strip()
                if not year.isdigit():
                    year = ''
                data['context_tokens'] = data['context_tokens'].replace(f"\"{title}\" ({year})", '[title]')
                data['context_tokens'] = data['context_tokens'].replace(f"\"{title.lower()}\" ({year})", '[title]')
                data['context_tokens'] = data['context_tokens'].replace(title, '[title]')
                data['context_tokens'] = data['context_tokens'].replace(title.lower(), '[title]')
                data['context_tokens'] = data['context_tokens'].replace(f"({year})", '')

                data['context_tokens'] = review_template % ('[title]', data['context_tokens'])
            train_data = [{'context_tokens': data['context_tokens'], 'item': target_item_list[idx], 'isNew': True} for
                          idx, data in enumerate(train_data)]
            for data in tqdm(train_data):
                data['context_tokens'] = tokenizer.decode(tokenizer(data['context_tokens']).input_ids[1:][:args.cutoff])

            with open(os.path.join(review_data_path, f'onlyReview_1.json'), 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            target_item_list_test = [data['item'] for data in test_data]

            for idx, data in enumerate(test_data):
                data['context_tokens'] = data['context_tokens'].replace(target_item_list[idx], '[title]')
                title = target_item_list[idx].split('(')[0].strip()
                year = target_item_list[idx].split('(')[-1][:-1].strip()
                if not year.isdigit():
                    year = ''
                data['context_tokens'] = data['context_tokens'].replace(f"\"{title}\" ({year})", '[title]')
                data['context_tokens'] = data['context_tokens'].replace(f"\"{title.lower()}\" ({year})", '[title]')
                data['context_tokens'] = data['context_tokens'].replace(title, '[title]')
                data['context_tokens'] = data['context_tokens'].replace(title.lower(), '[title]')
                data['context_tokens'] = data['context_tokens'].replace(f"({year})", '')

                data['context_tokens'] = review_template % ('[title]', data['context_tokens'])
            test_data = [{'context_tokens': data['context_tokens'], 'item': target_item_list_test[idx], 'isNew': True}
                         for idx, data in enumerate(test_data)]
            test_data = test_data[:300]
            for data in tqdm(test_data):
                data['context_tokens'] = tokenizer.decode(tokenizer(data['context_tokens']).input_ids[1:][:args.cutoff])

            if args.pretrain:
                train_data = origin_train_data

            if args.merge is True:
                train_data.extend(crs_train_data)
                test_data = crs_test_data

            if args.quiz_merge:
                train_data.extend(
                    [{'context_tokens': data[0], 'item': data[1], 'isNew': True} for data in quiz_train_data])

            if args.plot_merge:
                # train_data.extend(origin_train_data)
                if not args.pretrain:
                    plot_template = "I will give you a plot of a movie\nHere is the plot:\n%s\n\nGuess the movie title that the above plot is describing"
                    train_data.extend(
                        [{'context_tokens': plot_template % data['context_tokens'], 'item': data['item'], 'isNew': True}
                         for data in plot_train_data])
                else:
                    plot_template = "I will give you a plot of a movie %s\n%s"
                    train_data.extend(
                        [{'context_tokens': (data['item'], data['context_tokens']), 'item': '', 'isNew': True} for data
                         in plot_train_data])

                # train_data = origin_train_data

            logger.info('[Finish loading onlyReview datasets]')
            logger.info(f'[onlyReview Train Dataset Size: {len(train_data)}]')
            logger.info(f'[onlyReview Test Dataset Size: {len(test_data)}]')


        elif "cot" in args.data_type:
            cot_data_path = os.path.join(DATASET_PATH, 'cot')
            if 'train' in args.mode:
                with open(os.path.join(cot_data_path, f'train_data_{args.data_type}.json'), 'r', encoding='utf-8') as f:
                    train_data = json.load(f)
            if 'test' in args.mode:
                with open(os.path.join(cot_data_path, f'test_data_{args.data_type}.json'), 'r', encoding='utf-8') as f:
                    test_data = json.load(f)
            if args.oversample_ratio > 1:
                train_new_idx = json.load(open(f'data/redial/train_new_idx.json', 'r', encoding='utf-8'))
                train_old_idx = list(range(len(train_data)))
                train_old_idx = [item for item in train_old_idx if item not in train_new_idx]
                rec_train_data = [train_data[x] for x in train_new_idx]
                chat_train_data = [train_data[x] for x in train_old_idx]
                # oversample_ratio = int(len(rec_train_data) / len(chat_train_data))
                chat_train_data = chat_train_data * args.oversample_ratio
                train_data = rec_train_data + chat_train_data

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

    elif args.stage.lower() == 'plot':
        if args.pretrain:
            plot_template = "I will give you a plot of a movie %s\n%s"
            plot_train_data = [{'context_tokens': (data['item'], data['context_tokens']), 'item': '', 'isNew': True} for
                               data in plot_train_data]
        else:
            plot_train_data = [{'context_tokens': data['context_tokens'], 'item': data['item'], 'isNew': True} for data
                               in plot_train_data]

        train_instructions = [i['context_tokens'] for i in plot_train_data]
        train_labels = [i['item'] for i in plot_train_data]
        test_instructions = train_instructions
        test_labels = train_labels
        train_new = [True for i in plot_train_data]

    elif args.stage.lower() == "review_dialog":
        review_dataset = ReviewDataset(args)
        train_instructions = [i['context_tokens'] for i in review_dataset.return_data]
        train_labels = [i['item'] for i in review_dataset.return_data]

    elif args.stage.lower() == "quiz":
        train_instructions = [i[0] for i in quiz_train_data]
        train_labels = [i[1] for i in quiz_train_data]
        test_instructions = [i[0] for i in quiz_test_data]
        test_labels = [i[1] for i in quiz_test_data]
        train_new = [True for i in quiz_train_data]

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
