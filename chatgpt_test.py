#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import openai
import pickle
import time

from tqdm import tqdm

from utils.parser import parse_args



MODEL = "gpt-3.5-turbo"


def execute(args,
            instructions: list = None,
            labels: list = None):
    openai.api_key = args.chatgpt_key
    hit = args.chatgpt_hit
    cnt = args.chatgpt_cnt
    for instruction, label in tqdm(zip(instructions[cnt:], labels[cnt:]), bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request."},
                    {"role": "user", "content": instruction},
                ],
                temperature=0,
            )

            response = response['choices'][0]['message']['content']

            movie_name = label.replace('(', ')').split(')')[1].strip().lower()
            if 'example' in args.rq_num:
                check_response = response[response.lower().find("answer:"):].lower()
            else:
                check_response = response
            if movie_name in check_response.lower():
                hit += 1.0
            cnt += 1.0
            hit_ratio = hit / cnt
            args.log_file.write(json.dumps({'GEN': response, 'ANSWER': label, 'AVG_HIT': hit_ratio}, ensure_ascii=False) + '\n')

            if cnt % 100 == 0 and cnt != 0:
                print("%.2f" % (hit / cnt))
        except:
            print("ERROR hit: %d, cnt: %d" % (hit, cnt))
            print(args.log_file)
            args.chatgpt_hit = hit
            args.chatgpt_cnt = int(cnt)
            time.sleep(5)
            break

        openai.api_requestor._thread_context.session.close()
        if int(cnt) == len(instructions):
            return False


def chatgpt_test(args,
                 instructions: list = None,
                 labels: list = None
                 ):
    print('CHATGPT_TEST_START')
    while True:
        if execute(args=args, instructions=instructions, labels=labels) == False:
            break


if __name__ == "__main__":
    # fire.Fire(main)
    args = parse_args()
    chatgpt_test(args)
