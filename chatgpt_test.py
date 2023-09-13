#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import openai
import pickle
import time

from tqdm import tqdm

from utils.parser import parse_args

openai.api_key = ""

MODEL = "gpt-3.5-turbo"


def chatgpt_test(args,
                 instructions: list = None,
                 labels: list = None
                 ):
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
            if movie_name in response.lower():
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
            # os.system(f"python main.py --chatgpt_hit={hit} --chatgpt_cnt={cnt} --log_file={args.log_file}")
            # break
        openai.api_requestor._thread_context.session.close()
        if int(cnt) == len(instructions):
            return False


if __name__ == "__main__":
    # fire.Fire(main)
    args = parse_args()
    chatgpt_test(args)
