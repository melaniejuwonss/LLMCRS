#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import openai
import pickle
import time
import random

from tqdm import tqdm

from utils.parser import parse_args

MODEL = "gpt-3.5-turbo"

template = """
I will give you dialogs between an user and you (a recommender system) and candidate items.
Based on the context of each dialog, guess step-by-step which movie should be recommended to the user among the items in candidate items.

Dialog 1. 
System: hello User: Hello , I am looking for some movie recommendations . Do you have any Horror Movies that you could suggest for me ? System: I do ! have you ever seen The Conjuring (2013) ? User: I have seen The Conjuring (2013) I very much enjoyed this movie . When I was younger I liked movies like Friday the 13th (1980) . System: I liked that one too ! User: I also liked to watch the A Nightmare on Elm Street (1984) Series . System: I loved that as well mainly because Johnny Depp is in it User: I love Johnny Depp ! So handsome , such a great actor . I love johnny Depp in Alice in Wonderland (2010) .

Candidate items 1.
Stalingrad  (2013), Grey Gardens (1975), Forget Paris (1995), 12 Monkeys (1995), Hunter  (2015), Strange Circus (2005), Garfield: The Movie (2004), Chicken Little  (2005), The Girl on the Train  (2009), What's Eating Gilbert Grape (1993), The Hot Chick (2002), Snowpiercer (2013), Money  (1993), Poltergeist III (1988), The Puppetmaster  (1993), Écoute voir (1979), The Duchess  (2008), Driven  (2016), Miss You Already (2015), The Truth About Cats & Dogs (1996), The Cake Eaters (2007), Inferno  (2016), Red Firecracker, Green Firecracker (1994), PT 109  (1963), Jiro Dreams of Sushi (2011), H  (2002), The Warrior  (2001), Jack Frost  (1998), Archie: To Riverdale and Back Again (1990), Anabelle Acosta, The Astronaut's Wife (1999), Day of the Dead  (2008), The Phantom of the Opera  (1925), The Texas Chainsaw Massacre 2 (1986), Grimm  (2003), Where the Heart Is  (2000), Blood Diamond  (2006), The Arrow (1996), What Women Want (2000), Captain Blood  (1960), The Wind in the Willows  (1987)

Answer 1.
- The user is looking for horror movie recommendations.
- The user has mentioned enjoying movies like The Conjuring (2013), Friday the 13th (1980), and A Nightmare on Elm Street (1984) series.
- The user also mentioned liking Johnny Depp and specifically mentioned enjoying his performance in Alice in Wonderland (2010).
- Based on the user's preferences for horror movies and their appreciation for Johnny Depp, a recommended movie could be a horror film that features Johnny Depp in a prominent role. 
Therefore, The Astronaut's Wife (1999) should be recommended.

Dialog 2.
%s

Candidate items 2.
%s

Answer 2.
"""


def execute(args,
            instructions: list = None,
            labels: list = None,
            negItems: list = None):
    openai.api_key = args.chatgpt_key
    hit = args.chatgpt_hit
    cnt = args.chatgpt_cnt
    for instruction, label, negItem in tqdm(zip(instructions[cnt:], labels[cnt:], negItems[cnt:]),
                                   bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):

        candidate_item = negItem + [label]
        random.shuffle(candidate_item)
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "user",
                     "content": template % (instruction, ', '.join(candidate_item))}
                ],
                temperature=0,
            )

            response = response['choices'][0]['message']['content']
            if 'quiz' in args.stage:
                movie_name = label.replace('(', ')').split(')')[1].strip().lower()
            elif 'crs' in args.stage:
                check_response = response[response.lower().rfind("\n"):].lower()
                movie_name = label.split('(')[0].strip().lower()

            # if 'example' in args.rq_num:
            #     check_response = response[response.lower().find("answer:"):].lower()
            # else:
            #     check_response = response
            if movie_name in check_response.lower():
                hit += 1.0
            cnt += 1.0
            hit_ratio = hit / cnt

            args.log_file.write(
                json.dumps(
                    {'DIALOG': instruction, 'LABEL': label, 'CHECK_RESPONSE': check_response, 'RESPONSE': response,
                     'AVG_HIT': hit_ratio, 'NEW_ITEM': label.lower() not in instruction.lower()},
                    ensure_ascii=False,
                    indent=4) + '\n')

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
                 labels: list = None,
                 negItems: list = None
                 ):
    print('CHATGPT_TEST_START')
    while True:
        if execute(args=args, instructions=instructions, labels=labels, negItems=negItems) == False:
            break


if __name__ == "__main__":
    # fire.Fire(main)
    args = parse_args()
    chatgpt_test(args)
