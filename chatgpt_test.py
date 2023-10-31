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

template_cot_cand = """
I will give you dialogs between an user and you (a recommender system).
Based on the context of each dialog, guess step-by-step which movie should be recommended to the user.

Dialog 1. 
System: hello User: Hello , I am looking for some movie recommendations . Do you have any Horror Movies that you could suggest for me ? System: I do ! have you ever seen The Conjuring (2013) ? User: I have seen The Conjuring (2013) I very much enjoyed this movie . When I was younger I liked movies like Friday the 13th (1980) . System: I liked that one too ! User: I also liked to watch the A Nightmare on Elm Street (1984) Series . System: I loved that as well mainly because Johnny Depp is in it User: I love Johnny Depp ! So handsome , such a great actor . I love johnny Depp in Alice in Wonderland (2010) .

Answer 1.
- The user is looking for horror movie recommendations.
- The user has mentioned enjoying movies like The Conjuring (2013), Friday the 13th (1980), and A Nightmare on Elm Street (1984) series.
- The user also mentioned liking Johnny Depp and specifically mentioned enjoying his performance in Alice in Wonderland (2010).
- Based on the user's preferences for horror movies and their appreciation for Johnny Depp, a recommended movie could be a horror film that features Johnny Depp in a prominent role. 
Therefore, The Astronaut's Wife (1999) should be recommended since Johnny Depp stars in this sci-fi horror film, and it aligns with the user's interest in horror movies and their appreciation for Johnny Depp's acting.

Dialog 2.
%s

Answer 2.
"""

template = """
Pretend you are a movie recommender system. I will give you a conversation between a user and you (a recommender system).

Based on the conversation, guess which movie should be recommended to the user.
Do not provide any extra sentences.

Here is the conversation:
%s
"""

template_analyze_preference = """
I will give you dialogs between an user and you (a recommender system).
Based on the context of each dialog, guess step-by-step the preference of the user.

Dialog 1.
System: hello 
User: hi how are you 
System: good what kind of movies do you like?
User: I like all types of movies but today I'm in the mood for something scary I love ghost movies like Paranormal Activity (2007) 
System: It (2017) is a great movie choice 
User: I've watched It (2017) I think it was really good, but I love the original It (1990) 
System: oh ok 
User: I'm definitely not into movies with a lot of blood like Saw (2004). Those make me want to throw up.

Answer 1.
- The user is in the mood for a scary movie and specifically mentions enjoying ghost movies like Paranormal Activity (2007).
- The system recommends It (2017), but the user mentions already having watched it and preferring the original It (1990).
- The user expresses a dislike for movies with a lot of blood, specifically mentioning Saw (2004).
- Based on the user's preference for ghost movies and their dislike for movies with excessive blood, a recommended movie could be a ghost movie that is not overly gory.
Therefore, the user prefers ghost movies and dislikes for movies with excessive blood.

Dialog 2.
User: Hello 
System: Hi. I heard you are interested in a movie. What type of movies do you like? 
User: I am looking for recommendations for older 80 's Comedies like The Breakfast Club (1985) 
System: How about Pretty in Pink (1986) or Sixteen Candles (1984) Another good one is Some Kind of Wonderful (1987) 
User: I think I have seen Pretty in Pink (1986) but it has been a while. I haven't seen Sixteen Candles (1984) 
System: It is a classic to be sure 
User: I'm not familiar with Some Kind of Wonderful (1987) Who is in that one? 
System: I also really liked Lucas (1986) staring Charlie Sheen 
User: I Liked Lucas (1986) when I was a kid. It's been a while since I have seen that one.

Answer 2.
- The user is looking for recommendations for older 80's comedies.
- The user has mentioned enjoying movies like The Breakfast Club (1985) and Pretty in Pink (1986).
- The user has not seen Sixteen Candles (1984) and is not familiar with Some Kind of Wonderful (1987).
- The user also mentioned liking Lucas (1986) when they were a kid.
Therefore, the user prefers 80's comedies, and the actor Charlie Sheen.

Dialog 3.
%s

Answer 3.

"""

def execute(args,
            instructions: list = None,
            labels: list = None):
    openai.api_key = args.chatgpt_key
    hit = args.chatgpt_hit
    cnt = args.chatgpt_cnt
    for instruction, label in tqdm(zip(instructions[cnt:], labels[cnt:]),
                                   bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):

        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "user",
                     "content": template_analyze_preference % instruction.replace('User:', '\nUser:').replace('System:', '\nSystem:')}
                ],
                temperature=0,
            )

            response = response['choices'][0]['message']['content']
            if 'quiz' in args.stage:
                movie_name = label.replace('(', ')').split(')')[1].strip().lower()
            elif 'crs' in args.stage:
                check_response = response.lower()
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
                 ):
    print('CHATGPT_TEST_START')
    while True:
        if execute(args=args, instructions=instructions, labels=labels) == False:
            break


if __name__ == "__main__":
    # fire.Fire(main)
    args = parse_args()
    chatgpt_test(args)
