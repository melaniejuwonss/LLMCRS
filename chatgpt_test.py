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

Based on the conversation, you reply me with 1 recommendation without extra sentences.

Here is the conversation:
%s
"""

template_analyze_preference = """
I will give you dialogs between an user and you (a recommender system).
Based on the context of each dialog, guess step-by-step the preference of the user.

Dialog 1.
System: Hello. How can I help you?
User: Hello. I'm looking for a great documentary do you have any suggestions?

Answer 1.
- The user is specifically looking for a documentary.
- Based on the user's preference, a recommended documentary could be one that is critically acclaimed or popular in general, as there are no specific preferences indicated yet.

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
- Based on the user's preference, a recommended movie could be 80's comedies that stars Charlie Sheen.

Dialog 3.
User: Hi.
System: Hi. What kind of movie are you looking for today?
User: Some movie similar to Harry Potter?
System: I love Harry Potter and the Goblet of Fire (2005). Have you seen them all?
User: Yeah.
System: Fantastic Beasts and Where to Find Them (2016) is great! It goes along with the story.
User: Recommend something similar I have read all the books.

Answer 3.
- The user is looking for a movie similar to the Harry Potter series.
- The user has seen all the Harry Potter movies and read all the books, including related works like Fantastic Beasts and Where to Find Them.
- Given the user's familiarity with the Harry Potter universe, they are looking for similar fantasy movies but outside the Harry Potter series.
- Based on the user's preference, a recommended movie could be within the fantasy genre with a similar magical or school setting, but not directly connected to the Harry Potter or Fantastic Beasts series.

Dialog 4. %s

Answer 4.
"""

template_user_intention = "What is the user's intention of the last utterance in terms of the context of the dialog: Please tell me only one short sentence. Starts with 'The user's intention in the last utterance is'\n %s"

template_dialog_generation = """I will give you a review of movie %s.
%s

I will give you a example dialog.
User: Hey, I'm in the mood for a heartwarming and meaningful movie. Could you recommend a movie?
System: Have you seen Forrest Gump (1994)? It's a classic that has touched the hearts of many. It's a beautiful story of a man's extraordinary journey through life. Some say it's like a box of chocolates—you never know what you're gonna get.
User: It sounds interesting! What's so special about it?
System: Well, Forrest Gump (1994) is one of those movies that can have a different impact on you depending on where you are in life when you watch it. It's more than just a story about a man with a low IQ; it’s also a tale of resilience, innocence, and the profound impact one person can have on the lives of many. It teaches you to appreciate the little things in life and to keep moving forward no matter what challenges come your way.
User: That sounds like a powerful message. I'll give it a try.
System: You're welcome! I hope you find a deep connection with the movie. It's known to touch the hearts of many and leave them with a sense of inspiration and appreciation for the beauty of life. Enjoy your movie night!

Can you make a dialogue (within 3-turns) for talking about the movie %s by considering a review and sample dialogue?
"""


# - Therefore, Blair Witch (2016) should be recommended.
# - Therefore, Major League (1989) should be recommended.
# - Therefore, The Lord of the Rings: The Fellowship of the Ring (2001) should be recommended.
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
                     "content": template_dialog_generation % (label, instruction, label)}
                ],
                temperature=0,
            )

            response = response['choices'][0]['message']['content']
            # response += "\n -Therefore, %s should be recommended." % (label)
            if 'quiz' in args.stage:
                movie_name = label.replace('(', ')').split(')')[1].strip().lower()
            elif 'crs' in args.stage:
                check_response = response.lower()
                movie_name = label.split('(')[0].strip().lower()

            # if 'example' in args.rq_num:
            #     check_response = response[response.lower().find("answer:"):].lower()
            # else:
            #     check_response = response
            # if movie_name in check_response.lower():
            #     hit += 1.0
            cnt += 1.0
            # hit_ratio = hit / cnt

            args.log_file.write(
                json.dumps({'INPUT': template_dialog_generation % (label, instruction, label), 'OUTPUT': response}, ensure_ascii=False,
                           indent=4) + '\n')

            # if cnt % 100 == 0 and cnt != 0:
            #     print("%.2f" % (hit / cnt))
        except:
            # print("ERROR hit: %d, cnt: %d" % (hit, cnt))
            print("ERROR cnt: %d" % (cnt))
            print(args.log_file)
            # args.chatgpt_hit = hit
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
