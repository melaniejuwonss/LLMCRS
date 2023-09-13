#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import openai
import pickle
import time

from tqdm import tqdm

from utils.parser import parse_args

openai.api_key = "---"

MODEL = "gpt-3.5-turbo"

USER_INPUT_MSG = """
다음 사용자와 시스템간 대화를 영어로 번역:

1. 사용자: 겨울에 어울리는 영화를 찾아보고 싶은데 어떤 영화가 좋을까?			
2. 시스템: 어떤 장르의 영화를 좋아하시나요?			
3. 사용자: 장르는 상관없고 뭔가 좀 아포칼립스적인 소재를 원해.			
4. 시스템: 그렇다면 영화 설국열차를 추천할게요[토픽]: “설국열차(2013년 08월 01일_개봉작)”
5. 시스템: 얼어붙은 지구의 인류의 마지막 생존 지역인 설국열차를 다루고 있어요.[토픽]: “설국열차(2013년 08월 01일_개봉작)”][지식]:	“코멘트: 얼어붙은 지구, 인류 마지막 생존지역 `설국열차`가 질주한다.”
6. 사용자: 설국열차? 엄청 생소한데 감독이 뭔가 특이한 걸 다뤘네.[토픽]“설국열차(2013년 08월 01일_개봉작)”		
7. 시스템: 봉준호 감독이 10년 동안 키워왔던 상상력을 이 영화에서 분출한 거죠.[토픽]“설국열차(2013년 08월 01일_개봉작)”[지식]“코멘트: 10년 전부터 키워 온 봉준호 감독의 상상력, 마침내 <설국열차>로 태어나다!”	
8. 사용자: 봉준호 감독 영화였으면 주연 배우도 대단한 사람들이었겠어. 누가 나오지?[토픽]: “설국열차(2013년 08월 01일_개봉작)”		
9. 시스템: 영화 설국열차는 송강호 애드 해리스 옥타비아 스펜서 이완 브렘너 제이미 벨 존 허트 크리스 에반스 틸다 스윈튼이 출연합니다.[토픽]: “설국열차(2013년 08월 01일_개봉작)”[지식]: “배우: 송강호 애드 해리스 옥타비아 스펜서 이완 브렘너 제이미 벨 존 허트 크리스 에반스 틸다 스윈튼”	
10. 시스템: 아카데미와 골든 글로브 수상자부터 할리우드 히어로까지 국적이 제각각이지만 연기파 배우들이 이 영화를 통해 앙상블을 보여줍니다.[토픽]: “설국열차(2013년 08월 01일_개봉작)”[지식]: “코멘트: 아카데미와 골든 글로브 수상자부터 할리우드 히어로까지! 국적도, 개성도 제각각! 연기파 배우들의 앙상블 캐스트, 질주에 밀도를 더하다!”	
11. 사용자: 재밌겠는데. 봉준호 감독 작품이라 더 볼만할 것 같아.[토픽]: “설국열차(2013년 08월 01일_개봉작)”		
12. 시스템: 설국열차를 통해 한국 영화의 상상력의 경계를 넓힌 봉준호 감독의 새로운 세상을 만나보세요.[토픽]: “설국열차(2013년 08월 01일_개봉작)”[지식]: “코멘트: 한국 영화, 상상력의 경계를 넓히다! <살인의 추억> <괴물> <마더> 봉준호 감독의 새로운 세계 <설국열차>”

"""


# response = openai.ChatCompletion.create(
#             model=MODEL,
#             messages=[
#                 # {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": USER_INPUT_MSG},
#                 # {"role": "assistant", "content": "영화 아저씨의 영어이름은 AJJUSSI이다"},
#             ],
#             temperature=0,
#         )

def chatgpt_test(args,
                 instructions: list = None,
                 labels: list = None
                 ):
    hit = 0.0
    cnt = args.chatgpt_cnt
    for instruction, label in tqdm(zip(instructions[cnt:], labels[cnt:])):
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
            print("ERROR %d" % cnt)
            print(args.log_file)
            break


if __name__ == "__main__":
    # fire.Fire(main)
    args = parse_args()
    chatgpt_test(args)
