#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import openai
import pickle
import time

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
response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You will be provided with a dialog in Korean, and your task is to translate it into English."},
        {"role": "user", "content": USER_INPUT_MSG},
    ],
    temperature=0,
)

traslated = response['choices'][0]['message']['content']

# USER_INPUT_MSG= """
# 다음 json 은 사용자와 시스템간 대화와 그것과 관련된 정보를 갖고있다.
# json 형태를 유지하면서, 이 안의 한국말들을 모두 영어로 바꿔달라.
#
# {"goal": "", "user_profile": {}, "conversation": ["얼마 전에 캡틴 마블을 봤는데 강한 여성 캐릭터가 나와서 너무 좋았어.", "캡틴 마블은 마블의 새로운 히어로이면서 어벤져스의 희망이라 불리우는 히어로이지요.", "여성 히어로가 나오는 다른 영화 좀 추천해 줄래?", "그렇다면 마블의 새로운 여성 히어로인 와스프가 등장하는 영화 앤트맨과 와스프는 어떠세요?", "이 영화는 사람들의 평이 어떻지? 평점이 높은가?", "앤트맨과 와스프의 평점은 4.3점입니다.", "할리우드 배우들 중에 내가 알만한 배우들이 출연했어?", "로렌스 피시번 마이클 더글라스 미셸 파이퍼 에반젤린 릴리 폴 러드 해나 존 케이먼 등이 출연하는데요.", "이들은 영화 속에 등장하는 격투신과 스턴트 액션을 직접 소화했어요.", "이야. 대단한데. 그럼 다른 히어로 무비랑 차별되는 스토리가 있어?", "이 영화는 앤트맨과 와스프 앞에 정체불명의 빌런 고스트가 등장하며 펼쳐지는 이야기를 담고 있는데요.", "영화를 보는 내내 마블 시네마틱 유니버스의 세계관을 넓힌 양자 영역의 세계로 빠져드시게 될 거예요."], "situation": "", "goal_type_list": ["Chit-chat", "Chit-chat", "Movie Recommendation", "Movie Recommendation", "QA", "QA", "QA", "QA", "QA", "QA", "QA", "QA"], "goal_topic_list": ["캡틴 마블(2019년 03월 06일_개봉작)", "캡틴 마블(2019년 03월 06일_개봉작)", " ", "앤트맨과 와스프(2018년 07월 04일_개봉작)", "앤트맨과 와스프(2018년 07월 04일_개봉작)", "앤트맨과 와스프(2018년 07월 04일_개봉작)", "앤트맨과 와스프(2018년 07월 04일_개봉작)", "앤트맨과 와스프(2018년 07월 04일_개봉작)", "앤트맨과 와스프(2018년 07월 04일_개봉작)", "앤트맨과 와스프(2018년 07월 04일_개봉작)", "앤트맨과 와스프(2018년 07월 04일_개봉작)", "앤트맨과 와스프(2018년 07월 04일_개봉작)"], "role": ["user", "system", "user", "system", "user", "system", "user", "system", "system", "user", "system", "system"], "knowledge": ["", "캡틴 마블(2019년 03월 06일_개봉작) 코멘트: 마블의 새로운 히어로, 어벤져스의 희망! ", "", "앤트맨과 와스프(2018년 07월 04일_개봉작) 코멘트: 마블의 새로운 여성 히어로 ‘와스프’", "", "앤트맨과 와스프(2018년 07월 04일_개봉작) 평점: 4.3점", "", "앤트맨과 와스프(2018년 07월 04일_개봉작) 배우: 로렌스 피시번 마이클 더글라스 미셸 파이퍼 에반젤린 릴리 폴 러드 해나 존 케이먼", "앤트맨과 와스프(2018년 07월 04일_개봉작) 코멘트: 배우들이 직접 소화한 격투 & 스턴트 액션", "", "앤트맨과 와스프(2018년 07월 04일_개봉작) 코멘트: 시빌워 사건 이후 은둔하며 히어로와 가장의 역할 사이에서 고민 중이던 앤트맨과 와스프 앞에 정체불명의 빌런 고스트가 등장한다. 고스트를 쫓던 앤트맨과 와스프는 상상도 못했던 상황에 직면한다.", "앤트맨과 와스프(2018년 07월 04일_개봉작) 코멘트: <어벤져스4>의 결정적 힌트 ‘양자 영역’ 마블 시네마틱 유니버스의 세계관을 넓힌다"]}
#
# """\
english_dialog_list = []
cnt = 217
for dialog in (dialog_list[cnt:500]):
    print("****************%d**********************" % cnt)
    try:
        USER_INPUT_MSG = "다음 사용자와 시스템간 대화를 영어로 번역:\n" + ''.join(dialog)
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": USER_INPUT_MSG},
                # {"role": "assistant", "content": "영화 아저씨의 영어이름은 AJJUSSI이다"},
            ],
            temperature=0,
        )
        traslated = response['choices'][0]['message']['content']
        print(traslated)
        traslated_list = traslated.split('\n')
        traslated_list_by_topic_knowledge = [data.replace('[Topic]:', '--').replace('[Knowledge]:', '--').split('--') for data in traslated_list]
        final_list = []
        for idx, data in enumerate(traslated_list_by_topic_knowledge):
            uid = data[0][:data[0].find('.')]
            data[0] = data[0][data[0].find('.') + 1:]
            for i in range(3 - len(data)):
                data.append('')
            data.insert(0, uid)
            final_list.append(data)
        english_dialog_list.append([cnt, final_list])
        cnt += 1
        # time.sleep(60)
    except:
        print("ERROR %d" % cnt)
        break

# print('done')
with open('KT_eng_dialog_217-.pkl', 'wb') as f:
    pickle.dump(english_dialog_list, f)

# print(response)
import re

print()
print('done')

# !/usr/bin/env python
# -*- coding: utf-8 -*-
for dialog in (all_printed[cnt:884]):
    print("****************%d**********************" % cnt)
    try:
        # USER_INPUT_MSG = "다음 사용자와 시스템간 대화를 영어로 번역:\n" + ''.join(dialog)
        USER_INPUT_MSG = ''.join(dialog)
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You will be provided with a dialog in Korean, and your task is to translate it into English."},
                {"role": "user", "content": USER_INPUT_MSG},
            ],
            temperature=0,
        )
        traslated = response['choices'][0]['message']['content']
        print(f"{cnt, traslated}")
        tmplist.append(traslated)

        traslated_list = traslated.split('\n')
        traslated_list_by_topic_knowledge = [data.replace('[Topic]:', '--').replace('[Knowledge]:', '--').split('--') for data in traslated_list]
        final_list = []
        for idx, data in enumerate(traslated_list_by_topic_knowledge):
            uid = data[0][:data[0].find('.')]
            data[0] = data[0][data[0].find('.') + 1:]
            for i in range(3 - len(data)):
                data.append('')
            data.insert(0, uid)
            final_list.append(data)
        english_dialog_list.append([cnt, final_list])
        cnt += 1
        # time.sleep(60)
    except:
        print("ERROR %d" % cnt)
        break

with open('KT_eng_dialog_879-.pkl', 'wb') as f:
    pickle.dump(english_dialog_list, f)

print()
print('done')
