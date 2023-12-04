import json
from copy import deepcopy
import pandas as pd
import numpy as np
import jsonlines


def popularity_crs():
    files = ['train', 'test', 'valid']
    mention_cnt = dict()
    for file in files:
        datas = json.load((open('../data/' + file + '_data.json', 'r', encoding='utf-8')))

        for data in datas:
            dialogs = data['dialog']
            for utt in dialogs:
                texts = utt['text']
                for text in texts:
                    if '@' in text and text[1:].isdigit():
                        if text[1:] in mention_cnt.keys():
                            mention_cnt[text[1:]] += 1
                        else:
                            mention_cnt[text[1:]] = 1

    crsid2name = dict()
    content_data = json.load((open('../data/content_data.json', 'r', encoding='utf-8')))[0]
    for data in content_data:
        crs_id = data['crs_id']
        title = data['title']
        year = data['year']
        if year is not None or title == str(year) or str(year) in title:
            title = title + " (" + str(year) + ")"
        crsid2name[crs_id] = title

    moviename2cnt = dict()
    for key, value in mention_cnt.items():
        if key not in crsid2name.keys():
            continue
        name = crsid2name[key]
        if name not in moviename2cnt.keys():
            moviename2cnt[name] = value
        else:
            moviename2cnt[name] += value

    moviename2cnt = {k: v for k, v in sorted(moviename2cnt.items(), key=lambda item: item[1], reverse=True)}

    with open('../data/popularity.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(moviename2cnt, indent=4))


def popularity_model_rq12(model_name):
    files = ['rq2_firstreview']
    for file in files:
        correctFreq = dict()
        hitratio = dict()
        datas = json.load(
            (open('../data/modelResults/' + model_name + '/' + file + '.json', 'r', encoding='utf-8')))
        if 'rq2' not in file:
            num_datas = json.load(
                (open('../data/' + file + '_num.json', 'r', encoding='utf-8')))
        for data in datas:
            answer = data['ANSWER']
            label = deepcopy(answer)
            response = data['GEN']
            movie_name = label.replace('(', ')').split(')')[1].strip().lower()
            if movie_name in response.lower():
                if label[3:].lower() not in correctFreq.keys():
                    correctFreq[label[3:].lower()] = 1
                else:
                    correctFreq[label[3:].lower()] += 1
            else:
                if label[3:].lower() not in correctFreq.keys():
                    correctFreq[label[3:].lower()] = 0
                else:
                    correctFreq[label[3:].lower()] = 0

        if 'rq2' not in file:
            for key, value in num_datas.items():
                total_cnt = value
                if key not in correctFreq.keys():
                    hitratio[key] = 0
                else:
                    hitratio[key] = correctFreq[key] / total_cnt
            with open('../data/modelResults/' + model_name + '/' + file + 'hitratio.json', 'w', encoding='utf-8') as f:
                f.write(json.dumps(hitratio, indent=4))
        else:
            with open('../data/modelResults/' + model_name + '/' + file + 'hitratio.json', 'w', encoding='utf-8') as f:
                f.write(json.dumps(correctFreq, indent=4))


def popularity_model_rq3(model_name):
    correctFreq = dict()
    hitratio = dict()
    quizdatas = json.load(
        (open('../data/rq3_3choice.json', 'r', encoding='utf-8')))
    datas = json.load(
        (open('../data/modelResults/' + model_name + '/rq3.json', 'r', encoding='utf-8')))
    num_datas = json.load(
        (open('../data/rq3_num.json', 'r', encoding='utf-8')))
    for data, quizdata in zip(datas, quizdatas):
        question = quizdata['Question'].lower()
        for key, value in num_datas.items():
            if key in question[:question.index('choices:')]:
                standard_movie = key
                break

        answer = data['ANSWER']
        label = deepcopy(answer)
        response = data['GEN']
        movie_name = label.replace('(', ')').split(')')[1].strip().lower()
        if movie_name in response.lower():
            if standard_movie.lower() not in correctFreq.keys():
                correctFreq[standard_movie.lower()] = 1
            else:
                correctFreq[standard_movie.lower()] += 1

    for key, value in num_datas.items():
        total_cnt = value
        if key not in correctFreq.keys():
            hitratio[key] = 0
        else:
            hitratio[key] = correctFreq[key] / total_cnt
    with open('../data/modelResults/' + model_name + '/' + 'rq3hitratio.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(hitratio, indent=4))


def popularity_avg_hitratio(model_name):
    popularity_crs = json.load(
        (open('../data/popularity.json', 'r', encoding='utf-8')))
    cnt, bin = 0, 1
    movie2bin = dict()
    for key, value in popularity_crs.items():
        cnt += 1
        movie2bin[key] = bin
        if cnt == 627:
            cnt = 0
            bin += 1
    bin2hitratio = dict()
    bincnt = [0] * 11
    files = ['rq2_firstreviewhitratio']
    for file in files:
        datas = json.load(
            (open('../data/modelResults/' + model_name + '/' + file + '.json', 'r', encoding='utf-8')))
        for key, value in datas.items():
            if key in movie2bin.keys():
                bin = movie2bin[key]
                if bin not in bin2hitratio.keys():
                    bin2hitratio[bin] = value
                    bincnt[bin] = 1
                else:
                    cnt = bincnt[bin]
                    cur_hitratio = bin2hitratio[bin]
                    cur_hitratio *= cnt
                    cur_hitratio += value
                    new_hitratio = cur_hitratio / (cnt + 1)
                    bincnt[bin] += 1
                    bin2hitratio[bin] = new_hitratio
        myKeys = list(bin2hitratio.keys())
        myKeys.sort()
        sorted_dict = {i: bin2hitratio[i] for i in myKeys}
        print(sum(sorted_dict.values()) / 10)
        with open('../data/modelResults/' + model_name + '/' + file + '_avg.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(sorted_dict, indent=4))


def hitratio_type_rq1(model_name):
    quizdatas = json.load(
        (open('../data/rq1.json', 'r', encoding='utf-8')))
    datas = json.load(
        (open('../data/modelResults/' + model_name + '/rq1.json', 'r', encoding='utf-8')))
    type = ""
    total_cnt = [0] * 4
    correctFreq = dict()
    correctFreq['director'] = 0
    correctFreq['genre'] = 0
    correctFreq['writer'] = 0
    correctFreq['actor'] = 0
    for data, quizdata in zip(datas, quizdatas):
        question = quizdata['Question']

        if 'directed' in question:
            type = "director"
            total_cnt[0] += 1
        elif True in [True if word in question else False for word in
                      ['written', 'scripted', 'authored', 'penned', 'crafted']]:
            type = "writer"
            total_cnt[2] += 1
        elif True in [True if word in question else False for word in ['act', 'appear', 'cast', 'role', 'perform']]:
            type = "actor"
            total_cnt[3] += 1
        elif True in [True if word in question else False for word in ['genre', 'Which is']]:
            type = "genre"
            total_cnt[1] += 1

        answer = data['ANSWER']
        label = deepcopy(answer)
        response = data['GEN']
        movie_name = label.replace('(', ')').split(')')[1].strip().lower()
        if movie_name in response.lower():
            correctFreq[type] += 1
    saveResult = dict()
    for key, value in correctFreq.items():
        type_cnt = value
        if key == "director":
            dir_cnt = type_cnt / total_cnt[0]
            saveResult[key] = dir_cnt
        elif key == "genre":
            genre_cnt = type_cnt / total_cnt[1]
            saveResult[key] = genre_cnt
        elif key == "writer":
            writer_cnt = type_cnt / total_cnt[2]
            saveResult[key] = writer_cnt
        elif key == "actor":
            actor_cnt = type_cnt / total_cnt[3]
            saveResult[key] = actor_cnt
    with open('../data/modelResults/' + model_name + '/rq1_typeavg.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(saveResult, indent=4))


def hitratio_type_rq3(model_name):
    quizdatas = json.load(
        (open('../data/rq3_3choice.json', 'r', encoding='utf-8')))
    datas = json.load(
        (open('../data/modelResults/' + model_name + '/rq3.json', 'r', encoding='utf-8')))
    type = ""
    total_cnt = [0] * 4
    correctFreq = dict()
    correctFreq['director'] = 0
    correctFreq['genre'] = 0
    correctFreq['writer'] = 0
    correctFreq['actor'] = 0
    for data, quizdata in zip(datas, quizdatas):
        question = quizdata['Question']

        if True in [True if word in question else False for word in ['directed', 'director']]:
            type = "director"
            total_cnt[0] += 1
        elif True in [True if word in question else False for word in ['written', 'writer', 'authored']]:
            type = "writer"
            total_cnt[2] += 1
        elif True in [True if word in question else False for word in ['act', 'actor']]:
            type = "actor"
            total_cnt[3] += 1
        elif True in [True if word in question else False for word in ['genre']]:
            type = "genre"
            total_cnt[1] += 1

        answer = data['ANSWER']
        label = deepcopy(answer)
        response = data['GEN']
        movie_name = label.replace('(', ')').split(')')[1].strip().lower()
        if movie_name in response.lower():
            correctFreq[type] += 1
    saveResult = dict()
    for key, value in correctFreq.items():
        type_cnt = value
        if key == "director":
            dir_cnt = type_cnt / total_cnt[0]
            saveResult[key] = dir_cnt
        elif key == "genre":
            genre_cnt = type_cnt / total_cnt[1]
            saveResult[key] = genre_cnt
        elif key == "writer":
            writer_cnt = type_cnt / total_cnt[2]
            saveResult[key] = writer_cnt
        elif key == "actor":
            actor_cnt = type_cnt / total_cnt[3]
            saveResult[key] = actor_cnt
    with open('../data/modelResults/' + model_name + '/rq3_typeavg.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(saveResult, indent=4))


def rq2test_firstreview(model_name):
    datas = json.load(
        (open('../data/modelResults/' + model_name + '/rq2_5choice.json', 'r', encoding='utf-8')))
    hit = 0
    cnt = 0
    save_dict = []
    movie_name = ""
    for data in datas:
        label = data['ANSWER'].lower()
        movname = label.replace('(', ')').split(')')[1].strip().lower()
        gen = data['GEN'].lower()
        if movie_name != movname:
            movie_name = movname
            cnt += 1
            if movname in gen.lower():
                hit += 1
            hit_ratio = hit / cnt
            save_dict.append({'GEN': gen, 'ANSWER': label, 'AVG_HIT': hit_ratio})
    with open('../data/modelResults/' + model_name + '/rq2_5choice_firstreview.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(save_dict, indent=4))


def rq2_traintest_split():
    datas = json.load(
        (open('../data/gpt-3.5-turbo/rq2_3choice_test.json', 'r', encoding='utf-8')))

    test_dict, train_dict = [], []
    movie_name = ""
    for data in datas:
        question = data['Question']
        answer = data['Answer']
        movname = answer.replace('(', ')').split(')')[1].strip().lower()
        if movie_name != movname:
            movie_name = movname
            test_dict.append({'Question': question, 'Answer': answer})
        else:
            train_dict.append({'Question': question, 'Answer': answer})
    with open('../data/rq2_3choice_test_new.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(test_dict, indent=4))
    with open('../data/rq2_3choice_new.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(train_dict, indent=4))


def evaluate():
    new_idx = json.load(open('../data/redial/test_new_idx.json', 'r'))
    # for i in range(5):
    llamadatas = jsonlines.open(
        f'../result/meta-llama-Llama-2-7b-chat-hf/231204/1204044111_meta-llama-Llama-2-7b-chat-hf_onlyReview5_D2I_E4.json')
    # gpt_datas = json.load(open('../result/gpt-3.5-turbo/imp/lastUtt.json', 'r', encoding='utf-8'))
    test_datas = json.load(open('../data/redial/augmented/test_data_augment.json', 'r'))
    hit, cnt, mentioned_cnt, not_mentioned_cnt, mentioned_hit, not_mentioned_hit = 0, 0, 0, 0, 0, 0

    for idx, (test_data, data) in enumerate(zip(test_datas, llamadatas)):
        context = test_data['context_tokens']
        gen = data['GEN'].split('|')[0].strip()
        answer = data['ANSWER'].strip()

        if idx in new_idx:
            if answer in gen:
                not_mentioned_hit += 1
            not_mentioned_cnt += 1

        if answer in gen:
            hit += 1
            if idx in new_idx:
                not_mentioned_hit += 1
            else:
                mentioned_hit += 1
        if idx in new_idx:
            not_mentioned_cnt += 1
        else:
            mentioned_cnt += 1
        cnt += 1

    print(f"{round(hit/ cnt * 100,2)}\t{round(mentioned_hit / mentioned_cnt * 100,2)}\t{round(not_mentioned_hit / not_mentioned_cnt * 100, 2)}")  # 55
    ############################################
    # test_datas = json.load(open('../data/redial/augmented/test_data_augment.json', 'r'))
    # # for i in range(5):
    # hit, cnt, mentioned_cnt, mentioned_hit, not_mentioned_cnt, not_mentioned_hit = 0, 0, 0, 0, 0, 0,
    # gpt_datas = json.load(open('../result/gpt-3.5-turbo/imp/lastUtt.json', 'r',encoding='utf-8'))
    # with jsonlines.open(
    #         f'../result/meta-llama-Llama-2-7b-chat-hf/231116/1116203131_meta-llama-Llama-2-7b-chat-hf_D2I_new구분_E4.json') as llamadata:  # 1028225629_rqwithoutCoT_title_epoch5 # 1113202228_rqfineTuneCRS_CoT_p2i_E3
    #     for idx, (data, test_data) in enumerate(zip(llamadata, test_datas)):
    #         gen = data['GEN'].split('\n')[1].strip().split('(')[0].strip().lower()
    #         # gen = gen.replace("in this context, the system should recommend the following new item:", "").strip()
    #         # gen = gen.replace("in this context, the system should chat about the following mentioned item:",
    #         #                   "").strip()
    #         # gen = gen.replace("in this context, the system should mention the following item:", "").strip()
    #         context = test_data['context_tokens'].lower()
    #         if gen in context:
    #             mentioned_cnt += 1
    #         else:
    #             not_mentioned_cnt += 1
    # print(mentioned_cnt, mentioned_cnt/4003 * 100, not_mentioned_cnt, not_mentioned_cnt/4003 * 100)
    ############################################
    # new_idx = json.load(open('../data/redial/test_new_idx.json', 'r', encoding='utf-8'))
    # test_datas = json.load(open('../data/redial/cot/test_data_cot_intention.json', 'r', encoding='utf-8'))
    # result_datas_prompt = jsonlines.open(
    #     '../result/meta-llama-Llama-2-7b-chat-hf/231124/1124144016_meta-llama-Llama-2-7b-chat-hf_intention_validation_D2INIPROMPT_E2.json')
    # result_datas_real = jsonlines.open(
    #     '../result/meta-llama-Llama-2-7b-chat-hf/231124/1124160933_meta-llama-Llama-2-7b-chat-hf_intention_validation_D2INPROMPTI_E3.json')
    # revise_list = []
    # prompt_recommend, real_recommend, prompt_chat, real_chat = 0, 0, 0, 0
    # for idx, (prompt_data, test_data) in enumerate(zip(result_datas_real, test_datas)):
    #     context = test_data['context_tokens'].split('\n')[0].lower()
    #     prompt_type = prompt_data['GEN'].split('\n')[1].lower()  # .split('(')[0].strip()
    #     prompt_type = prompt_type.replace("in this context, the system should chat about the following mentioned item:",
    #                                       "")
    #     prompt_type = prompt_type.replace("in this context, the system should recommend the following new item:", "")
    #     prompt_type = prompt_type.replace("in this context, the system should mention the following item:", "")
    #     prompt_type = prompt_type.split('(')[0].strip()
    #     if idx in new_idx:
    #         if prompt_type not in context:
    #             real_recommend += 1
    #     elif idx not in new_idx:
    #         if prompt_type in context:
    #             real_chat += 1
    #
    # print(real_recommend, real_chat)
    ####################################### train_data_cot_only구분
    # train_datas = json.load(open('../data/redial/cot/train_data_cot_only구분.json', 'r', encoding='utf-8'))
    # for train_data in train_datas:
    #     context = train_data['context_tokens'].split('\n')[0]
    #     item = train_data['item']
    #     revise_list.append({'context_tokens': context, 'item': item})
    # with open('train_data_cot_d2ti.json', 'w', encoding='utf-8') as f:
    #     f.write(json.dumps(revise_list, indent=1))
    #########################################################
    # for i in range(5):
    #     llamadatas = jsonlines.open(
    #         f'../result/meta-llama-Llama-2-7b-chat-hf/imp/1117091756_meta-llama-Llama-2-7b-chat-hf_D2I_E2.json')
    #     gpt_datas = json.load(open('../result/gpt-3.5-turbo/imp/lastUtt.json', 'r', encoding='utf-8'))
    #     new_idx = json.load(open('../data/redial/test_new_idx.json', 'r'))
    #     test_datas = json.load(open('../data/redial/augmented/test_data_augment.json', 'r', encoding='utf-8'))
    #     hit, cnt, mentioned_cnt, not_mentioned_cnt, mentioned_hit, not_mentioned_hit = 0, 0, 0, 0, 0, 0
    #     for idx, (llama_data, test_data) in enumerate(zip(llamadatas,test_datas)):
    #         gen = llama_data['GEN'].lower()
    #         context = test_data['context_tokens'].lower()
    #         gen_title = gen.split('(')[0].strip()
    #         gen_year = gen.split('(')[-1].replace(')', '').replace('</s>','').replace('<unk>','').strip()
    #
    #         if gen_year.isdigit() is False:
    #             gen_year = ''
    #
    #         if idx in new_idx:
    #             # if "recommend" in gen:
    #             #     not_mentioned_hit += 1
    #             if gen_title not in context or gen_year not in context:
    #                 not_mentioned_hit += 1
    #             not_mentioned_cnt += 1
    #         elif idx not in new_idx:
    #             # if "chat" in gen:
    #             #     mentioned_hit += 1
    #             if gen_title in context and gen_year in context:
    #                 mentioned_hit += 1
    #             mentioned_cnt += 1
    #
    #     print(round(mentioned_hit / mentioned_cnt * 100, 2) ,round(not_mentioned_hit / not_mentioned_cnt * 100, 2))
    ###################################
    # din2i_trains = json.load(open('../data/redial/cot/train_data_cot_din2i.json', 'r'))
    # train_datas = json.load(open('../data/redial/augmented/train_data_augment.json', 'r'))
    # new_train_list = []
    # for din2i_train, train_data in zip(din2i_trains, train_datas):
    #     item = train_data['item']
    #     context_tokens = train_data['context_tokens']
    #     intention = "\n" + din2i_train['context_tokens'][din2i_train['context_tokens'].rfind("The user's intention in"):]
    #     new_train_list.append({'context_tokens': context_tokens + intention, 'item': item})
    # with open('../data/redial/cot/train_data_cot_din2i.json','w', encoding='utf-8') as f:
    #     f.write(json.dumps(new_train_list, indent=2))
    ######################################
    # movie2name = json.load(open('../data/redial/movie2name.json', 'r'))
    # test_datas = json.load(open('../data/redial/augmented/test_data_augment.json', 'r'))
    # all_movies = dict()
    # save_list = []
    # for key, value in movie2name.items():
    #     all_movies[value[1]] = 0
    # # llamadatas = jsonlines.open(
    # #     f'../result/meta-llama-Llama-2-7b-chat-hf/231127/1127150017_meta-llama-Llama-2-7b-chat-hf_D2I_neweval_E2.json')
    # # for llamadata in llamadatas:
    # #     gens = llamadata['GEN'].split(',')
    # #     for gen in gens:
    # #         if gen in all_movies.keys():
    # #             all_movies[gen] += 1
    # for test_data in test_datas:
    #     response = test_data['response']
    #     context = test_data['context_tokens']
    #     save_list.append({'context': context, 'response': response})
    # print(all_movies)
    # with open('dialog_response.json','w',encoding='utf-8') as f:
    #     f.write(json.dumps(save_list, indent=2))
    # # 전체 test data 상 1,570 개의 item 이 answer 였음
    # # DIN2I 803 만 gen
    # # D2I 664 으로 gen
    ######################################
    # syn_datas = json.load(open('../data/redial/synthetic/synthetic_dialog_review.json', 'r', encoding='utf-8'))
    # cnt = 0
    # save_data = []
    # for syn_data in syn_datas:
    #     cnt += 1
    #     prev_output = syn_data['OUTPUT']
    #     all_responses = prev_output.split("\n")
    #     input = ""
    #     for idx, response in enumerate(all_responses):
    #          if "System:" in response:
    #              output = response.replace("System: ","")
    #              save_data.append({'INPUT': input + "System: ", 'OUTPUT': output})
    #          input += (response + "\n")
    #
    #     # output = prev_output[prev_output.rfind("System:"):]
    #     # input = prev_output[:prev_output.rfind("System:")]
    # with open('../data/redial/synthetic/synthetic_dialog_review_train.json','w',encoding='utf-8') as f:
    #     f.write(json.dumps(save_data, indent=2))

if __name__ == "__main__":
    # popularity_crs()
    # popularity_model_rq12("llama7b")
    # popularity_model_rq3("chatgpt")
    # popularity_avg_hitratio("llama7b")
    # hitratio_type_rq1("chatgpt")
    # hitratio_type_rq3("chatgpt")

    # rq2test_firstreview("llama7b")
    # rq2_traintest_split()

    evaluate()
