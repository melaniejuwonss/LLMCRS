import json
from copy import deepcopy
import pandas as pd
import numpy as np


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
    hit, total = 0, 0
    savejson = []
    datas = json.load(
        (open('../result/gpt-3.5-turbo/1024164819_rqcrs_withCoT.json', 'r', encoding='utf-8')))
    for data in datas:
        gen = data['RESPONSE'][data['RESPONSE'].lower().rfind('\n'):].lower()
        answer = data['LABEL'].split('(')[0].strip().lower()
        total += 1
        if answer in gen:
            hit += 1
        AVG_HIT = hit / total
        savejson.append({'RESPONSE': gen, 'LABEL': data['LABEL'], 'AVG_HIT': AVG_HIT})
    print(total)
    with open('../result/gpt-3.5-turbo/1024164819_rqcrs_withCoT_onlyName.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(savejson, indent=4))


if __name__ == "__main__":
    # popularity_crs()
    # popularity_model_rq12("llama7b")
    # popularity_model_rq3("chatgpt")
    # popularity_avg_hitratio("llama7b")
    # hitratio_type_rq1("chatgpt")
    # hitratio_type_rq3("chatgpt")

    # rq2test_firstreview("llama7b")
    # rq2_traintest_split()

    # evaluate()
    movie2name = json.load(
        open('../data/redial/movie2name.json', 'r', encoding='utf-8'))
    saveList= dict()
    for key, value in movie2name.items():
        id = value[0]
        title = value[1]
        if id == -1:
            continue
        saveList[id] = title
    with open('entityid2name.json','w',encoding='utf-8') as wf:
        wf.write(json.dumps(saveList, indent=4))
