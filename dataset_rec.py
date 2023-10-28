import random
from collections import defaultdict
from copy import copy, deepcopy

from torch.utils.data import Dataset
import torch
import json
from loguru import logger

from tqdm import tqdm
import os

import numpy as np


class CRSDatasetRec:
    def __init__(self, args, data_path):
        super(CRSDatasetRec, self).__init__()
        self.args = args
        self.data_path = data_path
        self.movie2name = json.load(
            open(os.path.join(self.data_path, 'movie2name.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
        self.entityid2name = dict()
        for key, value in self.movie2name.items():
            self.entityid2name[value[0]] = value[1]
        self.entity2id = json.load(
            open(os.path.join(self.data_path, 'entity2id.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
        ## Neg candidate
        # self.negativeList = json.load(
        #     open(os.path.join(self.data_path, 'negative_id.json'), 'r', encoding='utf-8'))
        self.movie2name = json.load(
            open(os.path.join(self.data_path, 'movie2name.json'), 'r', encoding='utf-8'))
        self.item_ids = json.load(
            open(os.path.join(self.data_path, 'movie_ids.json'), 'r', encoding='utf-8'))
        self.all_movies = json.load((open('data/content_data.json', 'r', encoding='utf-8')))[0]
        self.entityid2crsid = dict()
        for key, value in self.movie2name.items():
            if value[0] != -1:
                self.entityid2crsid[value[0]] = key
        self.all_movie_name, self.all_movie_id = [], []
        for movie in self.all_movies:
            title = movie['title']
            year = movie['year']
            if year is not None or title == str(year) or str(year) in title:
                title = title + " (" + str(year) + ")"
            crs_id = movie['crs_id']
            entity_id = self.movie2name[crs_id][0]
            self.all_movie_name.append(title)
            self.all_movie_id.append(entity_id)
        self._load_data()

    def _load_raw_data(self):
        # load train/valid/test data
        with open(os.path.join(self.data_path, 'train_data.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            logger.debug(f"[Load train data from {os.path.join(self.data_path, 'train_data.json')}]")
        with open(os.path.join(self.data_path, 'valid_data.json'), 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
            logger.debug(f"[Load valid data from {os.path.join(self.data_path, 'valid_data.json')}]")
        with open(os.path.join(self.data_path, 'test_data.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            logger.debug(f"[Load test data from {os.path.join(self.data_path, 'test_data.json')}]")

        return train_data, valid_data, test_data

    def _load_data(self):
        augmented_data_path = os.path.join(self.data_path, 'augmented')
        if os.path.isdir(augmented_data_path):
            with open(os.path.join(augmented_data_path, 'train_data_augment.json'), 'r', encoding='utf-8') as f:
                self.train_data = json.load(f)
            with open(os.path.join(augmented_data_path, 'valid_data_augment.json'), 'r', encoding='utf-8') as f:
                self.valid_data = json.load(f)
            with open(os.path.join(augmented_data_path, 'test_data_augment.json'), 'r', encoding='utf-8') as f:
                self.test_data = json.load(f)

        else:
            train_data_raw, valid_data_raw, test_data_raw = self._load_raw_data()  # load raw train, valid, test data

            train_data = self._raw_data_process(train_data_raw)  # training sample 생성
            self.train_data = self.rec_process_fn(train_data)
            # self.mergeWithNegatives(self.test_data)
            # with open('train_data_augment.json', 'w', encoding='utf-8') as f:
            #     f.write(json.dumps(self.train_data, indent=4))
            logger.debug("[Finish train data process]")

            test_data = self._raw_data_process(test_data_raw)
            self.test_data = self.rec_process_fn(test_data)
            # self.mergeWithNegatives(self.train_data)
            # with open('test_data_augment.json', 'w', encoding='utf-8') as f:
            #     f.write(json.dumps(self.test_data, indent=4))
            logger.debug("[Finish test data process]")

            valid_data = self._raw_data_process(valid_data_raw)
            self.valid_data = self.rec_process_fn(valid_data)
            # self.mergeWithNegatives(self.valid_data)
            # with open('valid_data_augment.json', 'w', encoding='utf-8') as f:
            #     f.write(json.dumps(self.valid_data, indent=4))
            logger.debug("[Finish valid data process]")

    def mergeWithNegatives(self, dataset):
        for data in tqdm(dataset, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            saveListId = self.sampleNegative(data['item'])
            data['negItems'] = saveListId

    def sampleNegative(self, ground_truth):
        candName, candIdx = [], []
        saveListName, saveListId = [], []
        i = 0
        while True:
            idx = random.randrange(0, len(self.item_ids))
            if str(ground_truth) == str(self.item_ids[idx]):
                continue
            if self.item_ids[idx] not in self.all_movie_id:
                continue
            else:
                i += 1
                candIdx.append(self.item_ids[idx])
            if i == 40:
                break

        return candIdx
        # with open('negative_name', 'w', encoding='utf-8') as f:
        #     f.write(json.dumps(saveListName, indent=4))
        # with open('negative_id', 'w', encoding='utf-8') as f:
        #     f.write(json.dumps(saveListId, indent=4))

    def rec_process_fn(self, dataset):
        augment_dataset = []
        for conv_dict in tqdm(dataset, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            if conv_dict['role'] == 'Recommender':
                for idx, movie in enumerate(conv_dict['items']):
                    augment_conv_dict = deepcopy(conv_dict)
                    augment_conv_dict['item'] = movie
                    augment_conv_dict['response'] = conv_dict['response']
                    augment_dataset.append(augment_conv_dict)

        logger.info('[Finish dataset process before rec batchify]')
        logger.info(f'[Rec Dataset size: {len(augment_dataset)}]')

        return augment_dataset

    def _raw_data_process(self, raw_data):
        augmented_convs = [self._merge_conv_data(conversation["dialog"]) for
                           conversation in tqdm(raw_data,
                                                bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}')]  # 연속해서 나온 대화들 하나로 합침 (예) S1, S2, R1 --> S1 + S2, R1
        augmented_conv_dicts = []
        for conv in tqdm(augmented_convs):
            augmented_conv_dicts.extend(self._augment_and_add(conv))  # conversation length 만큼 training sample 생성
        return augmented_conv_dicts

    def _merge_conv_data(self, dialog):
        augmented_convs = []
        last_role = None
        for utt in dialog:
            # @IDX 를 해당 movie의 name으로 replace
            for idx, word in enumerate(utt['text']):
                if word[0] == '@' and word[1:].isnumeric():
                    # utt['text'][idx] = '%s' %(word[1:])
                    utt['text'][idx] = '%s' % (self.movie2name[word[1:]][1])

            text = ' '.join(utt['text'])
            movie_ids = [self.entity2id[movie] for movie in utt['movies'] if
                         movie in self.entity2id]  # utterance movie(entity2id) 마다 entity2id 저장
            entity_ids = [self.entity2id[entity] for entity in utt['entity'] if
                          entity in self.entity2id]  # utterance entity(entity2id) 마다 entity2id 저장

            if utt["role"] == last_role:
                augmented_convs[-1]["text"] += ' ' + text
                augmented_convs[-1]["movie"] += movie_ids
                augmented_convs[-1]["entity"] += entity_ids
            else:
                if utt["role"] == 'Recommender':
                    role_name = 'System'
                else:
                    role_name = 'User'

                augmented_convs.append({
                    "role": utt["role"],
                    "text": f'{role_name}: {text}',  # role + text
                    "entity": entity_ids,
                    "movie": movie_ids,
                })
            last_role = utt["role"]

        return augmented_convs

    def _augment_and_add(self, raw_conv_dict):
        augmented_conv_dicts = []
        context_tokens = ""
        context_entities, context_words, context_items = [], [], []
        entity_set, word_set = set(), set()
        for i, conv in enumerate(raw_conv_dict):
            text_tokens, entities, movies = conv["text"], conv["entity"], conv["movie"]
            if len(context_tokens) > 0:
                conv_dict = {
                    "role": conv['role'],
                    "context_tokens": copy(context_tokens),
                    "response": text_tokens,
                    "context_entities": copy(context_entities),
                    "context_items": copy(context_items),
                    "items": movies
                }
                augmented_conv_dicts.append(conv_dict)
            context_tokens += text_tokens + " "
            context_items += movies
            for entity in entities + movies:
                if entity not in entity_set:
                    entity_set.add(entity)
                    context_entities.append(entity)

        return augmented_conv_dicts
