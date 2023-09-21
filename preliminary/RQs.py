# To test whether LLM knows item-feature relationships and item-item relationships
import json
import random
from copy import deepcopy
from tqdm import tqdm

content_data = json.load((open('../data/content_data.json', 'r', encoding='utf-8')))[0]

genre_example_question = " Here is an example. Which movie shares genre with Soul (2020)? \n Choices: a) Inside Out (2015) b) The Pianist (2002)  c) Kiss the Girls (1997) \n"
genre_example_interpret = " Answer form: \n" \
                          " 1.Explanation: The genres of Soul (2020) are animation, adventure, and comedy." \
                          " The genres of Inside Out (2015) are animation, adventure, and comedy." \
                          " 2.Answer: a) Inside Out (2015). \n"
# " The genres of The Pianist (200) are biography, drama, and music." \
# " The genres of Kiss the Girls (1997) are crime, drama, and mystery." \
# " The genres of Harry Brown (2009) are action, crime, and drama." \
# " The genres of Meet Joe Black (1998) are drama, fantasy, and romance." \

director_example_question = " Here is an example. Which movie was directed by the same person who directed Superintelligence (2020)? \n Choices: a) Pretty when you cry (2001) b) The Boss (2016) c) Ghosts of Mars (2001) \n"
director_example_interpret = " Answer form: \n" \
                             " 1.Explanation: Superintelligence (2020) was directed by Ben Falcone." \
                             " The Boss (2016) was directed by Ben Falcone." \
                             " 2.Answer: b) The Boss (2016). \n"
# " Ghosts of Mars (2001) was directed by John Carpenter." \
# " Pretty when you cry (2001) was directed by Jack N. Green." \
# " Ace Ventura: when nature calls (1995) was directed by Steve Oedekerk." \
# " Brick (2005) was directed by Rian Johnson." \
writer_example_question = " Here is an example. Which movie was also written by Tenet (2020) writer? \n Choices: a) Strange Days (1995) b) A Nightmare on Elm Street (2010)  c) Inception (2010) \n"
writer_example_interpret = " Answer form: \n " \
                           " 1.Explanation: The writer of Tenet (2020) is Christopher Nolan." \
                           " Inception (2010) is written by Christopher Nolan." \
                           " 2.Answer: c) Inception (2010). \n"
# " Strange Days (1995) is written by James Cameron." \
# " A Nightmare on Elm Street (2010) is written by Wesley Strick." \
# " Dance with Me (1997) is written by Daryl Matthews." \
# " Frankenstein (1931) is written by John L. Balderston." \

actor_example_question = " Here is an example. In which another movie did an actor from Pixie (2020) act? \n Choices: a) The wolf of wall street (2013) b) Chicago (1927) c) Ouija (2014) \n"
actor_example_interpret = " Answer form: \n " \
                          " 1.Explanation: Olivia Cooke appeared in Pixie (2020)." \
                          " Olivia Cooke acted in Ouija (2014)." \
                          " 2.Answer: c) Ouija (2014). \n"
# " Leonardo DiCaprio acted in The wolf of wall street (1929)." \
# " Phyllis Haver acted in Chicago (1927)." \
    # " Jeremy Sisto acted in This Space between Us (1999)." \
# " Lindsay Lohan acted in Mean Girls (2004)." \
question_prompt = "Here is our question."
prefix_template = "The following multiple-choice quiz has 3 choices (a,b,c). Select the best answer from the given choices. \n First, I will show you an example."
obj_templates = [
    ["Which is %s movie?", "Which movie belongs to the %s genre?", "Which movie is classified as %s genre?",
     "Which movie falls under the %s genre category?", "Which movie is a part of the %s genre category?"],
    ["Which movie was directed by %s?", "Choose one of the movies directed by %s.", "Identify a movie directed by %s.",
     "Select a film directed by %s.", "Pick a movie that was directed by %s."],
    ["In which movie did %s act", "In which film did %s appear?", "Which film features %s in the cast?",
     "Which movie showcases %s in a role?", "In which cinematic production did %s perform?"],
    ["Which movie was written by %s?", "Which film was scripted by %s?", "Which cinematic piece was authored by %s?",
     "Which movie was penned by %s?", "Which film was crafted by %s?"]
]
subj_template = ["The following text is a review of a certain movie. %s Which movie is it?",
                 "The text below is a critique of a particular movie. %s Can you guess which movie it is?",
                 "Below is a review of a specific film. %s Can you identify the film in question?",
                 "Here is an evaluation of a specific movie. %s Can you pinpoint which movie is being discussed?",
                 "The passage below offers a critique of a particular film. %s Are you able to determine which film it is?"]

item_template = [
    ["Which movie shares the genre with %s?", "Which movie belongs to the same genre as %s?",
     "Can you name a movie that is in the same genre as %s?",
     "What is another movie that falls within the same genre as %s?", "Which movie has a similar genre to %s?"],
    ["Which movie shares a director with %s?", "Which movie was directed by the same person who directed %s?",
     "What is another movie directed by the %s director?", "Which movie has the same director as %s?",
     "Can you choose a movie that has the same director as %s?"],
    ["In which another movie did an actor from %s act?", "Can you choose another movie where an actor from %s?",
     "In which other film has an actor from %s participated?",
     "Can you choose another movie where an actor from %s appeared?",
     "Can you identify another movie where an actor from %s has had a role?"],
    ["Which movie was also written by %s writer?", "Which other movie was authored by the writer of %s?",
     "Can you name another movie that the %s writer worked on?", "What other film features work from the writer of %s?",
     "Do you know of another movie scripted by the writer who also wrote %s?"]
]
postfix_template = "\n Choices: a) %s b) %s c) %s"
choice_alphabet = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'}


def createSources():
    itemFeatures, writer2item, actor2item, genre2item, director2item, item2feature, feature2item = dict(), dict(), dict(), dict(), dict(), dict(), dict()
    all_titles = set()
    for data in content_data:
        title = data['title']
        year = data['year']
        if year is not None or title == str(year) or str(year) in title:
            title = title + " (" + str(year) + ")"
        genres = data['meta']['genre']
        directors = data['meta']['director']
        actors = data['meta']['stars']
        writers = data['meta']['writers']
        reviews = data['review']
        all_titles.add(title)
        item2feature[title] = [[] for _ in range(5)]
        itemFeatures[title] = []
        for genre in genres:
            if genre not in genre2item.keys():
                genre2item[genre] = [title]
            else:
                if title not in genre2item[genre]:
                    genre2item[genre].append(title)
            if title not in item2feature.keys():
                item2feature[title][0] = [genre]
                itemFeatures[title] = [genre]
            else:
                item2feature[title][0].append(genre)
                itemFeatures[title].append(genre)

        for director in directors:
            if director not in director2item.keys():
                director2item[director] = [title]
            else:
                if title not in director2item[director]:
                    director2item[director].append(title)
            if title not in item2feature.keys():
                item2feature[title][1] = [director]
                itemFeatures[title] = [director]
            else:
                item2feature[title][1].append(director)
                itemFeatures[title].append(director)

        for actor in actors:
            if actor not in actor2item.keys():
                actor2item[actor] = [title]
            else:
                if title not in actor2item[actor]:
                    actor2item[actor].append(title)
            if title not in item2feature.keys():
                item2feature[title][2] = [actor]
                itemFeatures[title] = [actor]
            else:
                item2feature[title][2].append(actor)
                itemFeatures[title].append(actor)

        for writer in writers:
            if writer not in writer2item.keys():
                writer2item[writer] = [title]
            else:
                if title not in writer2item[writer]:
                    writer2item[writer].append(title)
            if title not in item2feature.keys():
                item2feature[title][3] = [writer]
                itemFeatures[title] = [writer]
            else:
                item2feature[title][3].append(writer)
                itemFeatures[title].append(writer)

        item2feature[title][4] = reviews

    return all_titles, itemFeatures, genre2item, writer2item, actor2item, director2item, item2feature


def sample_choices(candidates, choice_num, itemFeatures, target_features=None):
    cnt = 0
    if target_features is None:
        choices = random.sample(candidates, choice_num)
        return choices
    else:
        while True:
            cnt = 0
            choices = random.sample(candidates, choice_num)
            for choice in choices:
                feature_cnt = 0
                for target_feature in target_features:
                    if target_feature not in itemFeatures[choice]:
                        feature_cnt += 1
                if feature_cnt == len(target_features):
                    cnt += 1
            if cnt == choice_num:
                break
        return choices


def create_rq1(item2feature, itemFeatures):
    result_list = []
    title_quiz_num = dict()
    cnt = 0
    for title in tqdm(item2feature.keys(), bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
        all_features = item2feature[title]
        all_items = deepcopy(list(itemFeatures.keys()))
        all_items.remove(title)
        for i in range(4):  # genre, director, actor, writer
            features = all_features[i]
            for feature in features:
                target_feature = feature
                templates = random.sample(obj_templates[i], 1)
                for template in templates:  # 각 feature type 의 template 마다 생성
                    choices = sample_choices(all_items, 4, itemFeatures, [target_feature])
                    choices.append(title)
                    random.shuffle(choices)
                    feature_template = template % (target_feature)
                    choice_template = postfix_template % (tuple(choices))
                    whole_template = prefix_template + feature_template + choice_template
                    alpha = choice_alphabet[choices.index(title)]
                    answer = alpha + ')' + ' ' + title
                    cnt += 1
                    if title not in title_quiz_num.keys():
                        title_quiz_num[title] = 1
                    else:
                        title_quiz_num[title] += 1
                    result_list.append({'Question': whole_template, 'Answer': answer})
    print("RQ1 AVG: " + str(cnt / len(title_quiz_num)))  # 41.2, #TOTAL: 278,665
    # with open('../data/rq1_num.json', 'w', encoding='utf-8') as result_f:
    #     result_f.write(json.dumps(title_quiz_num, indent=4))
    with open('../data/rq1_5choice.json', 'w', encoding='utf-8') as result_f:
        result_f.write(json.dumps(result_list, indent=4))


def create_rq2(item2feature, itemFeatures):
    result_list = []
    title_quiz_num = dict()
    cnt = 0
    for title in tqdm(item2feature.keys(), bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
        all_items = deepcopy(list(itemFeatures.keys()))
        all_items.remove(title)
        onlyMovieName = title[:title.find('(') - 1]
        reviews = item2feature[title][4][:5]
        for review in reviews:
            review = str.lower(review).replace(onlyMovieName, "<movie>")
            review500 = " ".join(review.split(' ')[:128])  # first 500 words
            templates = random.sample(subj_template, 1)
            for template in templates:
                choices = sample_choices(all_items, 4, itemFeatures, None)
                choices.append(title)
                random.shuffle(choices)
                feature_template = template % (review500)
                choice_template = postfix_template % (tuple(choices))
                whole_template = prefix_template + feature_template + choice_template
                alpha = choice_alphabet[choices.index(title)]
                answer = alpha + ')' + ' ' + title
                cnt += 1
                if title not in title_quiz_num.keys():
                    title_quiz_num[title] = 1
                else:
                    title_quiz_num[title] += 1
                result_list.append({'Question': whole_template, 'Answer': answer})

    print("RQ2 AVG: " + str(cnt / len(title_quiz_num)))  # 24.2, #TOTAL: 150,360
    # with open('../data/rq2_num.json', 'w', encoding='utf-8') as result_f:
    #     result_f.write(json.dumps(title_quiz_num, indent=4))
    with open('../data/rq2_5choice.json', 'w', encoding='utf-8') as result_f:
        result_f.write(json.dumps(result_list, indent=4))


def create_rq3(itemFeatures, genre2item, writer2item, actor2item, director2item, item2feature):
    result_list = []
    title_quiz_num = dict()
    cnt = 0
    for title in tqdm(item2feature.keys(), bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
        all_features = item2feature[title]
        all_items = deepcopy(list(itemFeatures.keys()))
        all_items.remove(title)

        for idx in range(4):  # genre, director, actor, writer
            target_features = all_features[idx]
            templates = random.sample(item_template[idx], 1)
            for template in templates:
                if idx == 0:
                    if len(target_features) == 0:
                        continue
                    genre_items = list()
                    genre_items.extend([genre2item[target_feature] for target_feature in target_features][0])
                    set(genre_items)
                    genre_items.remove(title)
                    if len(genre_items) == 0:
                        answer_title = "None"
                    else:
                        answer_title = random.sample(genre_items, 1)[
                            0]  # target item 과 genre 가 겹치는 다른 영화들 중에서 정답 하나 sample
                elif idx == 1:
                    if len(target_features) == 0:
                        continue
                    director_items = list()
                    director_items.extend([director2item[target_feature] for target_feature in target_features][0])
                    set(director_items)
                    director_items.remove(title)
                    if len(director_items) == 0:
                        answer_title = "None"
                    else:
                        answer_title = random.sample(genre_items, 1)[0]
                elif idx == 2:
                    if len(target_features) == 0:
                        continue
                    actor_items = list()
                    actor_items.extend([actor2item[target_feature] for target_feature in target_features][0])
                    set(actor_items)
                    actor_items.remove(title)
                    if len(actor_items) == 0:
                        answer_title = "None"
                    else:
                        answer_title = random.sample(genre_items, 1)[0]
                elif idx == 3:
                    if len(target_features) == 0:
                        continue
                    writer_items = list()
                    writer_items.extend([writer2item[target_feature] for target_feature in target_features][0])
                    set(writer_items)
                    writer_items.remove(title)
                    if len(writer_items) == 0:
                        answer_title = "None"
                    else:
                        answer_title = random.sample(genre_items, 1)[0]
                choices = sample_choices(all_items, 2, itemFeatures, target_features)  ################# choices
                choices.append(answer_title)
                random.shuffle(choices)

                feature_template = template % (title)
                choice_template = postfix_template % (tuple(choices))
                if idx == 0:  # genre, director, actor, writer
                    whole_template = prefix_template + genre_example_question + genre_example_interpret + question_prompt + feature_template + choice_template
                elif idx == 1:
                    whole_template = prefix_template + director_example_question + director_example_interpret + question_prompt + feature_template + choice_template
                elif idx == 2:
                    whole_template = prefix_template + actor_example_question + actor_example_interpret + question_prompt + feature_template + choice_template
                elif idx == 3:
                    whole_template = prefix_template + writer_example_question + writer_example_interpret + question_prompt + feature_template + choice_template

                    # whole_template = prefix_template + feature_template + choice_template
                alpha = choice_alphabet[choices.index(answer_title)]
                answer = alpha + ')' + ' ' + answer_title
                cnt += 1
                if title not in title_quiz_num.keys():
                    title_quiz_num[title] = 1
                else:
                    title_quiz_num[title] += 1
                result_list.append({'Question': whole_template, 'Answer': answer})
    print("RQ3 AVG: " + str(cnt / len(title_quiz_num)))  # 19.2, TOTAL: 129,690
    # with open('../data/rq3_random3choice_num.json', 'w', encoding='utf-8') as result_f:
    #     result_f.write(json.dumps(title_quiz_num, indent=4))
    with open('../data/rq3_3choice_example_llama.json', 'w', encoding='utf-8') as result_f:
        result_f.write(json.dumps(result_list, indent=4))


if __name__ == "__main__":
    all_titles, itemFeatures, genre2item, writer2item, actor2item, director2item, item2feature = createSources()
    # create_rq1(item2feature, itemFeatures)
    # create_rq2(item2feature, itemFeatures)
    create_rq3(itemFeatures, genre2item, writer2item, actor2item, director2item, item2feature)
