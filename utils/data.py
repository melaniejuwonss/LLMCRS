import json
import os


def quiz_read_data(args, mode):
    data_path = os.path.join(args.home, 'data', 'quiz')
    RQ_data = json.load((open(f"{data_path}/rq{str(args.rq_num)}_{mode}.json", 'r', encoding='utf-8')))
    question, answer = [], []
    data_samples = []
    for data in RQ_data:
        question.append(data['context_tokens'])
        answer.append(data['item'])

    for t_input, t_output in zip(question, answer):
        data_samples.append((t_input, t_output))
    if mode == "test":
        data_samples = data_samples[:100]

    instructions = [i[0] for i in data_samples]
    labels = [i[1] for i in data_samples]
    train_new = [True for i in data_samples]

    return instructions, labels, train_new


def plot_read_data(args, mode):
    data_path = os.path.join(args.home, 'data', 'redial', 'plot')
    if not os.path.exists(data_path): os.mkdir(data_path)
    with open(os.path.join(data_path, f'plot_1.json'), 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if mode == "test":
        dataset = dataset[:100]

    instructions = [data['context_tokens'] for data in dataset]
    labels = [data['item'] for data in dataset]
    train_new = [True for data in dataset]

    if args.pretrain:
        plot_template = "I will give you a plot of a movie %s\n%s"
        instructions = [plot_template % (label, instruction) for instruction, label in
                        zip(instructions, labels)]
        labels = ['' for label in labels]

    return instructions, labels, train_new


def meta_plot_review_read_data(args, mode='train'):
    instructions, labels, train_new = [], [], []
    data_path = os.path.join(args.home, 'data', 'redial', 'passage')
    if not os.path.exists(data_path): os.mkdir(data_path)
    with open(os.path.join(data_path, f'raw2refined.json'), 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    for data in dataset:
        title = data['item']
        raw_reviews = data['raw_reviews']
        refined_reviews = data['refined_reviews']

        if len(raw_reviews) != len(refined_reviews):
            print(title + "::: not equal lengths")

        for raw, refined in zip(raw_reviews, refined_reviews):
            if mode == 'train':
                context_tokens = f"""I will give you a raw review of movie {title}.\n{raw}\nPlease rephrase the raw review:\n{refined}"""
            else:
                context_tokens = f"""I will give you a raw review of movie {title}.\n{raw}\nPlease rephrase the raw review:\n"""
            instructions.append(context_tokens)
            labels.append('')
            train_new.append(True)

        # meta = data['meta']
        # plot = data['plot']
        # if len(data['review_list']) == 0 or plot == '':
        #     continue
        #
        # for review in data['review_list']:
        #     # review = data['review_list'][0]
        #     title = data['item']
        #     # if review == '':
        #     #     continue
        #     if mode == 'train':
        #         context_tokens = f"""I will give you information of movie {title}.\nGenres, directors, writers, and actors of {title}:\n{meta}\nPlot of the movie {title}:\n{plot}\nCan you write a review for the movie {title}:\n{review}"""
        #     else:
        #         context_tokens = f"""I will give you information of movie {title}.\nGenres, directors, writers, and actors of {title}:\n{meta}\nPlot of the movie {title}:\n{plot}\nCan you write a review for the movie {title}:\n"""
        #
        #     # elif review != '':
        #     #     context_tokens = f"""I will give you information of movie {title}.\nGenres, directors, writers, and actors of {title}:\n{meta}\nCan you write a review for the movie {title}:\n{review}"""
        #     # elif plot != '':
        #     #     context_tokens = f"""Movie {title}.\nGenres, directors, writers, and actors of {title}:\n{meta}\nPlot of the movie {title}:\n{plot}"""
        #     instructions.append(review)
        #     labels.append('')
        #     train_new.append(True)

    return instructions, labels, train_new


def review_read_data(args, mode):
    if 'synthetic' in args.data_type:
        review_data_path = os.path.join(args.dataset_path, 'synthetic')
        with open(os.path.join(review_data_path, f'{args.data_type}.json'), 'r', encoding='utf-8') as f:
            review_data = json.load(f)

        instructions = [data['OUTPUT'] for data in review_data]
        labels = [data['INPUT'].split('I will give you a review of movie')[1].split('\n')[0].strip() for data in
                  review_data]
        train_new = [True for data in review_data]
        new_instructions = []
        for idx, instruction in enumerate(instructions):
            labels[idx] = labels[idx].replace('.', '')
            instruction = instruction.replace(labels[idx], '[BLANK]')
            title = labels[idx].split('(')[0].strip()
            year = labels[idx].split('(')[-1][:-1].strip()
            if not year.isdigit():
                year = ''
            instruction = instruction.replace(f"\"{title}\" ({year})", '[BLANK]')
            instruction = instruction.replace(f"\"{title.lower()}\" ({year})", '[BLANK]')
            instruction = instruction.replace(title, '[BLANK]')
            instruction = instruction.replace(title.lower(), '[BLANK]')
            instruction = instruction.replace(f"({year})", '')

            new_instructions.append(f"{instruction}\n\n Based on the conversation, guess the item for [BLANK].")

        instructions = new_instructions

    else:
        review_data_path = os.path.join(args.dataset_path, 'review')
        if mode == "train":
            with open(os.path.join(review_data_path, f'{args.data_type}.json'), 'r', encoding='utf-8') as f:
                review_data = json.load(f)
        elif mode == "test":
            with open(os.path.join(review_data_path, f'onlyReview_1.json'), 'r', encoding='utf-8') as f:
                review_data = json.load(f)
                review_data = review_data[:300]

        # Choose templates
        review_template = """I will give you a review of a movie.\nIn the review, the movie title is masked with %s.\nHere is the review:\n%s\n\nBased on the review, guess the movie title for [title] without extra explanations."""
        if args.TH:
            review_template = """I will give you a review of a movie.\nIn the review, the movie title is masked with %s.\nHere is the review:\n%s\n\nBased on the review, guess the movie title that the above review is discussing"""
        elif args.JW:
            review_template = """I will give you a review of a movie\nIn this review, the movie title is maksed with %s.\nAfter reading the review, guess the movie title for [title] by considering actor, genre, director, writer, and plot discussed in the review.\nHere is the review:\n%s"""

        instructions = [data['context_tokens'] for data in review_data]
        labels = [data['item'] for data in review_data]
        train_new = [True for data in review_data]

        if args.pretrain:
            review_template = """I will give you a review of a movie %s\n%s"""
            instructions = [review_template % (label, instruction) for instruction, label in zip(instructions, labels)]
            labels = ['' for label in labels]
        else:
            new_instructions = []
            for idx, instruction in enumerate(instructions):
                instruction = instruction.replace(labels[idx], '[title]')
                title = labels[idx].split('(')[0].strip()
                year = labels[idx].split('(')[-1][:-1].strip()
                if not year.isdigit():
                    year = ''
                instruction = instruction.replace(f"\"{title}\" ({year})", '[title]')
                instruction = instruction.replace(f"\"{title.lower()}\" ({year})", '[title]')
                instruction = instruction.replace(title, '[title]')
                instruction = instruction.replace(title.lower(), '[title]')
                instruction = instruction.replace(f"({year})", '')

                new_instructions.append(instruction)

            instructions = new_instructions

    return instructions, labels, train_new


def context_review_read_data(args, mode):
    review_data_path = os.path.join(args.dataset_path, 'review')
    if mode == "train":
        with open(os.path.join(review_data_path, f'contextreview2item.json'), 'r', encoding='utf-8') as f:
            review_data = json.load(f)
    elif mode == "test":
        with open(os.path.join(review_data_path, f'contextreview2item.json'), 'r', encoding='utf-8') as f:
            review_data = json.load(f)
            review_data = review_data[:300]

    instructions = [data['context_tokens'] for data in review_data]
    labels = [data['item'] for data in review_data]
    train_new = [True for data in review_data]

    return instructions, labels, train_new


def process_crs_data(datas, mode, args):
    instructions, labels, train_new = [], [], []
    explanations = []
    candidate_items, candidate_scores = [], []
    if mode == "train":
        new_idx = json.load(open(os.path.join(args.dataset_path, 'train_new_idx.json'), 'r', encoding='utf-8'))
    else:
        new_idx = [i for i in range(len(datas))]
    for idx, data in enumerate(datas):
        if 'train' in mode and idx not in new_idx and args.only_new is True:
            continue
        instructions.append(data['context_tokens'])
        labels.append(data['item'])
        train_new.append(idx in new_idx)

        if args.data_type == 'explanation':
            explanations.append(data['explanation'])
            # candidate_items.append(data['candidate_items'])
            # candidate_scores.append(data['candidate_scores'])
    return instructions, labels, train_new, explanations


def crs_read_data(datas, mode, args):
    instructions, labels, train_new = [], [], []
    candidate_items, candidate_scores = [], []
    target = "item"
    if "cot" in args.data_type and mode != "valid":
        cot_data_path = os.path.join(args.dataset_path, 'cot')
        with open(os.path.join(cot_data_path, f'{mode}_data_{args.data_type}.json'), 'r', encoding='utf-8') as f:
            datas = json.load(f)
    else:
        if args.train_response:
            target = "response"

    if mode == "train":
        new_idx = json.load(open(os.path.join(args.dataset_path, 'train_new_idx.json'), 'r', encoding='utf-8'))
    else:
        new_idx = [i for i in range(len(datas))]
    for idx, data in enumerate(datas):
        instructions.append(data['context_tokens'])
        labels.append(data[target])
        train_new.append(idx in new_idx)
        if args.data_type == 'rerank' and mode == 'test':
            candidate_items.append(data['candidate_items'])
            candidate_scores.append(data['candidate_scores'])

    return instructions, labels, train_new, candidate_items, candidate_scores


def review_read_pretrain_data(args):
    data_path = os.path.join(args.home, 'data', 'redial', 'review')
    if not os.path.exists(data_path): os.mkdir(data_path)
    with open(os.path.join(data_path, f'onlyReview_5_augment.json'), 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    instructions = dataset
    labels = ['' for i in dataset]
    train_new = [True for i in dataset]

    return instructions, labels, train_new  # [data['context_tokens'] for data in dataset]


def refined_review_read_pretrain_data(args):
    data_path = os.path.join(args.home, 'data', 'redial', 'review')
    if not os.path.exists(data_path): os.mkdir(data_path)
    with open(os.path.join(data_path, f'refinedReview_1.json'), 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    # instructions = ["The following passages consist of reviews for the film %s provided by users.\n%s" % (data['item'], data['context_tokens']) for data in dataset]
    instructions = [data['context_tokens'] for data in dataset]
    labels = ['' for i in dataset]
    train_new = [True for i in dataset]

    return instructions, labels, train_new  # [data['context_tokens'] for data in dataset]


def synthetic_dialog_read_pretrain_data(args):
    data_path = os.path.join(args.home, 'data', 'redial', 'synthetic')
    if not os.path.exists(data_path): os.mkdir(data_path)
    with open(os.path.join(data_path, f'synthetic_dialog_review_augment.json'), 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    instructions = dataset
    labels = ['' for i in dataset]
    train_new = [True for i in dataset]

    return instructions, labels, train_new


def review_passage_read_pretrain_data(args):
    if args.JW_type == 1 or args.JW_type == 2:
        data_path = os.path.join(args.home, 'data', 'redial', 'review')
        with open(os.path.join(data_path, f'{args.data_type}.json'), 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        labels = ['' for data in dataset]

        if args.JW_type == 1:
            instructions = [data['context_tokens'] for data in dataset]
        else:
            instructions = [f"Here is a review of a movie {data['item']}.\n{data['context_tokens']}" for data in
                            dataset]

        train_new = [True for i in dataset]
    elif args.JW_type == 3 or args.JW_type == 4:
        data_path = os.path.join(args.home, 'data', 'redial', 'passage')
        with open(os.path.join(data_path, f'meta_plot_refinedreview_3.json'), 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        labels = ['' for data in dataset if len(data['review_list']) > 0]

        if args.JW_type == 3:
            instructions = [data['review_list'][0] for data in dataset if len(data['review_list']) > 0]
        else:
            instructions = [f"Here is a review of a movie {data['item']}.\n{data['review_list'][0]}" for data in dataset
                            if len(data['review_list']) > 0]

        train_new = [True for data in dataset if len(data['review_list']) > 0]

    return instructions, labels, train_new  # [data['context_tokens'] for data in dataset]


def meta_read_pretrain_data(args):
    data_path = os.path.join(args.home, 'data', 'redial', 'passage')
    if not os.path.exists(data_path): os.mkdir(data_path)
    with open(os.path.join(data_path, f'objective.json'), 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    instructions = [data['context_tokens'] for data in dataset]
    labels = ['' for i in dataset]
    train_new = [True for i in dataset]

    return instructions, labels, train_new
