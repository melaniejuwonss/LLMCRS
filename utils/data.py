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

    # tokenized_input = self.tokenizer(question, return_tensors="pt", padding=True, return_token_type_ids=False).to(
    #     self.args.device_id)
    # tokenized_output = self.tokenizer(answer, return_tensors="pt", padding=True, return_token_type_ids=False).to(
    #     self.args.device_id)
    for t_input, t_output in zip(question, answer):
        data_samples.append((t_input, t_output))
    if mode == "test":
        data_samples = data_samples[:100]
    return data_samples


def plot_read_data(args, mode):
    data_path = os.path.join(args.home, 'data', 'redial', 'plot')
    if not os.path.exists(data_path): os.mkdir(data_path)
    with open(os.path.join(data_path, f'plot_1.json'), 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset


def meta_plot_review_read_data(args, mode):
    data_path = os.path.join(args.home, 'data')
    if not os.path.exists(data_path): os.mkdir(data_path)
    with open(os.path.join(data_path, f'meta_plot_review.json'), 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset
