import json
import os


def read_data(args, mode):
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
    return data_samples
