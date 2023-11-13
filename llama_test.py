import json
import os
import sys
import torch
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel

from utils.prompter import Prompter
from utils.parser import parse_args

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class Textdataset(Dataset):
    def __init__(self, args, instructions, labels, tokenizer):
        self.args = args
        self.instructions = instructions
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # tokenizer.padding_side = 'left'
        # inputs = self.tokenizer(self.data_samples[idx], padding=True, return_tensors="pt", max_length=args.max_input_length, truncation=True)
        # input_ids = inputs["input_ids"].to(self.args.device_id)
        return self.instructions[idx], self.labels[idx]

    def __len__(self):
        return len(self.instructions)


class LLaMaEvaluator:
    def __init__(self, args, tokenizer, instructions: list = None, labels: list = None, negItems: list = None,
                 prompt_template_name: str = ""):
        self.args = args
        self.instructions = instructions
        self.labels = labels
        self.negItems = negItems
        self.tokenizer = tokenizer  # , LlamaTokenizer.from_pretrained(self.args.base_model)
        self.prompter = Prompter(args, prompt_template_name)
        self.new_idx = json.load(open(os.path.join(self.args.dataset_path, 'new_idx.json'), 'r', encoding='utf-8'))

        self.dataloader = self.prepare_dataloader()
        # self.model = self.prepare_model()

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def prepare_model(self,
                      base_model: str = "",
                      load_8bit: bool = False,
                      lora_weights: str = "",
                      server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
                      share_gradio: bool = False, ):
        print('prepare new model for evaluating')
        base_model = self.args.base_model
        if self.args.lora_weights != "":
            lora_weights = self.args.lora_weights

        assert (
            base_model
        ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

        if device == "cuda":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map='auto'
            )  # .to(self.args.device_id)

            # todo: For evaluating the PEFT model
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                base_model, device_map={"": device}, low_cpu_mem_usage=True
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )
        # unwind broken decapoda-research config
        model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        return model

    def prepare_dataloader(self):
        self.tokenizer.padding_side = 'left'

        instructions = [self.prompter.generate_prompt(instruction=instruction) for instruction in self.instructions]
        instruction_dataset = Textdataset(self.args, instructions, self.labels, self.tokenizer)
        dataloader = DataLoader(instruction_dataset, batch_size=self.args.eval_batch_size, shuffle=False)

        return dataloader

    def evaluate(self,
                 input_ids,
                 attention_mask,
                 model,
                 input=None,
                 temperature=0.1,
                 top_p=0.75,
                 top_k=40,
                 num_beams=4,  # todo: beam 1개로 바꿔보기
                 max_new_tokens=50,
                 **kwargs):
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        output = self.tokenizer.batch_decode(s, skip_special_tokens=True)
        return [self.prompter.get_response(i) for i in output]

    def test(self, model=None):
        if model is None:
            model = self.prepare_model()

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        hit, cnt = 0.0, 0.0
        idx = 0
        for batch in tqdm(self.dataloader, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            generated_results = []
            batched_inputs = self.tokenizer(batch[0], padding=True, return_tensors="pt")
            input_ids = batched_inputs["input_ids"].to(self.args.device_id)
            attention_mask = batched_inputs["attention_mask"].to(self.args.device_id)

            responses = self.evaluate(input_ids, attention_mask, model, max_new_tokens=self.args.max_new_tokens,
                                      num_beams=self.args.num_beams)
            labels = batch[1]
            # print("Instruction:", instruction)
            # print("Response:", response)
            # print("#################################################")
            # generated_results.extend(responses)
            for dialog, output, label in zip(batch[0], responses, labels):
                if 'quiz' in self.args.stage:
                    movie_name = label.replace('(', ')').split(')')[1].strip().lower()
                elif 'crs' in self.args.stage:
                    movie_name = label.split('(')[0].strip().lower()
                    check_response = output.lower()
                    if 'fineTuneCRS' in self.args.prompt:
                        check_response = check_response[check_response.rfind('therefore'):]
                    elif 'withCoT' in self.args.prompt:
                        check_response = check_response[check_response.rfind('\n'):]
                # if 'example' in self.args.rq_num or 'explain' in self.args.lora_weights:
                #     check_response = output[output.lower().find("answer:"):].lower()

                if movie_name in check_response.lower():
                    hit += 1.0

                cnt += 1.0
                hit_ratio = hit / cnt
                # args.log_file.write(json.dumps({'GEN': output, 'ANSWER': label, 'AVG_HIT': hit_ratio}, ensure_ascii=False) + '\n')
                generated_results.append({'GEN': output, 'ANSWER': label, 'AVG_HIT': hit_ratio,
                                          'NEW_ITEM': idx in self.new_idx})
                idx += 1

            if self.args.write:
                for i in generated_results:
                    self.args.log_file.write(json.dumps(i, ensure_ascii=False) + '\n')
            if cnt % 100 == 0 and cnt != 0:
                wandb.log({"hit_ratio": (hit / cnt)})
                print("%.4f" % (hit / cnt))

    # return generated_results

# if __name__ == "__main__":
#     # fire.Fire(main)
#     args = parse_args()
#     llama_test(args)
