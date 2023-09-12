import json
import sys
import torch
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
    def __init__(self, data_samples):
        self.data_samples = data_samples

    def __getitem__(self, idx):
        return self.data_samples[idx]

    def __len__(self):
        return len(self.data_samples)


def evaluate(
        instructions,
        tokenizer,
        prompter,
        model,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs):
    # prompt = prompter.generate_prompt(instruction, input)
    tokenizer.padding_side = 'left'
    inputs = tokenizer(instructions, padding=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )

    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
    }

    # Without streaming
    with torch.no_grad():
        # generation_output = model.generate(
        #     return_dict_in_generate=True,
        #     input_ids=input_ids,
        #     max_new_tokens=5,
        #
        # )
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences
    output = tokenizer.batch_decode(s, skip_special_tokens=True)
    return [prompter.get_response(i) for i in output]


def llama_test(
        args,
        instructions: list = None,
        labels: list = None,
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        prompt_template: str = "",  # The prompt template to use, will default to alpaca.
        server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
        share_gradio: bool = False,
):
    base_model = args.base_model
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(args, prompt_template)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
        ).to(args.device_id)
        # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

        # model = PeftModel.from_pretrained(
        #     model,
        #     lora_weights,
        #     torch_dtype=torch.float16,
        # )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # testing code for readme
    if instructions is None:
        instructions = [
            "The following multiple-choice quiz has 4 choices (a,b,c,d). Select the best answer from the given choices. Which film was scripted by Chris Buck? a) monty python and the holy grail (1975) b) winter soldier (1972) c) the net (1995) d) frozen (2013)\n"
        ]

    instructions = [prompter.generate_prompt(i) for i in instructions]
    instruction_dataset = Textdataset(instructions)
    dataloader = DataLoader(instruction_dataset, batch_size=args.batch_size, shuffle=False)

    generated_results = []
    for batch in tqdm(dataloader):
        responses = evaluate(batch, tokenizer, prompter, model)
        # print("Instruction:", instruction)
        # print("Response:", response)
        # print("#################################################")
        generated_results.extend(responses)
        for output, label in zip(responses, labels):
            args.log_file.write(json.dumps({'GEN': output, 'ANSWER': label}, ensure_ascii=False) + '\n')

    return generated_results


if __name__ == "__main__":
    # fire.Fire(main)
    args = parse_args()
    llama_test(args)
