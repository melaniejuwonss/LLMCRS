import math
import os
import sys
from typing import List
import pandas as pd
import torch
import transformers
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments, TrainerState, TrainerControl
import wandb
from peft import PeftModel
from peft import IA3Config

from utils.parser import parse_args

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback

from utils.prompter import Prompter


# class CustomTrainer(Trainer):
#     def compute_metrics(self, eval_pred):
#         print(eval_pred)
#         # predictions, labels = eval_pred
#         # predictions = np.argmax(predictions, axis=-1)  # 클래스 ID로 변환
#         #
#         # # 디코딩 및 사용자 정의 평가 메트릭 계산 코드 작성
#         # # 여기에서는 예시로 정확도만 계산했지만, 여러분이 원하는 어떤 평가 메트릭도 추가할 수 있습니다.
#         # return {'accuracy': accuracy_score(labels, predictions)}

class QueryEvalCallback(TrainerCallback):
    def __init__(self, args, evaluator):
        self.log_name = args.log_name
        self.mode = args.mode
        self.evaluator = evaluator

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        epoch = state.epoch
        path = os.path.join(args.output_dir, self.log_name + '_E' + str(int(epoch)))
        if not os.path.isdir(path):
            os.makedirs(path)
        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     model = torch.compile(model)
        model.save_pretrained(path, safe_serialization=False)
        # trainer = kwargs['trainer']
        # logs = kwargs['logs']
        # model = kwargs['model']
        # print("==============================Evaluate step==============================")
        # # predictions, labels = trainer.predict(trainer.eval_dataset)
        # # print(predictions.size())
        # if 'test' in self.mode:
        #     self.evaluator.test(model, epoch)
        # model.train()
        # # print(kwargs)
        # print("==============================End of evaluate step==============================")


def llama_finetune(
        args,
        tokenizer,
        evaluator,
        instructions: list = None,
        labels: list = None,
        isNews: list = None,
        explanations: list = [],
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        # training hyperparams
        batch_size: int = 128,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        warmup_steps=100,
        val_set_size: int = 0,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        add_eos_token: bool = False,
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca_legacy",  # The prompt template to use, will default to alpaca.
):
    base_model = args.base_model
    batch_size = args.batch_size
    train_on_inputs = args.train_on_inputs
    learning_rate = args.learning_rate
    gradient_accumulation_steps = args.num_device  # update the model's weights once every gradient_accumulation_steps batches instead of updating the weights after every batch.
    per_device_train_batch_size = batch_size // args.num_device
    resume_from_checkpoint = args.lora_weights
    cutoff_len = args.cutoff
    if args.warmup != 0:
        max_train_steps = num_epochs * math.ceil(math.ceil(len(instructions) / batch_size) / gradient_accumulation_steps)
        warmup_steps = int(args.warmup * max_train_steps)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"per_device_train_batch_size: {per_device_train_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    # gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(args, prompt_template_name)

    device_map = "auto"
    # device_map = {
    #     "transformer.word_embeddings": args.device_id,
    #     "transformer.word_embeddings_layernorm": args.device_id,
    #     "lm_head": "cpu",
    #     "transformer.h": args.device_id,
    #     "transformer.ln_f": args.device_id,
    #     # "model.embed_tokens": args.device_id,
    #     "model": args.device_id
    # }

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print("world_size: %d" % world_size)
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # model = LlamaForCausalLM.from_pretrained(
    #     base_model,
    #     # load_in_8bit=load_8bit,
    #     torch_dtype=torch.float16,
    #     # device_map='auto'
    # ).to(args.device_id)
    # tokenizer = LlamaTokenizer.from_pretrained(base_model)

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            # max_length=cutoff_len + 100,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                # and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point, writeFlag=None):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
            data_point['isNew']
        )
        if writeFlag:
            args.score_file.write("==========First Training sample==========\n")
            args.score_file.write(f"{full_prompt}\n")

        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # , llm_int8_enable_fp32_cpu_offload=True)
    # if data_path.endswith(".json") or data_path.endswith(".jsonl"):
    #     data = load_dataset("json", data_files=data_path)
    # else:
    #     data = load_dataset(data_path)

    data = []
    if args.prompt == 'DI2E':
        for inst, lab, explanation, isNew in zip(instructions, labels, explanations, isNews):
            data.append({"instruction": inst, "input": lab, "output": explanation, "isNew": isNew})
    else:
        for inst, lab, isNew in zip(instructions, labels, isNews):
            data.append({"instruction": inst, "input": "", "output": lab, "isNew": isNew})

    first_sample = Dataset.from_pandas(pd.DataFrame([data[0]]))
    data = Dataset.from_pandas(pd.DataFrame(data))

    if val_set_size > 0:
        train_val = data.train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        generate_and_tokenize_prompt(first_sample[0], True)
        train_data = data.shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        # load_in_8bit=True,
        # llm_int8_enable_fp32_cpu_offload=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        quantization_config=quantization_config,
    )

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # if args.lora_weights[args.lora_weights.rfind('/') + 1:] != "lora-alpaca":
    #     model = PeftModel.from_pretrained(
    #         model,
    #         args.lora_weights,
    #         torch_dtype=torch.float16,
    #     )
    # else:
    #     model = get_peft_model(model, config)
    model = get_peft_model(model, config)

    if resume_from_checkpoint[resume_from_checkpoint.rfind('/') + 1:] != "lora-alpaca":
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")
    else:
        resume_from_checkpoint = None
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=5 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            # run_name=args.wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[QueryEvalCallback(args, evaluator)]
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    # fire.Fire(llama_finetune)
    args = parse_args()
    llama_finetune(args, num_epochs=args.epoch)
