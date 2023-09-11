from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import json


# model = "meta-llama/Llama-2-7b-hf"
CUDA_LAUNCH_BLOCKING = 1
# tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)
# tokenizer.add_special_tokens({'pad_token': '<pad>'})
# tokenizer.padding_side = "left"
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     torch_dtype=torch.float16,
#     # device_map="auto",
#     device=0
# )
# pipeline.tokenizer.add_special_tokens({'pad_token': '<pad>'})
# pipeline.tokenizer.padding_side = "left"
# pipeline.model.resize_token_embeddings(len(tokenizer))
#
# RQ_data = json.load((open('../data/RQ1.json', 'r', encoding='utf-8')))
# question, answer = [], []
# for data in RQ_data[:12]:
#     question.append(data['Question'])
#     answer.append(data['Answer'])
#
# sequences = pipeline(
#     question, # 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
#     do_sample=True,
#     top_k=10, # The number of answers to return (will be chosen by order of likelihood)
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     max_length=500,
#     batch_size=4
#     # device=0
# )
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")

##############################################
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
if tokenizer.pad_token is None:
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # model.resize_token_embeddings(len(tokenizer))
    # tokenizer.padding_side = "left"
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
# use different length sentences to test batching
sentences = [
    "Hello, my dog is a little",
    "Today, I",
    "I love",
    "Repeat after me: I love you.",
    "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
    "You are a human that uses a lot of expletives. For each one of your responses to the user you use expletives liberally. If you find yourself questioning whether it's appropriate or not you're to say fuck that noise and keep using your expletives regardless. Human: Tell me a story. You:"
]

# sentence = "Hello, my dog is a little"

inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(model.device)
print(inputs['input_ids'].shape)

output_sequences = model.generate(**inputs, max_new_tokens=20, do_sample=True, top_p=0.9)

print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))