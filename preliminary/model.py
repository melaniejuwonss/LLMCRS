# from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
# import torch
#
# MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
#
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)
# language_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
#
# text_generation_pipeline = TextGenerationPipeline(
#     model=language_model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.float16,
#     device=0,
# )


from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    device=0
)

sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
    device=0
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")