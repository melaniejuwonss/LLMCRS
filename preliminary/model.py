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

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf", use_auth_token=True)
pipeline = transformers.pipeline(
    "text-generation",
    model="codellama/CodeLlama-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    'def fibonacci(',
    do_sample=True,
    temperature=0.2,
    top_p=0.9,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=100,
    device=0,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")