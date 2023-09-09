from transformers import AutoTokenizer
import transformers
import torch
HUGGINGFACE_AUTH_TOKEN = "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"

model = "meta-llama/Llama-2-7b-hf"
load_model = eval('transformers.AutoModelForCausalLM').from_pretrained(model, use_auth_token=HUGGINGFACE_AUTH_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(model)

input = "I like 'Breaking Bad' and 'Band of Brothers'. Do you have any recommendations of other shows I might like?\n"
inputs = tokenizer(input)
output_ids = load_model.generate(
            torch.as_tensor(inputs.input_ids).to(load_model.device),
            # temperature=temperature,
            max_new_tokens=128,
)
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )
#
# sequences = pipeline(
#     'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     max_length=200,
# )
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")