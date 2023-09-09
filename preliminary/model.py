from transformers import AutoTokenizer
import transformers
import torch

HUGGINGFACE_AUTH_TOKEN = "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"

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
pipeline = transformers.pipeline(
    "text-generation",
    model="codellama/CodeLlama-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto", use_auth_token=HUGGINGFACE_AUTH_TOKEN
)

sequences = pipeline(
    'def fibonacci(',
    do_sample=True,
    temperature=0.2,
    top_p=0.9,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=100,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")