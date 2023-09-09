from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import torch

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)
language_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

text_generation_pipeline = TextGenerationPipeline(
    model=language_model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device=0,
)