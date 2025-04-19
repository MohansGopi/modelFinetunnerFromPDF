from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct", # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

def question(content):
    global model,tokenizer
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    messages = [
        {"role": "user", "content": content},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    outputs = tokenizer.batch_decode(model.generate(input_ids = inputs, max_new_tokens = len(content), use_cache = True,
                            temperature = 1.5, min_p = 0.1))

    return outputs