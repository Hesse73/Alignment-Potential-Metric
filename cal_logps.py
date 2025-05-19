import os
import argparse
import torch
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--dataset", type=str, default="princeton-nlp/llama3-ultrafeedback-armorm")
parser.add_argument("--save_to", type=str, default="processed_datasets/llama-3-8b-instruct-v2-example")

def cal_logp(model, messages):
    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)

    # text to tokens
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
    full_ids = tokenizer.encode(full_text, return_tensors="pt", add_special_tokens=False).to(device)

    response_start_idx = prompt_ids.shape[1]

    # generate logits with LLM
    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits

    all_logprobs = logits.log_softmax(dim=-1)
    response_logprobs = all_logprobs[:, response_start_idx-1:-1]
    response_ids = full_ids[:, response_start_idx:]
    logprobs = torch.gather(response_logprobs, dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)
    logp, logp_norm = logprobs.sum().item(), logprobs.mean().item()
    return logp, logp_norm

if __name__ == "__main__":
    args = parser.parse_args()
    device='cuda'

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype='float16', device_map='cuda')
    if os.path.exists(args.dataset):
        ds = datasets.load_from_disk(args.dataset)
    else:
        ds = datasets.load_dataset(args.dataset)
    train_ds = ds['train']
    # train_ds = train_ds.select(range(100))

    def map_fn(example):
        chosen, rejected = example['chosen'], example['rejected']
        chosen_logps, chosen_logps_norm = cal_logp(model, chosen)
        rejected_logps, rejected_logps_norm = cal_logp(model, rejected)
        example['chosen_logps'] = chosen_logps
        example['chosen_logps_norm'] = chosen_logps_norm
        example['rejected_logps'] = rejected_logps
        example['rejected_logps_norm'] = rejected_logps_norm
        # margins
        rm_scores = example['all_rm_scores']
        example['rm_margin'] = max(rm_scores) - min(rm_scores)
        # logp_margin
        example['logp_margin_norm_abs'] = abs(chosen_logps_norm - rejected_logps_norm)
        return example

    new_ds = train_ds.map(map_fn)
    new_ds.save_to_disk(args.save_to)
