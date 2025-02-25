import os
import argparse
import json
import torch
import multiprocessing
import numpy as np
import datasets
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

parser = argparse.ArgumentParser(description='Merge generated prompts and annotate')
parser.add_argument("--reward_model", type=str, default="RLHFlow/ArmoRM-Llama3-8B-v0.1", 
                    help="Path to reward model")
parser.add_argument('--devices', type=int, nargs='+', default=[0,1,2,3])
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--save_dir', type=str, default='datasets')
parser.add_argument('--gen_seeds', type=int, nargs='+', default=[1,17,42,73,100])


if __name__ == "__main__":
    args = parser.parse_args()
    args.num_parallel=len(args.devices)
    args.num_gen = len(args.gen_seeds)
    print(args)
    # load generated prompts
    all_generated = []
    for seed in args.gen_seeds:
        filename = os.path.join(args.save_dir, f"generated_responses_seed{seed}.json")
        generated = json.load(open(filename))
        print(f"Loaded {len(generated)} generated responses from {filename}")
        all_generated.append(generated)
    all_responses = [dict(prompt=data['prompt'], all_generated_texts=[], 
                             all_logps=[], all_logp_norms=[]) for data in all_generated[0]]
    for generated in all_generated:
        assert len(generated) == len(all_responses)
        for i, data in enumerate(generated):
            assert all_responses[i]['prompt'] == data['prompt']
            all_responses[i]['all_generated_texts'].append(data['generated_text'])
            all_responses[i]['all_logps'].append(data['logp'])
            all_responses[i]['all_logp_norms'].append(data['logp_norm'])
    # save merged responses
    filename = os.path.join(args.save_dir, "all_responses.json")
    with open(filename, 'w') as f: json.dump(all_responses, f, indent=4)
    print(f"Merged responses saved to {filename}")
    # flatten responses for RM annotation
    flatten_responses = []
    for idx, data in enumerate(all_responses):
        for idx_2, response in enumerate(data['all_generated_texts']):
            flatten_responses.append({
                'prompt_index': idx,
                'response_index': idx_2,
                'prompt': data['prompt'],
                'response': response,
            })
    # parallel worker for RM annotation
    def annnotate_worker(worker_id):
        # set gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices[worker_id]) # str(worker_id)
        # load annotator model
        model = AutoModelForSequenceClassification.from_pretrained(args.reward_model, device_map="cuda", 
                                                           trust_remote_code=True, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(args.reward_model, use_fast=True)
        # get current worker's prompt-response pairs
        cur_pairs = flatten_responses[worker_id::args.num_parallel]
        print(f'worker {worker_id} has {len(cur_pairs)} pairs')
        # batch annotate
        for idx in tqdm(range(0, len(cur_pairs), args.batch_size)):
            batch_data = cur_pairs[idx:idx+args.batch_size]
            all_messages = []
            for data in batch_data:
                all_messages.append([{"role": "user", "content": data["prompt"]},
                                    {"role": "assistant", "content": data["response"]},])
            inputs = tokenizer.apply_chat_template(all_messages, return_tensors="pt", return_dict=True,
                                                padding=True, truncation=True).to("cuda")
            with torch.no_grad():
                output = model(**inputs)
                scores = output.score.cpu().float().tolist()
            
            for in_batch_idx in range(len(batch_data)):
                data_idx = idx + in_batch_idx
                cur_pairs[data_idx]['rm_score'] = scores[in_batch_idx]
        return cur_pairs
    # parallel annotation
    if args.num_parallel > 1:
        with multiprocessing.Pool(args.num_parallel) as pool:
            results = pool.map(annnotate_worker, range(args.num_parallel))
    else:
        results = [annnotate_worker(0)]
    # merge results
    all_results = []
    for res in results: all_results += res
    # back to original format
    for data in all_responses: data['all_rm_scores'] = [-1] * args.num_gen
    for data in all_results:
        prompt_idx, response_idx = data['prompt_index'], data['response_index']
        assert data['prompt'] == all_responses[prompt_idx]['prompt']
        assert data['response'] == all_responses[prompt_idx]['all_generated_texts'][response_idx]
        all_responses[prompt_idx]['all_rm_scores'][response_idx] = data['rm_score']
    # save results
    filename = os.path.join(args.save_dir, f"annotated_responses.json")
    with open(filename, 'w') as f: json.dump(all_responses, f, indent=4)
    print(f"Annotated responses saved to {filename}")
    # make preference dataset
    all_responses = json.load(open(filename))
    for data in all_responses:
        all_rm_scores = np.array(data['all_rm_scores'])
        rm_sorted = np.argsort(all_rm_scores)
        chosen_idx, rejected_idx = rm_sorted[-1], rm_sorted[0]
        chosen_text, rejected_text = data['all_generated_texts'][chosen_idx], data['all_generated_texts'][rejected_idx]
        text2template = lambda x: [{'role': 'user', 'content': data['prompt']}, {'role': 'assistant', 'content': x}]
        data['chosen'], data['rejected'] = text2template(chosen_text), text2template(rejected_text)
        data['chosen_logp_norms'], data['rejected_logp_norms'] = data['all_logp_norms'][chosen_idx], data['all_logp_norms'][rejected_idx]
        data['rm_margin'] = all_rm_scores[chosen_idx] - all_rm_scores[rejected_idx]
        data['logp_margin_norm'] = data['chosen_logp_norms'] - data['rejected_logp_norms']
        data['logp_margin_norm_abs'] = abs(data['logp_margin_norm'])
    generated_ds = datasets.Dataset.from_list(all_responses)
    print(generated_ds)
    # save dataset
    ds_dir = os.path.join(args.save_dir, f"hf_dataset")
    generated_ds.save_to_disk(ds_dir)
    print(f"Generated dataset saved to {ds_dir}")
    # print some statistics
    rm_data, logp_data = np.array(generated_ds['rm_margin']), np.array(generated_ds['logp_margin_norm_abs'])
    print(f"Reward Margin: mean={rm_data.mean():.3f}, std={rm_data.std():.3f}")
    print(f"Logprob Margin Abs: mean={logp_data.mean():.3f}, std={logp_data.std():.3f}")


