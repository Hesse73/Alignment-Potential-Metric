import os
import argparse
import json
import multiprocessing
import datasets
import numpy as np
from vllm import LLM, SamplingParams
from prompts import evolve_strategies


parser = argparse.ArgumentParser(description='Select prompts and evolve')
parser.add_argument('--dataset_dir', type=str, default="datasets/ultrafeedback_10000/hf_dataset")
parser.add_argument('--model', type=str, default="google/gemma-2-9b-it")
parser.add_argument('--temperature', type=float, default=0.8)
parser.add_argument('--top_p', type=float, default=0.95)
parser.add_argument('--max_tokens', type=int, default=4096)
parser.add_argument('--devices', type=int, nargs='+', default=[0,1,2,3])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--select_by', type=str, default='rm-logp', choices=['rm-logp', 'rand', 'rm_margin', 'logp_margin_norm_abs', 'rm-xlogp'])
parser.add_argument('--logp_weight', type=float, default=2.5)
parser.add_argument('--select_ratio', type=float, default=0.25)
parser.add_argument('--evol_full', action='store_true')
parser.add_argument('--prompt_type', type=int, default=5)


def get_top_score_idx(ds, select_by='rm-logp', k=100):
    if select_by == 'rand': return np.random.choice(len(ds), k, replace=False)
    rm_data = np.array(ds['rm_margin'])
    logp_data = np.array(ds['logp_margin_norm_abs'])
    rm_mean, rm_std = rm_data.mean(), rm_data.std()
    logp_mean, logp_std = logp_data.mean(), logp_data.std()
    rm_data_zscore = (rm_data - rm_mean)/rm_std
    logp_data_zscore = (logp_data - logp_mean)/logp_std
    if select_by == 'rm-logp':
        score_data = rm_data_zscore - logp_data_zscore
    elif select_by == 'rm-xlogp':
        score_data = rm_data_zscore - args.logp_weight * logp_data_zscore
    elif select_by == 'rm_margin':
        score_data = rm_data_zscore  # NOTE: use zscore for compatibility with rm-logp
    elif select_by == 'logp_margin_norm_abs':
        score_data = -logp_data_zscore  # NOTE: negative logp margin is better
    else:
        raise NotImplementedError
    topk_idx = np.argsort(score_data)[-k:]
    print(f'Avg score: {score_data.mean():.3f}, Topk score: {score_data[topk_idx].mean():.3f}')
    return topk_idx


if __name__ == "__main__":
    args = parser.parse_args()
    args.num_parallel=len(args.devices)
    print(args)
    np.random.seed(args.seed)
    save_dir = os.path.dirname(args.dataset_dir)
    # prompt types (only use this when evol_full)
    if args.prompt_type < len(evolve_strategies) and args.evol_full:
        assert args.prompt_type >= 1
        print(f"****\nSet number of prompt types to {args.prompt_type}.\n****")
        evolve_strategies = evolve_strategies[:args.prompt_type]
    # load dataset
    print('loading', args.dataset_dir)
    generated_ds = datasets.load_from_disk(args.dataset_dir)
    if args.evol_full:
        selected_prompts = generated_ds['prompt']
    else:
        # calculate score & topk dataset
        num_topk = int(len(generated_ds) * args.select_ratio)
        top_score_idx = get_top_score_idx(generated_ds, select_by=args.select_by, k=num_topk)
        topk_ds = generated_ds.select(top_score_idx)
        selected_prompts = topk_ds['prompt']
    # evolve prompts
    prompts_per_worker = len(selected_prompts) // args.num_parallel
    split_idxs = [i * prompts_per_worker for i in range(args.num_parallel)] + [len(selected_prompts)]
    # parallel worker
    def evol_worker(worker_id):
        # set gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices[worker_id])
        # load model
        llm = LLM(model=args.model, dtype='float16')
        tokenizer = llm.get_tokenizer()
        # get current worker's prompts
        cur_prompts = selected_prompts[split_idxs[worker_id]:split_idxs[worker_id+1]]
        print(f'worker {worker_id} has {len(cur_prompts)} prompts')
        all_evolved_prompts = [{'orig_prompt': prompt, 'evolved_prompts': []} for prompt in cur_prompts]
        for evol_strategy in evolve_strategies:
            print(f"worker {worker_id} start to evolve prompts using {evol_strategy.__name__}")
            evol_instructions = [evol_strategy(prompt) for prompt in cur_prompts]
            # apply chat template
            if 'gemma' in args.model:
                conversations = [tokenizer.apply_chat_template([{'role': 'user', 'content': inst}], 
                                tokenize=False, add_generation_prompt=True) for inst in evol_instructions]
            else:
                conversations = [tokenizer.apply_chat_template([{'role': 'system', 'content': 'You are a helpful assistant.'},
                                                                {'role': 'user', 'content': inst}], 
                                tokenize=False, add_generation_prompt=True) for inst in evol_instructions]
            sampling_params = SamplingParams(n=1,  # only 1 output for each evolve strategy
                                            temperature=args.temperature, 
                                            top_p=args.top_p, 
                                            max_tokens=args.max_tokens,
                                            seed=args.seed,)
            generated = llm.generate(conversations, sampling_params)
            # get evolved prompts
            for idx, output in enumerate(generated):
                all_evolved_prompts[idx]['evolved_prompts'].append(output.outputs[0].text)
        return all_evolved_prompts
    # evolve prompts
    with multiprocessing.Pool(args.num_parallel) as p:
        all_evolved_prompts = p.map(evol_worker, range(args.num_parallel))
    # merge results
    all_evolved_prompts = sum(all_evolved_prompts, [])
    if args.evol_full:
        filename = os.path.join(save_dir, f"evolved_prompts_full_{args.prompt_type}types.json")
    else:
        # save results
        if args.select_by == 'rm-xlogp':
            args.select_by = f'rm-x{args.logp_weight}logp'
        filename = os.path.join(save_dir, f"evolved_prompts_{args.select_by}_{args.select_ratio}.json")
    with open(filename, 'w') as f: json.dump(all_evolved_prompts, f, indent=4)
    print(f"Saved evolved prompts to {filename}")
