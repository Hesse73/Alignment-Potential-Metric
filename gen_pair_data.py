import os
import argparse
import json
import torch
import multiprocessing
import numpy as np
import datasets
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoModelForSequenceClassification, AutoTokenizer

parser = argparse.ArgumentParser(description='Generate responses with vllm, then annotate with RM')
parser.add_argument('--prompt_file', type=str, default="", 
                    help="path to the prompt file, default is empty (i.e. ultrafeedback prompts)")
parser.add_argument('--prompt_size', type=int, default=10000, help="number of prompts to generate")
parser.add_argument('--additional_uf', action='store_true', help='use additional ultrafeedback prompts')
parser.add_argument('--model', type=str, default="google/gemma-2-9b-it")
parser.add_argument("--reward_model", type=str, default="RLHFlow/ArmoRM-Llama3-8B-v0.1", 
                    help="Path to reward model")
parser.add_argument('--temperature', type=float, default=0.8)
parser.add_argument('--top_p', type=float, default=0.95)
parser.add_argument('--max_tokens', type=int, default=4096)
parser.add_argument('--num_gen', type=int, default=5)
parser.add_argument('--devices', type=int, nargs='+', default=[0,1,2,3])
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--save_dir', type=str, default='datasets')
parser.add_argument('--filter_prompt', action='store_true', help='filter out invalid prompts')
parser.add_argument('--len_diff', type=int, default=5, help='tolerance of length difference between original and generated prompt')
parser.add_argument('--n_per_try', type=int, default=100)


if __name__ == "__main__":
    args = parser.parse_args()
    args.num_parallel=len(args.devices)
    print(args)
    np.random.seed(args.seed)
    # load prompts
    if args.prompt_file:
        prompt_file = json.load(open(args.prompt_file))
        save_dir = args.prompt_file.replace('.json', '_gen')
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        if args.filter_prompt:
            tokenizer = AutoTokenizer.from_pretrained(args.reward_model, use_fast=True)
            tokenized_length = lambda p: len(tokenizer.apply_chat_template([{"role": "user", "content": p}, 
                                                                            {"role": "assistant", "content": "RESPONSE"}]))
            # filter out invalid prompts (keep length*5 > orig_length and orig_length > length/5), and too long prompts
            for data in prompt_file:
                valid_prompts = [p for p in data['evolved_prompts'] if len(p)*args.len_diff > len(data['orig_prompt']) 
                                                                            and len(data['orig_prompt'])*args.len_diff > len(p)]
                data['valid_prompts'] = [p for p in valid_prompts if tokenized_length(p) < tokenizer.model_max_length] 
            print(f"Original num_prompts: {sum([len(data['evolved_prompts']) for data in prompt_file])}")
            print(f"Filtered num_prompts: {sum([len(data['valid_prompts']) for data in prompt_file])}")
            all_prompts = sum([data['valid_prompts'] for data in prompt_file], [])
            # save filtered prompts
            filename = args.prompt_file.replace('.json', '_filtered.json')
            with open(filename, 'w') as f: json.dump(prompt_file, f, indent=4)
            print(f"Filtered prompts saved to {filename}")
        else:
            all_prompts = sum([data['evolved_prompts'] for data in prompt_file], [])
            print(f"Loaded num_prompts: {sum([len(data['evolved_prompts']) for data in prompt_file])}")
    else:
        ufb_ds = datasets.load_dataset("HuggingFaceH4/ultrafeedback_binarized")
        if args.additional_uf:
            # use additional ultrafeedback prompts (different from the original 10000 prompts)
            sampled_prompts = np.random.choice(ufb_ds['train_prefs']['prompt'], args.prompt_size * 2, replace=False).tolist()
            all_prompts = sampled_prompts[args.prompt_size:]
            # check same prompts
            prev_ds = datasets.load_from_disk(os.path.join(args.save_dir, f"ultrafeedback_{args.prompt_size}/hf_dataset"))
            prev_prompts = prev_ds['prompt']
            assert prev_prompts == sampled_prompts[:args.prompt_size]
            save_dir = os.path.join(args.save_dir, f"ultrafeedback_{args.prompt_size}_additional")
            if not os.path.exists(save_dir): os.makedirs(save_dir)
        else:
            # use the original ultrafeedback prompts
            all_prompts = ufb_ds['train_prefs']['prompt']
            all_prompts = np.random.choice(all_prompts, args.prompt_size, replace=False)
            save_dir = os.path.join(args.save_dir, f"ultrafeedback_{args.prompt_size}")
            if not os.path.exists(save_dir): os.makedirs(save_dir)
    print("Files will be saved to", save_dir)
    # parallel worker
    prompt_per_worker = len(all_prompts) // args.num_parallel
    split_idxs = [i * prompt_per_worker for i in range(args.num_parallel)] + [len(all_prompts)]
    def gen_worker(worker_id):
        # set gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices[worker_id])
        # load model
        llm = LLM(model=args.model, dtype='float16', swap_space=32)
        tokenizer = llm.get_tokenizer()
        if tokenizer.model_max_length < args.max_tokens:
            print("Setting tokenzier max length to", args.max_tokens)
            tokenizer.model_max_length = args.max_tokens  # set max tokens (trained model by trl is set to 2048, which is too small)
        # get current worker's prompts
        cur_prompts = all_prompts[split_idxs[worker_id]:split_idxs[worker_id+1]]
        print(f'worker {worker_id} has {len(cur_prompts)} prompts')
        # apply chat template
        if 'gemma' in args.model:
            conversations = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], 
                            tokenize=False, add_generation_prompt=True) for prompt in cur_prompts]
        else:
            conversations = [tokenizer.apply_chat_template([{'role': 'system', 'content': 'You are a helpful assistant.'},
                                                            {'role': 'user', 'content': prompt}], 
                            tokenize=False, add_generation_prompt=True) for prompt in cur_prompts]
        # generate responses
        sampling_params = SamplingParams(n=args.num_gen,
                                        temperature=args.temperature, 
                                        top_p=args.top_p, 
                                        max_tokens=args.max_tokens, 
                                        logprobs=0,    # return the sampled token's logprob
                                        seed=args.seed,)
        generated = llm.generate(conversations, sampling_params)
        # get responses & logprobs
        ret_data = []
        for prompt, output in zip(cur_prompts, generated):
            gen_info = {
                'prompt': prompt,
                'all_generated_texts': [],
                'all_logps': [],
                'all_logp_norms': [],
            }
            no_response = False
            for out in output.outputs:
                logp = out.cumulative_logprob
                logprobs = out.logprobs  # list of logprob for each token
                if len(logprobs) == 0:
                    print('Warning: length = 0!', output, sep='\n')
                    no_response = True
                    break
                logp_norm = logp / len(logprobs)
                generated_text = out.text
                gen_info['all_generated_texts'].append(generated_text)
                gen_info['all_logps'].append(logp)
                gen_info['all_logp_norms'].append(logp_norm)
            if not no_response: ret_data.append(gen_info)
        return ret_data
    # parallel generation
    if args.num_parallel > 1:
        with multiprocessing.Pool(args.num_parallel) as pool:
            results = pool.map(gen_worker, range(args.num_parallel))
    else:
        print("Note: Single worker")
        results = [gen_worker(0)]
    # merge results
    all_responses = []
    for res in results: all_responses += res
    # save generated responses
    filename = os.path.join(save_dir, f"generated_responses.json")
    with open(filename, 'w') as f: json.dump(all_responses, f, indent=4)
    print(f"Generated responses saved to {filename}")
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
    filename = os.path.join(save_dir, f"annotated_responses.json")
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
    ds_dir = os.path.join(save_dir, f"hf_dataset")
    generated_ds.save_to_disk(ds_dir)
    print(f"Generated dataset saved to {ds_dir}")
    # print some statistics
    rm_data, logp_data = np.array(generated_ds['rm_margin']), np.array(generated_ds['logp_margin_norm_abs'])
    print(f"Reward Margin: mean={rm_data.mean():.3f}, std={rm_data.std():.3f}")
    print(f"Logprob Margin Abs: mean={logp_data.mean():.3f}, std={logp_data.std():.3f}")