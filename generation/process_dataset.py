import os
import datasets
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--train_dss", type=str, default=["datasets/ultrafeedback_10000/evolved_prompts_full_gen/hf_dataset"], nargs="+")
parser.add_argument("--save_names", type=str, default=[""], nargs="+")
parser.add_argument("--train_size", type=int, default=10_000)
parser.add_argument("--select_by", type=str, default="rand")
parser.add_argument("--logp_weight", type=float, default=2.5)
parser.add_argument("--default_ds", type=str, default="datasets/ultrafeedback_10000/hf_dataset")
parser.add_argument("--save_dir", type=str, default="processed_datasets")


def save_ds(train_ds, test_ds, name):
    ds = datasets.DatasetDict({'train': train_ds,'test': test_ds})
    os.makedirs(name, exist_ok=True)
    ds.save_to_disk(name)
    print(ds)
    print('saved to', name)

def check_rm_range(ds):
    rm_data = np.array(ds['rm_margin'])
    print(f'rm_margin = 0: {np.sum(rm_data == 0)}')


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
        print(f'Using logp_weight: {args.logp_weight}')
        score_data = rm_data_zscore - args.logp_weight * logp_data_zscore
    elif select_by == 'rm_margin':
        score_data = rm_data_zscore  # NOTE: use zscore for compatibility with rm-logp
    elif select_by == 'logp_margin_norm_abs':
        score_data = -logp_data_zscore  # NOTE: negative logp margin is better
    else:
        raise NotImplementedError
    topk_idx = np.argsort(score_data)[-k:]
    print(f'Avg score: {score_data.mean():.3f}, Topk score: {score_data[topk_idx].mean():.3f}')
    potential_score = rm_data_zscore - logp_data_zscore
    print(f'Avg potential score: {potential_score[topk_idx].mean():.3f}')
    return topk_idx


# NOTE: This is dirty, but we have to keep the same dataset sizes for the sake of comparison
# and add test dataset for compatability with the original script
args = parser.parse_args()
# load test dataset
test_ds = datasets.load_dataset("HuggingFaceH4/ultrafeedback_binarized")['test_prefs']
test_ds = test_ds.select_columns(['prompt', 'chosen', 'rejected'])
# load default train dataset
default_train_ds = datasets.load_from_disk(args.default_ds)
# print(f"RM range checking for default dataset"); check_rm_range(default_train_ds)
default_train_ds = default_train_ds.select_columns(['prompt', 'chosen', 'rejected'])
save_ds(default_train_ds, test_ds, os.path.join(args.save_dir, f"default_{args.train_size}"))
# load additional train datasets (additional from uf, evolved datasets via different scores)
for train_ds_name, save_name in zip(args.train_dss, args.save_names):
    print(f"Convert {train_ds_name} to {save_name}")
    train_ds = datasets.load_from_disk(train_ds_name)
    if len(train_ds) > args.train_size:
        np.random.seed(42)
        # selected_idx = np.random.choice(len(train_ds), args.train_size, replace=False)
        selected_idx = get_top_score_idx(train_ds, select_by=args.select_by, k=args.train_size)
        train_ds = train_ds.select(selected_idx)
        # print(f"RM range checking for {train_ds_name}"); check_rm_range(default_train_ds)
    train_ds = train_ds.select_columns(['prompt', 'chosen', 'rejected'])
    # concatenate with default train dataset
    train_ds = datasets.concatenate_datasets([default_train_ds, train_ds])
    save_ds(train_ds, test_ds, os.path.join(args.save_dir, save_name))

