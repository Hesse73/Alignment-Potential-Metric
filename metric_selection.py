import argparse
import datasets
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="SimPO/processed_datasets/llama-3-8b-instruct-v2")
parser.add_argument("--save_to", type=str, default="")
parser.add_argument("--metric", type=str, default="rm-logp", choices=["rand", "rm_margin", "logp_margin_norm_abs", "rm-logp", "rm-xlogp"])
parser.add_argument("--logp_weight", type=float, default=1.0)
parser.add_argument("--select_ratio", type=float, default=0.4)


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
    return topk_idx

if __name__ == "__main__":
    np.random.seed(42)
    args = parser.parse_args()
    if args.save_to == "":
        args.save_to = f"{args.dataset}-{args.metric}"
    print(f"Will save to {args.save_to}")
    # load dataset
    ds = datasets.load_from_disk(args.dataset)
    select_size = int(args.select_ratio * len(ds))
    # top-k selection
    selected_idx = get_top_score_idx(ds, select_by=args.metric, k=select_size)
    train_ds = ds.select(selected_idx)
    train_ds = train_ds.select_columns(['prompt', 'chosen', 'rejected'])
    train_ds_to_save = datasets.DatasetDict(dict(train=train_ds))
    train_ds_to_save.save_to_disk(args.save_to)

