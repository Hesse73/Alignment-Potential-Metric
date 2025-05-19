# Alignment Potential Metric

Source code for our ICML'25 paper: *Larger or Smaller Reward Margins to Select Preferences for Alignment?*

## Data Metrics

To select dataset subsets using the data quality metrics, we need to compute the logprobs for the chosen & rejected responses.

To process an existing preference dataset and calculate the logprobs, run:
```bash
python cal_logps.py --model ${model} --dataset ${dataset} --save_to ${save_path}
```

Then you can select top-k subsets using various metrics:
```bash
python metric_selection.py --dataset ${dataset_path} --metric ${metric_name}
```

## Alignment Traning

We follow [SimPO](https://github.com/princeton-nlp/SimPO)'s recipe for alignment training.

You can replace the dataset path in their scripts with the previously selected dataset by various metrics.

## High-Quality Data Generation

For the "evolve-then-select" data generation, please check the `./generation/` directory:

- `evol_instruct.py`: evolve prompts given a dataset
- `gen_pair_data.py` & `merge_gen_annotate.py`: generate response data and annotate with reward model
- `process_dataset.py`: select dataset subsets with metrics
