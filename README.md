# Alignment Potential Metric

Source code for our ICML'25 paper: *Larger or Smaller Reward Margins to Select Preferences for Alignment?*

## Alignment Traning

We follow [SimPO](https://github.com/princeton-nlp/SimPO)'s recipe for alignment training.
Please clone their repository and follow their documentation to set up the environment and install necessary dependencies.

```shell
git clone https://github.com/princeton-nlp/SimPO.git
cd SimPO
# Follow SimPO's instructions to install requirements
```

After preparing the SimPO environment, you can replace the dataset path in their training scripts with the paths to datasets selected by various metrics, as detailed below.

## Data Metrics

Preference datasets provided by SimPO (e.g., [gemma2-ultrafeedback-armorm](https://huggingface.co/datasets/princeton-nlp/gemma2-ultrafeedback-armorm)) do not include pre-computed logprobs, which are requried to calculate implicit margins.

To ensure consistency with SimPO's data preprocessing, we have modified their scripts to calculate and save these logprobs during model inference.

The modified scripts are located in the `infer_scripts/` directory of this repository. 
To process a SimPO preference dataset and generate the required logprobs, run:
```bash
cp -r infer_scripts/ SimPO/
cd SimPO
bash infer_scripts/process.sh
```

Once the dataset has been processed and includes logprobs, you can select top-k subsets based on various metrics using our script:
```bash
python metric_selection.py --dataset ${dataset_path} --metric ${metric_name}
```

## High-Quality Data Generation

For the "evolve-then-select" data generation, please check the `./evol_select/` directory:

- `evol_instruct.py`: evolve prompts given a dataset
- `gen_pair_data.py` & `merge_gen_annotate.py`: generate response data and annotate with reward model
- `process_dataset.py`: select dataset subsets with metrics
