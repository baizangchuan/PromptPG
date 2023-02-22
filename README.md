# PromptPG: Prompt Selection via Policy Gradient

![MWP](https://img.shields.io/badge/Task-Math_Word_Problems-green) ![TableQA](https://img.shields.io/badge/Task-TableQA-green) ![TabMWP](https://img.shields.io/badge/Dataset-TabMWP-blue) 

![In_Context_Learning](https://img.shields.io/badge/Model-In--Context_Learning-red) ![Policy Gradient](https://img.shields.io/badge/Model-Policy_Gradient-red) ![Chain-of-Thought](https://img.shields.io/badge/Model-RL-red) ![Chain-of-Thought](https://img.shields.io/badge/Model-Chain_of_Thought-red) ![GPT-3](https://img.shields.io/badge/Model-GPT--3-red)

Data and code for our ICLR 2023 Paper [Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning](https://arxiv.org/abs/2209.14610).

For more details, please refer to the project page with dataset exploration and visualization tools: https://promptpg.github.io.

:bell: If you have any questions or suggestions, please don't hesitate to let us know. You can directly email [Pan Lu](https://lupantech.github.io/) at UCLA using the email address lupantech@gmail.com, comment on the [Twitter](https://twitter.com/lupantech/status/1623039527026831361), or post an issue on this repository.


## :fire: Leaderboard :fire:

:bell: The leaderboard is continuously being updated. If you have any new results to contribute, please feel free to reach out to us.

| **#** | **Method**                                 | **Sources**                                               | **Date**   | **FREE**  | **MC**    | **INT**   | **DEC**   | **EXTR**  | **BOOL**  | **OTH**   | **Avg**   |
| ----- | ------------------------------------------ | --------------------------------------------------------- | ---------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| 0     | **Human**                                  | [Pan et al., ICLR 2023](https://arxiv.org/abs/2209.14610) | 09/29-2022 | **84.61** | **93.32** | **84.95** | **83.29** | **97.18** | **88.69** | **96.20** | **90.22** |
| 1     | **Heuristic guess**                        | [Pan et al., ICLR 2023](https://arxiv.org/abs/2209.14610) | 09/29-2022 | 6.71      | 39.81     | 8.37      | 0.26      | 30.80     | 51.22     | 26.67     | 15.29     |
| 3     | **UnifiedQA_Small** (pre-trained)          | [Pan et al., ICLR 2023](https://arxiv.org/abs/2209.14610) | 09/29-2022 | 1.18      | 43.62     | 1.37      | 0.43      | 38.70     | 49.78     | 37.14     | 12.18     |
| 4     | **UnifiedQA_Base** (pre-trained)           | [Pan et al., ICLR 2023](https://arxiv.org/abs/2209.14610) | 09/29-2022 | 4.60      | 43.02     | 5.28      | 1.97      | 37.08     | 50.11     | 38.10     | 14.56     |
| 5     | **UnifiedQA_Large** (pre-trained)          | [Pan et al., ICLR 2023](https://arxiv.org/abs/2209.14610) | 09/29-2022 | 4.48      | 48.80     | 5.19      | 1.72      | 48.33     | 50.33     | 40.00     | 15.96     |
| 6     | **TAPEX_Base** (pre-trained)               | [Pan et al., ICLR 2023](https://arxiv.org/abs/2209.14610) | 09/29-2022 | 7.32      | 39.76     | 8.68      | 2.06      | 35.06     | 47.11     | 20.95     | 15.73     |
| 7     | **TAPEX_Large** (pre-trained)              | [Pan et al., ICLR 2023](https://arxiv.org/abs/2209.14610) | 09/29-2022 | 8.80      | 46.59     | 10.62     | 1.72      | 46.91     | 48.11     | 30.48     | 18.59     |
| 8     | **UnifiedQA_Small** (fine-tuned)           | [Pan et al., ICLR 2023](https://arxiv.org/abs/2209.14610) | 09/29-2022 | 22.27     | 51.31     | 27.27     | 2.83      | 52.28     | 48.11     | 69.52     | 29.79     |
| 9     | **UnifiedQA_Base** (fine-tuned)            | [Pan et al., ICLR 2023](https://arxiv.org/abs/2209.14610) | 09/29-2022 | 34.02     | 70.68     | 40.74     | 7.90      | 84.09     | 55.67     | 73.33     | 43.52     |
| 10    | **UnifiedQA_Large** (fine-tuned)           | [Pan et al., ICLR 2023](https://arxiv.org/abs/2209.14610) | 09/29-2022 | 48.67     | 82.18     | 55.97     | 20.26     | 94.63     | 68.89     | 79.05     | 57.35     |
| 11    | **TAPEX_Base** (fine-tuned)                | [Pan et al., ICLR 2023](https://arxiv.org/abs/2209.14610) | 09/29-2022 | 39.59     | 73.09     | 46.85     | 11.33     | 84.19     | 61.33     | 69.52     | 48.27     |
| 12    | **TAPEX_Large** (fine-tuned)               | [Pan et al., ICLR 2023](https://arxiv.org/abs/2209.14610) | 09/29-2022 | 51.00     | 80.02     | 59.92     | 16.31     | 95.34     | 64.00     | 73.33     | 58.52     |
| 13    | **Zero-shot GPT-3**                        | [Pan et al., ICLR 2023](https://arxiv.org/abs/2209.14610) | 09/29-2022 | 53.57     | 66.67     | 55.55     | 45.84     | 78.22     | 55.44     | 54.29     | 56.96     |
| 14    | **Zero-shot-CoT GPT-3**                    | [Pan et al., ICLR 2023](https://arxiv.org/abs/2209.14610) | 09/29-2022 | 54.36     | 66.92     | 55.82     | 48.67     | 78.82     | 55.67     | 51.43     | 57.61     |
| 15    | **Few-shot GPT-3** (2-shot)                | [Pan et al., ICLR 2023](https://arxiv.org/abs/2209.14610) | 09/29-2022 | 54.69     | 64.11     | 58.36     | 40.40     | 75.95     | 52.41     | 53.02     | 57.13     |
| 16    | **Few-shot-CoT GPT-3** (2-shot)            | [Pan et al., ICLR 2023](https://arxiv.org/abs/2209.14610) | 09/29-2022 | 60.76     | 69.09     | 60.04     | 63.58     | 76.49     | 61.19     | 67.30     | 62.92     |
| 17    | **Few-shot-CoT GPT-3 + PromptPG** (2-shot) | [Pan et al., ICLR 2023](https://arxiv.org/abs/2209.14610) | 09/29-2022 | 66.17     | 74.11     | 64.12     | 74.16     | 76.19     | 72.81     | **65.71** | 68.23     |
| 18    | **Few-shot PoT Codex** (4-shot)            | [Chen et al., arXiv](https://arxiv.org/abs/2211.12588)    | 11/22-2022 | **79.5**  | **88.4**  | **77.1**  | **88.9**  | **88.7**  | **92.7**  | 48.6      | **81.8**  |




## About PromptPG

Recent large pre-trained language models such as GPT-3 have achieved remarkable progress on mathematical reasoning tasks written in text form, such as math word problems (MWP). However, it is unknown if the models can handle more complex problems that involve math reasoning over heterogeneous information, such as tabular data. To fill the gap, we present **Tabular Math Word Problems (TabMWP)**, a new dataset containing 38,431 open-domain grade-level problems that require mathematical reasoning on both textual and tabular data.

![scienceqa](data/promptpg.png)

We evaluate different pre-trained models on TabMWP, including the GPT-3 model in a few-shot setting. As earlier studies suggest, since few-shot GPT-3 relies on the **selection of in-context examples**, its performance is unstable and can degrade to near chance. The unstable issue is more severe when handling complex problems like **TabMWP**. To mitigate this, we further propose a novel approach, **PromptPG**, which utilizes **policy gradient** to learn to **select in-context examples** from a small amount of training data and then constructs the corresponding prompt for the test example. Our proposed PromptPG is able to learn to select performing in-context examples via policy gradient when interacting with the GPT-3 API without any manually designed heuristics.

Experimental results show that our method outperforms the best baseline by **5.31%** on the accuracy metric and reduces the prediction variance significantly compared to random selection. For more details, you can find our project page [here](https://promptpg.github.io/) and our paper [here](https://arxiv.org/abs/2209.14610).



## The TabMWP Dataset

The **TabMWP** dataset contains 38,431 tabular math word problems. Each question in **TabMWP** is aligned with a tabular context, which is presented as an image, semi-structured text, and a structured table. There are two types of questions: *free-text* and *multi-choice*, and each problem is annotated with gold solutions to reveal the multi-step reasoning process. Two examples are shown below.

![domains](data/dataset.png)

The **TabMWP** dataset is provided in [`data/tabmwp`](https://github.com/lupantech/PromptPG/blob/main/data/tabmwp). For more details, you can explore the datatset and check out the [Explore](https://promptpg.github.io/explore.html) page and [Visualize](https://promptpg.github.io/visualize.html) page!



## Requirements

```
python==3.8.10
huggingface-hub==0.0.12
numpy==1.23.2
openai==0.23.0
pandas==1.4.3
torch==1.12.1+cu113
transformers==4.21.1
```

Install all required python dependencies:

```
pip install -r requirements.txt
```



## Run GPT-3 via PromptPG for TabMWP

The in-context examples can be randomly or retrieval-based selected from the training set. Recent research, however, has shown that few-shot GPT-3 can be highly unstable across different selections of in-context examples. We aim propose a novel approach, **PromptPG**, that can learn to select performing in-context examples using a policy gradient strategy, without brute-force searching or manually designed heuristics.

![algorithm](data/algorithm.png)

### Train the PromptPG Strategy

First, we train the policy network to select 2 (`shot_number=2`) examples from 20 (`cand_number=2`) canditate examples from the training data. With the selected two examples, we build the prompt input for GPT-3 API and get the predictions. The rewards are calulated over 160  (`train_number=2`) training examples and used to update the parameters of policy network.

```sh
cd run_gpt3_rl

python learn_policy.py \
--label exp1 \
--ckpt_root ../checkpoints \
--shot_number 2 \
--prompt_format TQ-SA \
--seed 2 \
--model_config bert-base-uncased \
--train_number 160 \
--cand_number 20 \
--lr 0.001 \
--epochs 20 \
--embedding_size 128 \
--batch_size 20 \
--gpu 0
```

###  Use PromptPG for Inference

We then use the learned policy network to select the in-context examples for the few-shot GPT-3 model on the test set.

```sh
python run_gpt3.py \
--label exp1 \
--ckpt_root ../checkpoints \
--model gpt3_rl \
--test_split test \
--test_number -1 \
--shot_number 2 \
--prompt_format TQ-SA \
--seed 2 \
--cand_number 20 \
--embedding_size 128 \
--model_config bert-base-uncased \
--ckpt exp1/ckpt_best_reward.pt \
--gpu 0
```

It will generate the predictions and save the results at `results/gpt3_rl/exp1_test_TQ-SA_2_seed_2.json`.

For the few-shot GPT-3 model, we repeated the experiments with three different random seeds, and reported the average accuracy in the paper.

### Evaluate the results

Our few-shot GPT-3 model via **PromptPG** achieves a state-of-the-art accuracy of 68.23% on the test split. One prediction example is visualized bellow.

![scienceqa](data/prediction.png)

We can get the accuracy metrics on average and across different question classes by running:

```sh
python evaluate_acc.py \
--data_file data/tabmwp/problems_test.json \
--result_root results/gpt3_rl \
--result_files exp1_test_TQ-SA_2_seed_2.json
```

We can repeat the experiments with three different random seeds, and report the average accuracy as the paper did.

```sh
python evaluate_acc.py \
--data_file data/tabmwp/problems_test.json \
--result_root results/gpt3_rl \
--result_files "trial1.json, trial2.json, trial3.json"
```



## Run GPT-3 Baselines

We compare the several GPT-3 baselines on **TabMWP** as follows.

### Zero-shot GPT-3

```sh
python run_gpt3.py \
--label exp1 \
--test_split test \
--test_number -1 \
--shot_number 0 \
--prompt_format TQ-A \
--seed 1
```

### Zero-shot CoT GPT-3

```sh
python run_gpt3_step.py \
--label exp2 \
--test_split test \
--test_number -1 \
--shot_number 0 \
--prompt_format TQ-A \
--seed 1
```

### Few-shot GPT-3

Run the few-shot GPT-3 model:

```sh
python run_gpt3.py \
--label exp3 \
--test_split test \
--test_number -1 \
--shot_number 2 \
--prompt_format TQ-A \
--seed 1
```

Similarly,  we repeat the experiments with three different random seeds, and report the average accuracy as the paper did.

```sh
python evaluate_acc.py \
--data_file data/tabmwp/problems_test.json \
--result_root results/gpt3 \
--result_files "trial1.json, trial2.json, trial3.json"
```

### Few-shot CoT GPT-3

Run the few-shot CoT GPT-3 model:

```sh
python run_gpt3.py \
--label exp4 \
--test_split test \
--test_number -1 \
--shot_number 2 \
--prompt_format TQ-SA \
--seed 1
```

Similarly,  we repeat the experiments with three different random seeds, and report the average accuracy as the paper did.

```sh
python evaluate_acc.py \
--data_file data/tabmwp/problems_test.json \
--result_root results/gpt3 \
--result_files "trial1.json, trial2.json, trial3.json"
```



## Run UnifiedQA

[UnifiedQA](https://github.com/allenai/unifiedqa) is one of the SOTA QA models. We developed both pre-trained and fine-tuned UnifiedQA baselines on **TabMWP**.

For the pre-trained baseline, we load the pre-trained checkpoint and evaluate UnifiedQA on **TabMWP**:

```sh
cd run_t5

python inference.py --label exp1_pretrained_unifiedqa \
--test_split test \
--test_num -1 \
--model small \
--gpu 0 

python eval.py --result_file t5/exp1_pretrained_unifiedqa_small.json
```

For the fine-tuned baseline,  we train UnifiedQA on the training set and evaluate it on the test set:

```sh
cd run_t5

## Training
python train.py --label exp4 \
--model small \
--batch_size 32 \
--eval_batch_size 32 \
--gpu 1

## Inference
python inference.py --label exp4 \
--test_split test \
--test_num -1 \
--model small \
--gpu 1 \
--check_point best_model.pth

python eval.py --result_file t5/exp4_small.json
```



## Run TAPEX

[TAPEX](https://github.com/microsoft/Table-Pretraining) is one of the SOTA TableQA models. We developed both pre-trained and fine-tuned TAPEX baselines on **TabMWP**.

For the pre-trained baseline, we load the pre-trained checkpoint and evaluate TAPEX on **TabMWP**:

```sh
cd run_tapex

## Pre-trained
python inference.py --label exp1 \
--test_split test \
--test_num -1 \
--model tapex-base \
--gpu 0

python eval.py --result_file tapex/exp1_tapex-base.json
```

For the fine-tuned baseline,  we train TAPEX on the training set and evaluate it on the test set:

```sh
cd run_tapex

## Training
python train.py --label exp2 \
--model tapex-base \
--batch_size 16 \
--eval_batch_size 16 \
--gpu 0 \
--save_all

## Inference
python inference.py --label exp2 \
--test_split test \
--test_num -1 \
--model tapex-base \
--gpu 0 \
--check_point best_model.pth

python eval.py --result_file tapex/exp2_tapex-base.json
```



## License

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

This work is licensed under a [MIT License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

[![License: CC BY-SA 4.0](https://camo.githubusercontent.com/bdc6a3b8963aa99ff57dfd6e1e4b937bd2e752bcb1f1936f90368e5c3a38f670/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4c6963656e73652d434325323042592d2d5341253230342e302d6c69676874677265792e737667)](https://creativecommons.org/licenses/by-sa/4.0/)

The **TabMWP** dataset is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).



## Cite

If the paper, codes, or the dataset inspire you, please cite us:

```latex
@inproceedings{lu2023dynamic,
  title={Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning},
  author={Lu, Pan and Qiu, Liang and Chang, Kai-Wei and Wu, Ying Nian and Zhu, Song-Chun and Rajpurohit, Tanmay and Clark, Peter and Kalyan, Ashwin},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
```
