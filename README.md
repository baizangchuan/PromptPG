# PromptPG: Prompt Selection via Policy Gradient

![MWP](https://img.shields.io/badge/Task-Math_Word_Problems-green) ![TableQA](https://img.shields.io/badge/Task-TableQA-green) ![TabMWP](https://img.shields.io/badge/Dataset-TabMWP-blue) 

![In_Context_Learning](https://img.shields.io/badge/Model-In--Context_Learning-red) ![Policy Gradient](https://img.shields.io/badge/Model-Policy_Gradient-red) ![Chain-of-Thought](https://img.shields.io/badge/Model-RL-red) ![Chain-of-Thought](https://img.shields.io/badge/Model-Chain_of_Thought-red) ![GPT-3](https://img.shields.io/badge/Model-GPT--3-red)

Data and code for our ICLR 2023 Paper [Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning](https://arxiv.org/abs/2209.14610).

For more details, please refer to the project page with dataset exploration and visualization tools: https://promptpg.github.io.

:bell: If you have any questions or suggestions, please don't hesitate to let us know. You can directly email [Pan Lu](https://lupantech.github.io/) at UCLA using the email address lupantech@gmail.com, comment on the [Twitter](https://twitter.com/lupantech/status/1623039527026831361), or post an issue on this repository.


## :fire: Leaderboard :fire:

:bell: The leaderboard is continuously being updated. If you have any new results to contribute, please feel free to reach out to us.

😀 You are invited to contribute your results to the TabMWP test split! Please send your result scores to [this email](mailto:lupantech@gmail.com) or open a new issue at the [github repository](https://github.com/lupantech/PromptPG/issues).

| **#** | **Model**                  | **Table** | **Method**   | **Type** | **Source**                                                | **Date** | **Avg**   | **FREE** | **MC** | **INT** | **DEC** | **EXTR** | **BOOL** | **OTH** |
| ----- | -------------------------- | --------- | ------------ | -------- | --------------------------------------------------------- | -------- | --------- | -------- | ------ | ------- | ------- | -------- | -------- | ------- |
| *     | **Human Performance**      | Image     | -            | -        | [Link](https://arxiv.org/abs/2209.14610)                  | 22-09-29 | **90.22** | 84.61    | 93.32  | 84.95   | 83.29   | 97.18    | 88.69    | 96.20   |
| 1     | **Chameleon (GPT-4) 🥇**    | Text-GT   | Few-shot     | Tool     | [Link](https://arxiv.org/abs/2304.09842)                  | 23-04-19 | **98.78** | 98.95    | 98.29  | 99.34   | 97.42   | 98.58    | 98.56    | 93.33   |
| 2     | **PoT GPT-4 🥈**            | Text-GT   | Few-shot (4) | Code     | [Link](https://arxiv.org/abs/2304.09842)                  | 23-04-19 | **96.93** | 97.40    | 95.58  | 98.48   | 93.22   | 96.25    | 98.00    | 68.57   |
| 3     | **CREATOR (ChatGPT) 🥉**    | Text-GT   | Few-shot     | Tool     | [Link](https://arxiv.org/abs/2305.14318)                  | 23-05-23 | **94.7**  | -        | -      | -       | -       | -        | -        | -       |
| 4     | **Chameleon (ChatGPT)**    | Text-GT   | Few-shot     | Tool     | [Link](https://arxiv.org/abs/2304.09842)                  | 23-04-19 | **93.28** | 93.13    | 93.72  | 92.71   | 94.76   | 91.29    | 98.11    | 78.85   |
| 5     | **TaCo (TAPEX-large)**     | Text-GT   | Fine-tuned   | CoT      | [Link](https://aclanthology.org/2023.findings-emnlp.734/) | 23-12-06 | **92.91** | 91.69    | 93.47  | 92.54   | 88.41   | 96.05    | 91.44    | 86.67   |
| 6     | **PoT ChatGPT + Doc**      | Text-GT   | Zero-shot    | Tool     | [Link](https://arxiv.org/abs/2308.00675)                  | 23-08-01 | **92.69** | -        | -      | -       | -       | -        | -        | -       |
| 7     | **CoT GPT-4**              | Text-GT   | Few-shot (8) | CoT      | [Link](https://arxiv.org/abs/2304.09842)                  | 23-04-19 | **90.81** | 88.48    | 97.49  | 86.16   | 97.51   | 96.86    | 99.11    | 89.52   |
| 8     | **CoS-Planning (ChatGPT)** | Text-GT   | Few-shot     | Tool     | [Link](https://arxiv.org/abs/2310.05155)                  | 23-10-08 | **90.00** | -        | -      | -       | -       | -        | -        | -       |
| 9     | **PoT ChatGPT**            | Text-GT   | Few-shot (4) | Code     | [Link](https://arxiv.org/abs/2304.09842)                  | 23-04-19 | **89.49** | 90.24    | 87.35  | 89.31   | 93.82   | 92.10    | 85.89    | 55.24   |
| 10    | **BM25 (ChatGPT)**         | Text-GT   | Few-shot     | Tool     | [Link](https://arxiv.org/abs/2309.17428)                  | 23-09-29 | **89.2**  | -        | -      | -       | -       | -        | -        | -       |
| 11    | **CRITIC (ChatGPT)**       | Text-GT   | Few-shot     | Tool     | [Link](https://arxiv.org/abs/2305.11738)                  | 23-09-30 | **89.0**  | -        | -      | -       | -       | -        | -        | -       |
| 12    | **RetICL (Codex)**         | Text-GT   | Few-shot     | CoT      | [Link](https://arxiv.org/abs/2305.14502)                  | 23-05-23 | **88.51** | -        | -      | -       | -       | -        | -        | -       |
| 13    | **CRAFT (ChatGPT)**        | Text-GT   | Few-shot     | Tool     | [Link](https://arxiv.org/abs/2309.17428)                  | 23-09-29 | **88.4**  | -        | -      | -       | -       | -        | -        | -       |
| 14    | **CRITIC (GPT-3)**         | Text-GT   | Few-shot     | Tool     | [Link](https://arxiv.org/abs/2305.11738)                  | 23-09-30 | **87.6**  | -        | -      | -       | -       | -        | -        | -       |
| 15    | **TaCo (TAPEX-base)**      | Text-GT   | Fine-tuned   | CoT      | [Link](https://aclanthology.org/2023.findings-emnlp.734/) | 23-12-06 | **86.12** | 85.53    | 85.74  | 85.29   | 86.44   | 93.31    | 77.89    | 81.90   |
| 16    | **SimCSE (ChatGPT)**       | Text-GT   | Few-shot     | Tool     | [Link](https://arxiv.org/abs/2309.17428)                  | 23-09-29 | **83.8**  | -        | -      | -       | -       | -        | -        | -       |
| 17    | **CoT ChatGPT**            | Text-GT   | Few-shot (8) | CoT      | [Link](https://arxiv.org/abs/2304.09842)                  | 23-04-19 | **82.03** | 78.43    | 92.32  | 75.38   | 90.30   | 92.30    | 92.89    | 87.62   |
| 18    | **PoT-SC Codex**           | Text-GT   | Few-shot (4) | Code     | [Link](https://arxiv.org/abs/2211.12588)                  | 22-11-22 | **81.8**  | 79.5     | 88.4   | 77.1    | 88.9    | 88.7     | 92.7     | 48.6    |
| 19    | **SEGSBS-PAL (Codex)**     | Text-GT   | Few-shot     | Code     | [Link](https://arxiv.org/abs/2305.00633)                  | 23-05-01 | **80.9**  | -        | -      | -       | -       | -        | -        | -       |
| 20    | **CRITIC (LLaMA-2-70B)**   | Text-GT   | Few-shot     | Tool     | [Link](https://arxiv.org/abs/2305.11738)                  | 23-09-30 | **75.0**  | -        | -      | -       | -       | -        | -        | -       |
| 21    | **ToRA (70B)**             | Text-GT   | -            | Tool     | [Link](https://arxiv.org/abs/2309.17452)                  | 23-09-29 | **74.0**  | -        | -      | -       | -       | -        | -        | -       |
| 22    | **ToRA-Code (34B)**        | Text-GT   | -            | Code     | [Link](https://arxiv.org/abs/2309.17452)                  | 23-09-29 | **70.5**  | -        | -      | -       | -       | -        | -        | -       |
| 23    | **CoT GPT-3 + PromptPG**   | Text-GT   | Few-shot (2) | CoT      | [Link](https://arxiv.org/abs/2209.14610)                  | 22-09-29 | **68.23** | 66.17    | 74.11  | 64.12   | 74.16   | 76.19    | 72.81    | 65.71   |
| 24    | **ToRA-Code (13B)**        | Text-GT   | -            | Code     | [Link](https://arxiv.org/abs/2309.17452)                  | 23-09-29 | **65.4**  | -        | -      | -       | -       | -        | -        | -       |
| 25    | **CodeLLaMA (PAL) (34B)**  | Text-GT   | -            | Code     | [Link](https://arxiv.org/abs/2309.17452)                  | 23-09-29 | **63.1**  | -        | -      | -       | -       | -        | -        | -       |
| 26    | **CoT GPT-3**              | Text-GT   | Few-shot (2) | CoT      | [Link](https://arxiv.org/abs/2209.14610)                  | 22-09-29 | **60.76** | 69.09    | 60.04  | 63.58   | 76.49   | 61.19    | 67.30    | 62.92   |
| 27    | **CodeLLaMA (PAL) (13B)**  | Text-GT   | -            | Code     | [Link](https://arxiv.org/abs/2309.17452)                  | 23-09-29 | **59.5**  | -        | -      | -       | -       | -        | -        | -       |
| 28    | **LLaMA-2 (PAL) (70B)**    | Text-GT   | -            | Code     | [Link](https://arxiv.org/abs/2309.17452)                  | 23-09-29 | **59.5**  | -        | -      | -       | -       | -        | -        | -       |
| 29    | **TAPEX_Large**            | Text-GT   | Fine-tuned   | PLM      | [Link](https://arxiv.org/abs/2209.14610)                  | 22-09-29 | **58.52** | 51.00    | 80.02  | 59.92   | 16.31   | 95.34    | 64.00    | 73.33   |
| 30    | **CoT GPT-3**              | Text-GT   | Zero-shot    | CoT      | [Link](https://arxiv.org/abs/2209.14610)                  | 22-09-29 | **57.61** | 54.36    | 66.92  | 55.82   | 48.67   | 78.82    | 55.67    | 51.43   |
| 31    | **LLaMA-2 (70B)**          | Text-GT   | -            | -        | [Link](https://arxiv.org/abs/2309.17452)                  | 23-09-29 | **57.5**  | -        | -      | -       | -       | -        | -        | -       |
| 32    | **UnifiedQA_Large**        | Text-GT   | Fine-tuned   | PLM      | [Link](https://arxiv.org/abs/2209.14610)                  | 22-09-29 | **57.35** | 48.67    | 82.18  | 55.97   | 20.26   | 94.63    | 68.89    | 79.05   |
| 33    | **GPT-3**                  | Text-GT   | Few-shot (2) | CoT      | [Link](https://arxiv.org/abs/2209.14610)                  | 22-09-29 | **57.13** | 54.69    | 64.11  | 58.36   | 40.40   | 75.95    | 52.41    | 53.02   |
| 34    | **GPT-3**                  | Text-GT   | Zero-shot    | CoT      | [Link](https://arxiv.org/abs/2209.14610)                  | 22-09-29 | **56.96** | 53.57    | 66.67  | 55.55   | 45.84   | 78.22    | 55.44    | 54.29   |
| 35    | **ToRA-Code (7B)**         | Text-GT   | -            | Code     | [Link](https://arxiv.org/abs/2309.17452)                  | 23-09-29 | **51.6**  | -        | -      | -       | -       | -        | -        | -       |
| 36    | **TAPEX_Base**             | Text-GT   | Fine-tuned   | PLM      | [Link](https://arxiv.org/abs/2209.14610)                  | 22-09-29 | **48.27** | 39.59    | 73.09  | 46.85   | 11.33   | 84.19    | 61.33    | 69.52   |
| 37    | **CodeLLaMA (PAL) (7B)**   | Text-GT   | -            | Code     | [Link](https://arxiv.org/abs/2309.17452)                  | 23-09-29 | **47.3**  | -        | -      | -       | -       | -        | -        | -       |
| 38    | **ToRA (13B)**             | Text-GT   | -            | Tool     | [Link](https://arxiv.org/abs/2309.17452)                  | 23-09-29 | **47.2**  | -        | -      | -       | -       | -        | -        | -       |
| 39    | **UnifiedQA_Base**         | Text-GT   | Fine-tuned   | PLM      | [Link](https://arxiv.org/abs/2209.14610)                  | 22-09-29 | **43.52** | 34.02    | 70.68  | 40.74   | 7.90    | 84.09    | 55.67    | 73.33   |
| 40    | **ToRA (7B)**              | Text-GT   | -            | Tool     | [Link](https://arxiv.org/abs/2309.17452)                  | 23-09-29 | **42.4**  | -        | -      | -       | -       | -        | -        | -       |
| 41    | **LLaMA-2 (13B)**          | Text-GT   | -            | -        | [Link](https://arxiv.org/abs/2309.17452)                  | 23-09-29 | **39.5**  | -        | -      | -       | -       | -        | -        | -       |
| 42    | **LLaMA-2 (7B)**           | Text-GT   | -            | -        | [Link](https://arxiv.org/abs/2309.17452)                  | 23-09-29 | **31.1**  | -        | -      | -       | -       | -        | -        | -       |
| 43    | **UnifiedQA_Small**        | Text-GT   | Fine-tuned   | PLM      | [Link](https://arxiv.org/abs/2209.14610)                  | 22-09-29 | **22.27** | 51.31    | 27.27  | 2.83    | 52.28   | 48.11    | 69.52    | 29.79   |
| 44    | **TAPEX_Large**            | Text-GT   | Pre-trained  | PLM      | [Link](https://arxiv.org/abs/2209.14610)                  | 22-09-29 | **18.59** | 8.80     | 46.59  | 10.62   | 1.72    | 46.91    | 48.11    | 30.48   |
| 45    | **UnifiedQA_Large**        | Text-GT   | Pre-trained  | PLM      | [Link](https://arxiv.org/abs/2209.14610)                  | 22-09-29 | **15.96** | 4.48     | 48.80  | 5.19    | 1.72    | 48.33    | 50.33    | 40.00   |
| 46    | **TAPEX_Base**             | Text-GT   | Pre-trained  | PLM      | [Link](https://arxiv.org/abs/2209.14610)                  | 22-09-29 | **15.73** | 7.32     | 39.76  | 8.68    | 2.06    | 35.06    | 47.11    | 20.95   |
| 47    | **UnifiedQA_Base**         | Text-GT   | Pre-trained  | PLM      | [Link](https://arxiv.org/abs/2209.14610)                  | 22-09-29 | **14.56** | 4.60     | 43.02  | 5.28    | 1.97    | 37.08    | 50.11    | 38.10   |
| 48    | **UnifiedQA_Small**        | Text-GT   | Pre-trained  | PLM      | [Link](https://arxiv.org/abs/2209.14610)                  | 22-09-29 | **12.18** | 1.18     | 43.62  | 1.37    | 0.43    | 38.70    | 49.78    | 37.14   |
| *     | **Heuristic Guess**        | -         | -            | -        | [Link](https://arxiv.org/abs/2209.14610)                  | 22-09-29 | **15.29** | 6.71     | 39.81  | 8.37    | 0.26    | 30.80    | 51.22    | 26.67   |

**Table formats**

- **Image**: taking the image format of the table as the input

- **Text-GT**: taking the textual ground truth parsed format of the table as the input

**Model types**

- **PLM**: pre-trained language model
- **CoT**: chain-of-thought prompting large language mode
- **Code**: code-augmented large language model
- **Tool**: tool-augmented large langauge model

**Accuracies for different question types:**

- **Avg**: all problems (reporting the average accuracy)
- **FREE**: free-text questions
- **MC**: multi-choice questions
- **INT**: questions with integer answers
- **DEC**: questions with decimal answers
- **EXTR**: questions with extractive text answers
- **BOOL**: questions with Boolean text answers
- **OTH**: questions with other text answers


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
