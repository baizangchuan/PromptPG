from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch.nn as nn
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# class policy_network(nn.Module):

#     def __init__(self,
#                  model_config="bert-base-uncased",
#                  add_linear=False,
#                  embedding_size=128,#这个不知道多少合适
#                  freeze_encoder=True) -> None:
#         super().__init__()
#         self.tokenizer = AutoTokenizer.from_pretrained(model_config)
#         print("model_config:", model_config)
#         self.model = AutoModelForTokenClassification.from_pretrained(model_config)


#         # Freeze transformer encoder and only train the linear layer
#         if freeze_encoder:
#             for param in self.model.parameters():
#                 param.requires_grad = False

#         if add_linear:
#             # Add an additional small, adjustable linear layer on top of BERT tuned through RL
#             self.embedding_size = embedding_size
#             self.linear = nn.Linear(self.model.config.hidden_size,
#                                     embedding_size)  # 768 for bert-base-uncased, distilbert-base-uncased
#         else:
#             self.linear = None

#     def forward(self, input_list):
#         input = self.tokenizer(input_list, truncation=True, padding=True, return_tensors="pt").to(self.model.device)
#         # print(f"input: {input}")
#         output = self.model(**input, output_hidden_states=True)
#         # Get last layer hidden states
#         last_hidden_states = output.hidden_states[-1]
#         # Get [CLS] hidden states
#         sentence_embedding = last_hidden_states[:, 0, :]  # len(input_list) x hidden_size
#         # print(f"sentence_embedding: {sentence_embedding}")

#         if self.linear:
#             sentence_embedding = self.linear(sentence_embedding)  # len(input_list) x embedding_size

#         return sentence_embedding

class policy_network(nn.Module):
    def __init__(self,
                 model_config="/data/qq/Qwen",
                 add_linear=False,
                 embedding_size=128,  # 根据你的需求调整
                 freeze_encoder=True,
                gpu_base='1',
                gpu_embedding='2') -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_config, trust_remote_code=True)
        # 设置padding token为eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("model_config:", model_config)
        print(f"-----gpu_embedding: {gpu_embedding}")
        self.model = AutoModelForCausalLM.from_pretrained(model_config, device_map=f'cuda:{gpu_embedding}', trust_remote_code=True).eval()


        # Freeze transformer encoder and only train the linear layer
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

        if add_linear:
            # Add an additional small, adjustable linear layer on top of BERT tuned through RL
            self.embedding_size = embedding_size
            self.linear = nn.Linear(self.model.config.hidden_size, embedding_size)  # 768 for BERT-base-uncased, distilbert-base-uncased
        else:
            self.linear = None

    def forward(self, input_list):
        input = self.tokenizer(input_list, truncation=True, padding=True, return_tensors="pt").to(self.model.device)
        output = self.model(**input, output_hidden_states=True)
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]
        # Get [CLS] hidden states
        sentence_embedding = last_hidden_states[:, 0, :]  # len(input_list) x hidden_size

        if self.linear:
            sentence_embedding = self.linear(sentence_embedding)  # len(input_list) x embedding_size

        return sentence_embedding

# #这里的AutoModelForCausalLM和源代码采用的AutoModelForTokenClassification不同，看看俊这里是怎么得到embedding的
#         torch.manual_seed(1234)
#         os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         tokenizer = AutoTokenizer.from_pretrained("/data/qq/Qwen", trust_remote_code=True)
#         model = AutoModelForCausalLM.from_pretrained("/data/qq/Qwen", device_map='cuda', trust_remote_code=True).eval()

def test_policy_network():
    test_pids = [1]
    cand_pids = [0, 2, 4]
    problems = [
        "This is problem 0", "This is the first question", "Second problem is here", "Another problem",
        "This is the last problem"
    ]
    ctxt_list = [problems[pid] for pid in test_pids]
    cands_list = [problems[pid] for pid in cand_pids]

    model = policy_network(model_config="bert-base-uncased", add_linear=True, embedding_size=256)
    scores = model(ctxt_list, cands_list).cpu().detach().numpy()
    print(f"scores: {scores}")
    for i, test_pid in enumerate(test_pids):
        print(f"test_problem: {problems[test_pid]}")
        scores = scores[i, :].tolist()
        cand_rank = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        cand_pids = [cand_pids[cid] for cid in cand_rank]
        print(f"====== candidates rank: {[problems[pid] for pid in cand_pids]}")


if __name__ == "__main__":
    test_policy_network()
