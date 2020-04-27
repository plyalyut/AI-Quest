from transformers import BertModel
import torch

# Work In Progress
from BERT.BertBiranker import BertBiranker

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained_model = BertModel.from_pretrained("bert-base-cased").to(DEVICE)

model = BertBiranker(pretrained_model, seq_length=1)

