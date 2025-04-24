# 07_evaluate_model.py

import torch
import torch.nn.functional as F
import pickle
import re
from torch import nn
from tqdm import tqdm
import pandas as pd

def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"(\w)'(\w)", r"\1_APOS_\2", text)
    for char in '"#$%&()*+,-./:;<=>?@[\\]^_`{|}~':
        text = text.replace(char, ' ')
    text = text.replace('_APOS_', "'")
    return [t for t in text.split() if t]

# Load vocab
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
word2idx = {word: idx for idx, word in enumerate(vocab)}

# Load embedding matrix
embedding_matrix = torch.load("embedding_matrix.pt")

# Define query encoder
class RNNEncoder(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()
        _, embedding_dim = embedding_matrix.size()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.rnn = nn.GRU(embedding_dim, 128, batch_first=True)

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        _, hidden = self.rnn(embedded)
        return hidden.squeeze(0)

# Load tokenized test queries
with open("tokenizedjson/test_tokenized.pkl", "rb") as f:
    test_data = pickle.load(f)
queries = test_data["tokenized_queries"]
query_texts = test_data.get("query_ids", [f"query_{i}" for i in range(len(queries))])

# Load passage lookup
df_passages = pd.read_csv("ms_marco_passages_cleaned.csv")  # confirm this exists
passages = df_passages["passage_text"].tolist()

# Load precomputed document vectors
all_doc_encodings = torch.load("full_corpus_doc_encodings.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_doc_encodings = all_doc_encodings.to(device)

# Load trained query encoder
query_encoder = RNNEncoder(embedding_matrix)
query_encoder.load_state_dict(torch.load("models/best_encoder_q.pt"))
query_encoder.eval().to(device)

# Retrieval
print("\n🔍 Running Top-5 Retrieval...")
for i in range(5):
    tokens = queries[i]
    indices = torch.tensor([word2idx.get(token, 0) for token in tokens], dtype=torch.long).unsqueeze(0).to(device)
    query_vec = query_encoder(indices)
    sims = F.cosine_similarity(query_vec, all_doc_encodings)
    top_k = torch.topk(sims, 5).indices.tolist()

    print(f"\nQuery {i} ({query_texts[i]}):")
    print(f"  Tokens: {tokens[:10]}...")
    print("  Top-5 Doc Indexes:", top_k)
    for j, idx in enumerate(top_k):
        doc_text = passages[idx][:300].replace("\n", " ")
        print(f"   [{j+1}] Similarity: {sims[idx].item():.4f}\n        Passage: {doc_text}...")

# Token coverage check
total, matched, missing = 0, 0, set()
for tokens in queries:
    for token in tokens:
        total += 1
        if token in word2idx: matched += 1
        else: missing.add(token)

print(f"✅ Token Coverage: {matched}/{total} ({matched/total:.2%})")
print(f"🚫 Missing Tokens Sample: {list(missing)[:10]}")
