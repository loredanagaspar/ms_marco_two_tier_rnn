# encode_corpus.py

import torch
import pickle
from tqdm import tqdm
import re
from torch import nn
from datasets import load_dataset

# RNN Encoder inline definition
class RNNEncoder(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()
        num_embeddings, embedding_dim = embedding_matrix.size()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.rnn = nn.GRU(embedding_dim, 128, batch_first=True)

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        _, hidden = self.rnn(embedded)
        return hidden.squeeze(0)

# Simple tokenizer
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

# Load pretrained word2vec embeddings
import gensim.downloader as api
glove_vectors = api.load("word2vec-google-news-300")
embedding_dim = glove_vectors.vector_size

embedding_matrix = []
for word in vocab:
    if word in glove_vectors:
        embedding_matrix.append(torch.tensor(glove_vectors[word]))
    else:
        embedding_matrix.append(torch.randn(embedding_dim))
embedding_matrix = torch.stack(embedding_matrix)

# Init encoder
doc_encoder = RNNEncoder(embedding_matrix)
doc_encoder.load_state_dict(torch.load("models/best_encoder_d.pt"))
doc_encoder.eval()
doc_encoder.to("cuda" if torch.cuda.is_available() else "cpu")
device = next(doc_encoder.parameters()).device

# Load MS MARCO passage corpus from HuggingFace
print("Loading MS MARCO passage corpus from HuggingFace...")
dataset = load_dataset("ms_marco", "v1.1", split="train")

# Extract all passage texts from 'passages' field
print("Extracting passage texts...")
all_passages = []
for row in dataset:
    passages = row.get("passages", {})
    if isinstance(passages, dict) and "passage_text" in passages:
        all_passages.extend(passages["passage_text"])

# Encode all passages
print("Encoding full document corpus...")
encodings = []
with torch.no_grad():
    for passage in tqdm(all_passages):
        tokens = simple_tokenize(passage)
        if not tokens:
            continue
        indices = torch.tensor([word2idx.get(token, 0) for token in tokens], dtype=torch.long).unsqueeze(0).to(device)
        enc = doc_encoder(indices)
        encodings.append(enc.cpu())

all_doc_encodings = torch.cat(encodings, dim=0)
torch.save(all_doc_encodings, "full_corpus_doc_encodings.pt")

print("✅ Full corpus encoding completed and saved to full_corpus_doc_encodings.pt")