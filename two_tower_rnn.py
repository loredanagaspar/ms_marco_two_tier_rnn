# two_tower_rnn.py

import gensim.downloader as api
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import re
import random
import os
import wandb
import pandas as pd
import pickle
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from dotenv import load_dotenv
import time
load_dotenv()

# Set device for GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load pretrained Word2Vec embeddings
glove_vectors = api.load("word2vec-google-news-300")
embedding_dim = glove_vectors.vector_size

# Step 2: Simple whitespace + punctuation tokenizer


def simple_tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# Step 3: Build vocabulary and embedding matrix from saved vocab file


def load_vocab_and_build_matrix(vocab_path):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    print(f"Loaded vocabulary of size {len(vocab)}")
    embedding_matrix = []
    word2idx = {}
    for i, word in enumerate(vocab):
        if word in glove_vectors:
            embedding_matrix.append(torch.tensor(glove_vectors[word]))
        else:
            embedding_matrix.append(torch.randn(embedding_dim))
        word2idx[word] = i
    return torch.stack(embedding_matrix), word2idx

# Step 4: RNN Encoder


class RNNEncoder(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()
        _, embedding_dim = embedding_matrix.size()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=True)
        self.rnn = nn.GRU(embedding_dim, 128, batch_first=True)

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        _, hidden = self.rnn(embedded)
        return hidden.squeeze(0)

# Step 5: Processing a single (query, doc_pos, doc_neg) triple (already tokenized)


def process_tokenized_tokens(triple_tokens, word2idx):
    token_ids = []
    for tokens in triple_tokens:
        indices = [word2idx.get(token, 0) for token in tokens]
        token_ids.append(torch.tensor(indices))
    return pad_sequence(token_ids, batch_first=True)

# Step 6: Triplet Loss


def triplet_loss(query, pos, neg, margin=1.0):
    pos_dist = F.pairwise_distance(query, pos, p=2)
    neg_dist = F.pairwise_distance(query, neg, p=2)
    return torch.mean(F.relu(pos_dist - neg_dist + margin))


# Step 7: Load real MS MARCO triples using pandas
def load_triples_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df[["query", "positive_doc", "negative_doc"]].values.tolist()

# Step 8: Load pre-tokenized MS MARCO triples from pickle file


def load_tokenized_triples_from_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    queries = data["tokenized_queries"]
    positives = data["tokenized_positives"]
    negatives = data["tokenized_negatives"]
    return list(zip(queries, positives, negatives))

# Step 9: Training loop with validation, early stopping, best model saving, and wandb logging


def evaluate(val_triples, encoder_q, encoder_d, word2idx, batch_size=64):
    encoder_q.eval()
    encoder_d.eval()
    total_loss = 0
    batches = create_batches(val_triples, word2idx, batch_size)
    print("Starting validation...")
    with torch.no_grad():
        for i, (q, pos, neg) in enumerate(tqdm(batches, desc="Validation")):
            q_enc = encoder_q(q)
            pos_enc = encoder_d(pos)
            neg_enc = encoder_d(neg)
            loss = triplet_loss(q_enc, pos_enc, neg_enc)
            if i % 100 == 0:
                print(f"[Validation] Batch {i}, Loss: {loss.item():.4f}")
            total_loss += loss.item()
    return total_loss / len(batches)


def evaluate_test(test_triples, encoder_q, encoder_d, word2idx, batch_size=64):
    encoder_q.eval()
    encoder_d.eval()
    total_loss = 0
    cos_pos_all, cos_neg_all = [], []
    batches = create_batches(test_triples, word2idx, batch_size)
    print("Starting test evaluation...")
    with torch.no_grad():
        for i, (q, pos, neg) in enumerate(tqdm(batches, desc="Testing")):
            q_enc = encoder_q(q)
            pos_enc = encoder_d(pos)
            neg_enc = encoder_d(neg)
            loss = triplet_loss(q_enc, pos_enc, neg_enc)
            if i % 100 == 0:
                print(f"[Test] Batch {i}, Loss: {loss.item():.4f}")
            total_loss += loss.item()
            cos_pos = F.cosine_similarity(q_enc, pos_enc)
            cos_neg = F.cosine_similarity(q_enc, neg_enc)
            cos_pos_all.extend(cos_pos.cpu().tolist())
            cos_neg_all.extend(cos_neg.cpu().tolist())

    test_loss = total_loss / len(batches)
    wandb.log({
        "test_loss": test_loss,
        "cos_sim/positive": wandb.Histogram(cos_pos_all),
        "cos_sim/negative": wandb.Histogram(cos_neg_all)
    })
    print(f"Test Loss: {test_loss:.4f}")

def create_batches(triples, word2idx, batch_size):
    batches = []
    for i in range(0, len(triples), batch_size):
        batch = triples[i:i+batch_size]
        queries, positives, negatives = zip(*batch)
        q_ids = [torch.tensor([word2idx.get(t, 0) for t in q], dtype=torch.long) for q in queries]
        p_ids = [torch.tensor([word2idx.get(t, 0) for t in p], dtype=torch.long) for p in positives]
        n_ids = [torch.tensor([word2idx.get(t, 0) for t in n], dtype=torch.long) for n in negatives]
        q_pad = pad_sequence(q_ids, batch_first=True).to(device)
        p_pad = pad_sequence(p_ids, batch_first=True).to(device)
        n_pad = pad_sequence(n_ids, batch_first=True).to(device)
        batches.append((q_pad, p_pad, n_pad))
    return batches

def train(triples, encoder_q, encoder_d, word2idx, epochs=5, lr=1e-3, val_triples=None, test_triples=None, patience=2, save_dir="models", batch_size=64):
    os.makedirs(save_dir, exist_ok=True)
    encoder_q.to(device)
    encoder_d.to(device)
    optimizer = torch.optim.Adam(list(encoder_q.parameters()) + list(encoder_d.parameters()), lr=lr)

    wandb.init(project="two-tower-rnn", config={"epochs": epochs, "lr": lr})
    wandb.watch([encoder_q, encoder_d], log='all')

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Starting training for {epochs} epochs on {len(triples)} triples...")

    for epoch in range(epochs):
        print(f"\n[Epoch {epoch + 1}] Starting new epoch...")
        encoder_q.train()
        encoder_d.train()
        random.shuffle(triples)
        batches = create_batches(triples, word2idx, batch_size)
        total_loss = 0
        start_time = time.time()

        for i, (q_batch, p_batch, n_batch) in enumerate(tqdm(batches, desc=f"Epoch {epoch + 1}")):
            if i == 0:
                print(f"[Epoch {epoch + 1}] Starting batch loop with {len(batches)} batches")

            q = encoder_q(q_batch)
            pos = encoder_d(p_batch)
            neg = encoder_d(n_batch)
            loss = triplet_loss(q, pos, neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 10 == 0:
                avg_loss_so_far = total_loss / (i + 1)
                print(f"[Epoch {epoch + 1}][Batch {i}] Loss: {loss.item():.4f}, Avg: {avg_loss_so_far:.4f}")

        epoch_time = time.time() - start_time
        avg_train_loss = total_loss / len(batches)
        val_loss = evaluate(val_triples, encoder_q, encoder_d, word2idx) if val_triples else None

        print(f"[Epoch {epoch + 1}] Completed in {epoch_time:.2f}s. Train Loss: {avg_train_loss:.4f}" + (f", Val Loss: {val_loss:.4f}" if val_loss is not None else ""))

        wandb.log({"loss/train": avg_train_loss, "loss/val": val_loss, "epoch": epoch + 1})

        if val_triples:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(encoder_q.state_dict(), os.path.join(save_dir, "best_encoder_q.pt"))
                torch.save(encoder_d.state_dict(), os.path.join(save_dir, "best_encoder_d.pt"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    if test_triples:
        evaluate_test(test_triples, encoder_q, encoder_d, word2idx)


# MAIN FUNCTION
if __name__ == "__main__":

    # Paths to your prepared data
    train_path = "tokenizedjson/train_tokenized.pkl"
    val_path = "tokenizedjson/validation_tokenized.pkl"
    test_path = "tokenizedjson/test_tokenized.pkl"

    vocab_path = "new_vocab.pkl"

    # Load vocab and embedding matrix
    embedding_matrix, word2idx = load_vocab_and_build_matrix(vocab_path)
    # ✅ Save matrix for reuse in evaluation and corpus encoding
    torch.save(embedding_matrix, "embedding_matrix.pt")
    # Load tokenized triples
    train_triples = load_tokenized_triples_from_pickle(train_path)
    val_triples = load_tokenized_triples_from_pickle(val_path)
    test_triples = load_tokenized_triples_from_pickle(test_path)

    # Init encoders
    query_encoder = RNNEncoder(embedding_matrix)
    doc_encoder = RNNEncoder(embedding_matrix)

    # Start training
    train(train_triples, query_encoder, doc_encoder, word2idx,
          val_triples=val_triples, test_triples=test_triples)
