import numpy as np
from gensim.models import Word2Vec


def average_pooling(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


# Example:
query_tokens = "how to bake a cake".lower().split()
positive_tokens = "cake baking guide and tips".lower().split()
negative_tokens = "how to fix a bike".lower().split()

model = Word2Vec.load("models/word2vec_combined.bin")

query_vec = average_pooling(query_tokens, model)
pos_vec = average_pooling(positive_tokens, model)
neg_vec = average_pooling(negative_tokens, model)
