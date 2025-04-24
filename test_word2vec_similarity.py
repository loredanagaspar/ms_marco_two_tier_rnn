# file: test_word2vec_similarity.py
from gensim.models import Word2Vec

model_path = "models/word2vec_combined.bin"
model = Word2Vec.load(model_path)

query_words = ["information", "search", "language", "ranking"]
for word in query_words:
    if word in model.wv:
        print(f"\nTop words similar to '{word}':")
        for similar_word, score in model.wv.most_similar(word, topn=5):
            print(f"  {similar_word}: {score:.4f}")
    else:
        print(f"\n'{word}' not found in the vocabulary.")
