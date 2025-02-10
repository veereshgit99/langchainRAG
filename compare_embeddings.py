import numpy as np
from scipy.spatial.distance import cosine
from langchain_community.embeddings import HuggingFaceEmbeddings

def main():
    # Use free local embedding model
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Get embeddings for words
    words = ["apple", "iphone"]
    vectors = [embedding_function.embed_query(word) for word in words]

    print(f"✅ Vector for '{words[0]}': {vectors[0][:5]}... (truncated)")
    print(f"✅ Vector length: {len(vectors[0])}")

    # Compute cosine similarity between embeddings
    similarity_score = 1 - cosine(vectors[0], vectors[1])
    print(f"🔍 Cosine Similarity between '{words[0]}' and '{words[1]}': {similarity_score:.4f}")

if __name__ == "__main__":
    main()
