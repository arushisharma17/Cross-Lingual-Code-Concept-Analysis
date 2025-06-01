import torch
import numpy as np
import json
import os
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def extract_cluster_centroids(model_name, corpus_path, cluster_files, output_dir, layer=12):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load clusters
    clusters = [{}, {}]
    for i, file_path in enumerate(cluster_files):
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('|||')
                    word = parts[0].strip()
                    cluster_num = int(parts[-1].strip())
                    if cluster_num not in clusters[i]:
                        clusters[i][cluster_num] = []
                    clusters[i][cluster_num].append(word)
    
    # Load corpus
    with open(corpus_path, 'r') as f:
        corpus_data = f.read().strip().split('\n')
    
    # Extract words from corpus
    all_words = [set(), set()]
    for line in corpus_data:
        parts = line.split(' ||| ')
        if len(parts) == 2:
            src_words = parts[0].strip().split()
            tgt_words = parts[1].strip().split()
            all_words[0].update(src_words)
            all_words[1].update(tgt_words)
    
    # Get embeddings for all words
    embeddings = [{}, {}]
    
    for lang_idx, words in enumerate(all_words):
        for word in tqdm(words, desc=f"Extracting embeddings for language {lang_idx+1}"):
            inputs = tokenizer(word, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get embeddings from specified layer
            hidden_states = outputs.hidden_states[layer]
            # Use mean of subword embeddings as word embedding
            word_embedding = hidden_states[0, 1:-1, :].mean(dim=0).cpu().numpy()
            embeddings[lang_idx][word] = word_embedding
    
    # Compute centroids for each cluster
    centroids = [{}, {}]
    for lang_idx in range(2):
        for cluster_num, words in clusters[lang_idx].items():
            cluster_embeddings = []
            for word in words:
                if word in embeddings[lang_idx]:
                    cluster_embeddings.append(embeddings[lang_idx][word])
            if cluster_embeddings:
                centroids[lang_idx][cluster_num] = np.mean(cluster_embeddings, axis=0)
    
    # Compute cosine similarities between cluster centroids
    similarities = {}
    for src_cluster, src_centroid in centroids[0].items():
        similarities[src_cluster] = {}
        for tgt_cluster, tgt_centroid in centroids[1].items():
            sim = cosine_similarity([src_centroid], [tgt_centroid])[0][0]
            similarities[src_cluster][tgt_cluster] = float(sim)
    
    # Save results
    with open(os.path.join(output_dir, "centroid_similarities.json"), "w") as f:
        json.dump(similarities, f, indent=2)
    
    return similarities

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/codebert-base")
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--cluster_file1", required=True)
    parser.add_argument("--cluster_file2", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layer", type=int, default=12)
    args = parser.parse_args()
    
    extract_cluster_centroids(
        args.model_name,
        args.corpus_path,
        [args.cluster_file1, args.cluster_file2],
        args.output_dir,
        args.layer
    ) 