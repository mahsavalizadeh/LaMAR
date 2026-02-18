import pandas as pd
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE  

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


with open('4_features_Recformer_Office/finetune_data/Office/meta_data.json', 'r', encoding='utf-8') as f:
    data_1 = json.load(f)

df = pd.DataFrame.from_dict(data_1, orient='index')




model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
df["function_embedding"] = df["function"].apply(lambda x: model.encode(x))
embeddings_matrix = np.array(df["function_embedding"].tolist())



models = SentenceTransformer("multi-qa-mpnet-base-cos-v1", device=device)
df["semantic_embedding"] = df["function"].apply(lambda x: models.encode(x))
semantic_embeddings_matrix = np.array(df["semantic_embedding"].tolist())


# Compute similarities in chunks
def compute_similarities_in_chunks(embeddings, threshold=0.6, chunk_size=5000):
    num_samples = len(embeddings)
    high_sim_counts = np.zeros(num_samples, dtype=int)

    for start in range(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)
        chunk = embeddings[start:end]
        sim_matrix = cosine_similarity(chunk, embeddings)

        # Count values above threshold (excluding self-comparison)
        high_sim_counts[start:end] = (sim_matrix > threshold).sum(axis=1) - 1  # Exclude self-similarity

    return high_sim_counts


df["redundant_function"] = compute_similarities_in_chunks(embeddings_matrix, threshold=0.75, chunk_size=5000)
df["redundant_function_semantic"] = compute_similarities_in_chunks(semantic_embeddings_matrix, threshold=0.75, chunk_size=5000)


#similarity_df = pd.DataFrame(similarity_matrix, index=df["title"], columns=df["title"])

#threshold = 0.9

#high_similarity_counts = (similarity_matrix > threshold).sum(axis=1)

#df["redundant_function"] = high_similarity_counts

# Apply t-SNE on both embeddings
for perplexity in [50, 80, 120, 140]:
    tsne_g = TSNE(n_components=2, random_state=42, perplexity=perplexity, learning_rate = 100)
    tsne_general = tsne_g.fit_transform(embeddings_matrix)
    #df["tsne_x_G"] = tsne_general[:, 0]
    #df["tsne_y_G"] = tsne_general[:, 1]
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_general[:, 0], tsne_general[:, 1], c='blue', alpha=0.6, edgecolor='k')
    #plt.title("t-SNE Visualization of Embeddings obtained by General Model - threshold:0.7")
    plt.title(f"t-SNE (General Model) - threshold: 0.75 (Perplexity {perplexity})")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.show()
    #plt.savefig("t-SNE Visualization of Embeddings obtained by General Model - threshold 0.7_Office.png")
    plt.savefig(f"tsne_general_model_threshold0.75_perplexity{perplexity}_Office.png")
    print("*********tsne for general model:",tsne_g.kl_divergence_, "*********")
    
    
    tsne_sem = TSNE(n_components=2, random_state=42, perplexity=perplexity, learning_rate = 100)
    tsne_semantic = tsne_sem.fit_transform(semantic_embeddings_matrix)
    #df["tsne_sem_x"] = tsne_semantic[:, 0]
    #df["tsne_sem_y"] = tsne_semantic[:, 1]
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_semantic[:, 0], tsne_semantic[:, 1], c='blue', alpha=0.6, edgecolor='k')
    #plt.title("t-SNE Visualization of Embeddings obtained by Semantic Model - threshold:0.7")
    plt.title(f"t-SNE (Semantic Model) - threshold: 0.75 (Perplexity {perplexity})")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.show()
    #plt.savefig("t-SNE Visualization of Embeddings obtained by Semantic Model - threshold 0.7_Office.png")
    plt.savefig(f"tsne_Semantic_model_threshold0.75_perplexity{perplexity}_Office.png")
    print("*********tsne for semantic model:",tsne_sem.kl_divergence_, "*********")


df.drop(columns=["function_embedding", "semantic_embedding"], inplace=True)
df.to_json('meta_data_gpt_bias_2_models_tSNE_0.75.json', orient='index')


