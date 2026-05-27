import os
import sys
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath("."))
from src.topic_model import FallbackTopicModel

def main():
    cleaned_path = "data/processed/cleaned_articles.csv"
    embeddings_path = "data/processed/embeddings.npy"
    
    if not os.path.exists(cleaned_path) or not os.path.exists(embeddings_path):
        print("[Lỗi] Không tìm thấy dữ liệu tiền xử lý hoặc embeddings.")
        return
        
    print("Loading data...")
    df = pd.read_csv(cleaned_path)
    embeddings = np.load(embeddings_path)
    
    docs = df["cleaned_text"].fillna("").astype(str).tolist()
    
    print("Training Fallback KMeans Topic Model on 67,711 articles...")
    model = FallbackTopicModel(n_topics=7)
    model.fit_transform(docs, embeddings)
    
    output_dir = "models/bertopic_model/fallback"
    print(f"Saving Fallback model to {output_dir}...")
    model.save(output_dir)
    
    print("Topic Info of Fallback KMeans Model:")
    print(model.get_topic_info())

if __name__ == "__main__":
    main()
