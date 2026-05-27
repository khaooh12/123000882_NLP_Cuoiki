import os
import numpy as np
import pandas as pd

def main():
    print("\n[Embedding] Bắt đầu trích xuất Sentence Embeddings...")
    
    input_path = "data/processed/cleaned_articles.csv"
    output_path = "data/processed/embeddings.npy"
    
    if not os.path.exists(input_path):
        print(f"[Lỗi] Không tìm thấy file dữ liệu đã tiền xử lý: {input_path}")
        print("Hãy chạy 'python src/preprocess.py' trước.")
        return
        
    df = pd.read_csv(input_path)
    # Ensure cleaned_text is string and fill NaNs
    texts = df["cleaned_text"].fillna("").astype(str).tolist()
    
    print(f"  Số lượng văn bản cần xử lý: {len(texts)}")
    
    # Try importing sentence_transformers
    try:
        from sentence_transformers import SentenceTransformer
        print("  Đang tải mô hình paraphrase-multilingual-MiniLM-L12-v2...")
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        
        print("  Đang trích xuất embeddings (có thể sử dụng GPU nếu có)...")
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        
    except Exception as e:
        print(f"\n[Cảnh báo] Lỗi khi sử dụng SentenceTransformer: {e}")
        print("  Tự động chuyển sang phương án dự phòng: Sử dụng TF-IDF làm vector đặc trưng...")
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=384)  # 384 is the embedding size of MiniLM-L12-v2
        tfidf_matrix = vectorizer.fit_transform(texts)
        embeddings = tfidf_matrix.toarray()
        
    # Save to disk
    np.save(output_path, embeddings)
    print(f"[Embedding Hoàn tất] Đã lưu ma trận vector đặc trưng kích thước {embeddings.shape} vào: {output_path}")

if __name__ == "__main__":
    main()
