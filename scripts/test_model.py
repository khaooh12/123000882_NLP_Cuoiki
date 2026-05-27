import os
import sys
import pickle
import numpy as np
import pandas as pd

# Add src to system path
sys.path.append(os.path.abspath("."))
from src.preprocess import preprocess_text

def load_lda():
    print("\n[Test Pipeline] Đang tải mô hình LDA...")
    try:
        from gensim import corpora
        from gensim.models import LdaModel
        dictionary = corpora.Dictionary.load("models/lda_model/dictionary.dict")
        lda_model = LdaModel.load("models/lda_model/lda_model.model")
        is_fallback = False
        print("  -> Tải Gensim LDA Model thành công.")
    except Exception as e:
        print(f"  [Thông báo] Sử dụng Fallback LDA Model (Lỗi Gensim: {e})")
        from src.topic_model import FallbackLdaModel
        lda_model = FallbackLdaModel.load("models/lda_model")
        dictionary = None
        is_fallback = True
        
    return lda_model, dictionary, is_fallback

def load_official_bertopic():
    print("\n[Test Pipeline] Đang tải mô hình BERTopic chính thức (HDBSCAN)...")
    official_path = "models/bertopic_model/bertopic"
    if os.path.exists(official_path):
        from bertopic import BERTopic
        model = BERTopic.load(official_path)
        print("  -> Tải BERTopic Model chính thức thành công.")
        return model
    else:
        print("  -> Không tìm thấy BERTopic Model chính thức.")
        return None

def load_fallback_bertopic():
    print("\n[Test Pipeline] Đang tải mô hình BERTopic dự phòng (KMeans)...")
    fallback_path = "models/bertopic_model/fallback"
    if os.path.exists(f"{fallback_path}/model.pkl"):
        from src.topic_model import FallbackTopicModel
        model = FallbackTopicModel.load(fallback_path)
        print("  -> Tải Fallback BERTopic Model (KMeans) thành công.")
        return model
    else:
        print("  -> Không tìm thấy Fallback BERTopic Model (KMeans).")
        return None

def get_vietnamnet_embeddings(texts, train_df_path="data/processed/cleaned_articles.csv"):
    print("\n[Test Pipeline] Trích xuất vector đặc trưng cho tập kiểm thử...")
    try:
        from sentence_transformers import SentenceTransformer
        print("  -> Đang dùng paraphrase-multilingual-MiniLM-L12-v2...")
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        embeddings = model.encode(texts, show_progress_bar=True)
    except Exception as e:
        print(f"  [Cảnh báo] Lỗi khi trích xuất vector SBERT: {e}")
        print("  -> Tự động chuyển sang phương án dự phòng TF-IDF...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        train_df = pd.read_csv(train_df_path)
        train_texts = train_df["cleaned_text"].fillna("").astype(str).tolist()
        
        vectorizer = TfidfVectorizer(max_features=384)
        vectorizer.fit(train_texts)
        embeddings = vectorizer.transform(texts).toarray()
        
    return embeddings

def main():
    print("="*60)
    print("  ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP KIỂM THỬ VIETNAMNET")
    print("="*60)
    
    test_path = "data/raw/test_articles.csv"
    train_cleaned_path = "data/processed/cleaned_articles.csv"
    
    if not os.path.exists(test_path):
        print(f"[Lỗi] Không tìm thấy tệp kiểm thử: {test_path}")
        print("Vui lòng chạy 'python src/scrape_test_set.py' trước để cào tập test.")
        return
        
    # Read test articles
    test_df = pd.read_csv(test_path)
    print(f"Đã đọc {len(test_df)} bài viết kiểm thử từ {test_df['category'].nunique()} chuyên mục của VietnamNet.")
    
    # 1. Preprocess test articles
    print("\n[Test Pipeline] Đang chạy tiền xử lý văn bản kiểm thử...")
    test_df["combined"] = (test_df["title"].fillna("") + " " + test_df["description"].fillna("")).str.strip()
    test_df["cleaned_text"] = test_df["combined"].apply(preprocess_text)
    
    # Filter out empty texts
    test_df = test_df[test_df["cleaned_text"].str.strip().str.len() > 5].copy()
    print(f"  -> Hoàn tất tiền xử lý. Còn lại {len(test_df)} bài viết kiểm thử hợp lệ.")
    
    cleaned_docs = test_df["cleaned_text"].tolist()
    
    # 2. Load models
    lda_model, dictionary, is_fallback_lda = load_lda()
    official_bertopic = load_official_bertopic()
    fallback_bertopic = load_fallback_bertopic()
    
    # 3. Predict using LDA
    print("\n[Test Pipeline] Dự đoán bằng mô hình LDA...")
    lda_preds = []
    for doc in cleaned_docs:
        if is_fallback_lda:
            topic_probs = lda_model.get_document_topics(doc)
        else:
            bow = dictionary.doc2bow(doc.split())
            topic_probs = lda_model.get_document_topics(bow)
            
        dominant_topic = sorted(topic_probs, key=lambda x: x[1], reverse=True)[0] if topic_probs else (-1, 0.0)
        lda_preds.append(dominant_topic[0])
        
    test_df["lda_topic"] = lda_preds
    
    # Generate SBERT embeddings if we have any BERTopic model loaded
    embeddings = None
    if official_bertopic is not None or fallback_bertopic is not None:
        embeddings = get_vietnamnet_embeddings(cleaned_docs, train_cleaned_path)
    
    # 4. Predict using Official BERTopic
    if official_bertopic is not None:
        print("\n[Test Pipeline] Dự đoán bằng mô hình BERTopic chính thức (HDBSCAN)...")
        bert_preds, _ = official_bertopic.transform(cleaned_docs, embeddings)
        test_df["bertopic_official"] = bert_preds
        
    # 5. Predict using Fallback BERTopic
    if fallback_bertopic is not None:
        print("\n[Test Pipeline] Dự đoán bằng mô hình BERTopic dự phòng (KMeans)...")
        bert_fallback_preds, _ = fallback_bertopic.transform(cleaned_docs, embeddings)
        test_df["bertopic_fallback"] = bert_fallback_preds

    # 6. Output mapping results & analysis
    print("\n" + "="*60)
    print("  KẾT QUẢ ĐỐI CHIẾU PHÂN LOẠI CHUYÊN MỤC")
    print("="*60)
    
    categories = test_df["category"].unique()
    
    print("\n[1] Thống kê phân bổ theo LDA:")
    for cat in categories:
        sub = test_df[test_df["category"] == cat]
        lda_counts = sub["lda_topic"].value_counts().to_dict()
        top_topic = max(lda_counts, key=lda_counts.get) if lda_counts else -1
        top_count = lda_counts[top_topic] if lda_counts else 0
        pct = (top_count / len(sub)) * 100
        
        if is_fallback_lda:
            words = [w[0] for w in lda_model.show_topic(top_topic, topn=4)]
        else:
            words = [w[0] for w in lda_model.show_topic(top_topic, topn=4)] if top_topic != -1 else []
        words_str = ", ".join(words)
        
        print(f"  - Chuyên mục '{cat:<10}': {len(sub):>2} bài -> Phần lớn quy vào Chủ đề {top_topic} ({top_count} bài - {pct:.1f}%)")
        print(f"    * Từ khóa chủ đề: {words_str}")
        
    if official_bertopic is not None:
        print("\n[2] Thống kê phân bổ theo BERTopic chính thức (HDBSCAN):")
        topic_info = official_bertopic.get_topic_info()
        topic_name_map = dict(zip(topic_info["Topic"], topic_info["Name"]))
        
        for cat in categories:
            sub = test_df[test_df["category"] == cat]
            bert_counts = sub["bertopic_official"].value_counts().to_dict()
            top_topic = max(bert_counts, key=bert_counts.get) if bert_counts else -1
            top_count = bert_counts[top_topic] if bert_counts else 0
            pct = (top_count / len(sub)) * 100
            
            topic_name = topic_name_map.get(top_topic, f"Topic_{top_topic}")
            print(f"  - Chuyên mục '{cat:<10}': {len(sub):>2} bài -> Phần lớn quy vào {topic_name} ({top_count} bài - {pct:.1f}%)")
            
    if fallback_bertopic is not None:
        print("\n[3] Thống kê phân bổ theo BERTopic dự phòng (KMeans):")
        topic_info_fb = fallback_bertopic.get_topic_info()
        topic_name_map_fb = dict(zip(topic_info_fb["Topic"], topic_info_fb["Name"]))
        
        for cat in categories:
            sub = test_df[test_df["category"] == cat]
            bert_counts_fb = sub["bertopic_fallback"].value_counts().to_dict()
            top_topic = max(bert_counts_fb, key=bert_counts_fb.get) if bert_counts_fb else -1
            top_count = bert_counts_fb[top_topic] if bert_counts_fb else 0
            pct = (top_count / len(sub)) * 100
            
            topic_name = topic_name_map_fb.get(top_topic, f"Topic_{top_topic}")
            print(f"  - Chuyên mục '{cat:<10}': {len(sub):>2} bài -> Phần lớn quy vào {topic_name} ({top_count} bài - {pct:.1f}%)")
        
    # Save results
    results_path = "data/processed/test_predictions.csv"
    test_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    print("\n" + "="*60)
    print(f"Đã lưu kết quả chi tiết vào: {results_path}")
    print("="*60)

if __name__ == "__main__":
    main()
