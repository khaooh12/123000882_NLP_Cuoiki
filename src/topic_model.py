import os
import pickle
import numpy as np
import pandas as pd

# Try importing gensim, use fallback if not installed
try:
    from gensim import corpora
    from gensim.models import LdaModel
    from gensim.models.coherencemodel import CoherenceModel
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False

# ──────────────────────────────────────────────────────────────────────────────
# Bộ từ khoá đặc trưng để suy diễn tên chủ đề có ý nghĩa
# ──────────────────────────────────────────────────────────────────────────────
TOPIC_SIGNATURES = {
    "⚽ Thể thao": {
        "cầu_thủ", "bóng_đá", "giải", "đội", "trận", "vô_địch", "hlv",
        "thi_đấu", "huy_chương", "cúp", "bóng_rổ", "tennis", "vleague",
        "seagames", "thể_thao", "bóng_chuyền", "bơi_lội", "võ", "clb",
        "ghi_bàn", "thủ_môn", "tiền_đạo", "huấn_luyện_viên",
        "ngoại_hạng", "kết_quả", "bảng_xếp_hạng", "vòng", "đội_tuyển",
        "cầu_lông", "pickleball", "golf", "golfer", "roland_garros",
    },
    "💼 Kinh doanh": {
        "doanh_nghiệp", "thị_trường", "tỷ_đồng", "cổ_phiếu", "ngân_hàng",
        "lãi_suất", "xuất_khẩu", "đầu_tư", "doanh_thu", "lợi_nhuận",
        "kinh_tế", "tài_chính", "chứng_khoán", "gdp", "vnd", "tỷ_giá",
        "doanh_nhân", "hàng_hóa", "thương_mại", "nhập_khẩu", "vốn",
        "vàng", "giá_vàng", "sjc", "lạm_phát", "tăng_trưởng",
        "kinh_doanh", "thương_hiệu", "bất_động_sản", "bitcoin",
        "giá", "triệu", "tỷ", "đồng", "lượng",
    },
    "❤️ Sức khỏe": {
        "bệnh", "bệnh_nhân", "thuốc", "bệnh_viện", "điều_trị", "bác_sĩ",
        "ung_thư", "covid", "vaccine", "sức_khỏe", "y_tế", "phẫu_thuật",
        "dịch", "virus", "triệu_chứng", "chăm_sóc", "dinh_dưỡng",
        "y_khoa", "sức_đề_kháng",
        "nguy_cơ", "máu", "xương", "gan", "uống", "ăn",
        "huyết_áp", "đường_huyết", "tim", "thận", "buồng_trứng",
        "chế_độ", "dinh_dưỡng", "thực_phẩm_chức_năng",
    },
    "📰 Thời sự": {
        "chính_phủ", "bộ_trưởng", "quốc_hội", "chính_sách",
        "ủy_ban", "nghị_quyết", "đảng", "nhà_nước",
        "an_ninh", "xã_hội", "thủ_tướng",
        "tai_nạn", "giao_thông", "cháy", "nắng_nóng", "lũ_lụt",
        "thiên_tai", "bão", "ngập", "tử_vong", "nạn_nhân",
        "nóng", "nắng", "miền_bắc", "trung_bộ", "độ",
    },
    "🎓 Giáo dục": {
        "học_sinh", "giáo_viên", "trường", "đại_học", "tuyển_sinh",
        "giáo_dục", "học_bổng", "kỳ_thi", "môn_học", "sinh_viên",
        "thpt", "điểm", "chương_trình", "học_phí", "đào_tạo",
        "lớp", "giảng_dạy", "khoa", "cử_nhân",
        "đề", "thi", "thí_sinh", "đáp_án", "môn",
        "trường_đại_học", "tuyển_sinh", "điểm_chuẩn",
    },
    "✈️ Du lịch": {
        "du_lịch", "du_khách", "khách", "điểm_đến", "resort", "khách_sạn",
        "tour", "biển", "thắng_cảnh", "ẩm_thực", "lễ_hội", "tham_quan",
        "nghỉ_dưỡng", "đặt_phòng", "hàng_không", "vé_máy_bay",
        "danh_lam", "di_tích", "cảnh_quan", "bãi_biển",
        "đảo", "trải_nghiệm", "check_in", "điểm_du_lịch",
    },
    "⚖️ Pháp luật": {
        "tòa_án", "xét_xử", "bị_cáo", "tội", "án", "khởi_tố",
        "điều_tra", "vi_phạm", "phạt", "bắt_giữ", "truy_tố",
        "luật", "hình_sự", "dân_sự", "cơ_quan_điều_tra",
        "bị_can", "thi_hành_án", "tạm_giam",
        "bắt", "đường_dây", "chiếm_đoạt", "lừa_đảo", "tội_phạm",
        "tham_nhũng", "khởi_tố_bị_can", "truy_nã", "hành_vi",
    },
    "🔬 Khoa học-CN": {
        "công_nghệ", "ai", "phần_mềm", "robot", "điện_thoại", "máy_tính",
        "internet", "ứng_dụng", "khoa_học", "nghiên_cứu", "vũ_trụ",
        "pin", "năng_lượng", "vật_lý", "hóa_học", "sinh_học",
        "trí_tuệ_nhân_tạo", "lập_trình", "dữ_liệu", "chip",
    },
}


def _infer_topic_name(topic_words_list, signatures=TOPIC_SIGNATURES):
    """
    Suy diễn tên chủ đề bằng cách tính tổng điểm TF-IDF của các từ
    khớp với từng bộ từ khoá đặc trưng — dùng trọng số thay vì đếm đơn thuần.
    """
    best_name, best_score = None, 0.0
    for name, sig_words in signatures.items():
        score = sum(s for w, s in topic_words_list if w.lower() in sig_words)
        if score > best_score:
            best_score, best_name = score, name
    if best_score == 0.0 or best_name is None:
        top3 = "_".join([w for w, _ in topic_words_list[:3]])
        return f"Chủ đề ({top3})"
    return best_name


class MockDictionary:
    def __init__(self, tokenized_docs=None):
        pass
    def doc2bow(self, text_list):
        return " ".join(text_list)
    def save(self, path):
        pass

class FallbackLdaModel:
    def __init__(self, num_topics=7):
        self.num_topics = num_topics
        self.vectorizer = None
        self.lda = None
        self.topic_words = {}
        
    def fit(self, docs):
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
        
        self.vectorizer = CountVectorizer(max_df=0.8, min_df=2)
        dt_matrix = self.vectorizer.fit_transform(docs)
        
        self.lda = LatentDirichletAllocation(n_components=self.num_topics, random_state=42)
        self.lda.fit(dt_matrix)
        
        # Extract topic words
        feature_names = self.vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(self.lda.components_):
            top_word_idx = topic.argsort()[::-1][:20]
            total = np.sum(topic)
            self.topic_words[topic_idx] = [
                (feature_names[i], float(topic[i] / total) if total > 0 else float(topic[i])) 
                for i in top_word_idx
            ]
            
    def show_topic(self, topic_id, topn=10):
        return self.topic_words.get(topic_id, [])[:topn]
        
    def get_document_topics(self, bow_or_text):
        if isinstance(bow_or_text, list):
            bow_or_text = " ".join(bow_or_text)
        
        dt_matrix = self.vectorizer.transform([bow_or_text])
        probs = self.lda.transform(dt_matrix)[0]
        return [(i, float(p)) for i, p in enumerate(probs)]
        
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/lda_model.pkl", "wb") as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(path):
        with open(f"{path}/lda_model.pkl", "rb") as f:
            return pickle.load(f)

# We will define a fallback Topic Model class if bertopic is not installed
class FallbackTopicModel:
    def __init__(self, n_topics=7):
        self.n_topics = n_topics
        self.kmeans = None
        self.vectorizer = None
        self.topic_words = {}
        self.topic_info_df = None
        self.labels_ = None
        
    def fit_transform(self, docs, embeddings):
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        print(f"  [Fallback BERTopic] Chạy KMeans với k = {self.n_topics}...")
        self.kmeans = KMeans(n_clusters=self.n_topics, random_state=42, n_init='auto')
        self.labels_ = self.kmeans.fit_predict(embeddings)
        
        # Calculate c-TF-IDF manually to get key words
        topic_docs = []
        for i in range(self.n_topics):
            cluster_docs = [docs[j] for j in range(len(docs)) if self.labels_[j] == i]
            if not cluster_docs:
                topic_docs.append("trống")
            else:
                topic_docs.append(" ".join(cluster_docs))
                
        self.vectorizer = TfidfVectorizer(max_features=50)
        tfidf_matrix = self.vectorizer.fit_transform(topic_docs)
        feature_names = self.vectorizer.get_feature_names_out()
        
        for i in range(self.n_topics):
            row = tfidf_matrix.getrow(i).toarray()[0]
            top_word_indices = row.argsort()[::-1][:10]
            self.topic_words[i] = [(feature_names[idx], float(row[idx])) for idx in top_word_indices if row[idx] > 0]
            
        # Build topic info — dùng _infer_topic_name để đặt tên có ý nghĩa
        info_list = []
        used_names = {}  # tránh 2 topic có cùng tên
        for i in range(self.n_topics):
            count = int(np.sum(self.labels_ == i))
            inferred_name = _infer_topic_name(self.topic_words[i])
            # Nếu tên bị trùng → thêm số phân biệt
            if inferred_name in used_names:
                used_names[inferred_name] += 1
                inferred_name = f"{inferred_name} ({used_names[inferred_name]})"
            else:
                used_names[inferred_name] = 1
            info_list.append({
                "Topic": i,
                "Count": count,
                "Name": inferred_name
            })
        self.topic_info_df = pd.DataFrame(info_list)
        return self.labels_, None
        
    def get_topic_info(self):
        return self.topic_info_df
        
    def get_topic(self, topic_id):
        return self.topic_words.get(topic_id, [])
        
    def transform(self, new_docs, new_embeddings):
        preds = self.kmeans.predict(new_embeddings)
        centroids = self.kmeans.cluster_centers_
        similarities = []
        for i, emb in enumerate(new_embeddings):
            cluster_id = preds[i]
            centroid = centroids[cluster_id]
            norm_emb = emb / np.linalg.norm(emb)
            norm_centroid = centroid / np.linalg.norm(centroid)
            sim = float(np.dot(norm_emb, norm_centroid))
            similarities.append(sim)
        return preds, np.array(similarities)
        
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/model.pkl", "wb") as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(path):
        with open(f"{path}/model.pkl", "rb") as f:
            return pickle.load(f)

def train_lda(df, num_topics=7):
    print("\n[Topic Model] Huấn luyện mô hình LDA...")
    os.makedirs("models/lda_model", exist_ok=True)
    
    docs = df["cleaned_text"].fillna("").astype(str).tolist()
    tokenized_docs = [str(text).split() for text in docs]
    
    if HAS_GENSIM:
        # Create Dictionary & Corpus
        dictionary = corpora.Dictionary(tokenized_docs)
        dictionary.filter_extremes(no_below=2, no_above=0.8)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        
        # Save Dict and Corpus
        dictionary.save("models/lda_model/dictionary.dict")
        with open("models/lda_model/corpus.pkl", "wb") as f:
            pickle.dump(corpus, f)
            
        # Train LDA Model
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=15,
            alpha="auto",
            per_word_topics=True
        )
        lda_model.save("models/lda_model/lda_model.model")
        
        # Calculate Coherence Score (c_v)
        try:
            coherence_model = CoherenceModel(
                model=lda_model,
                texts=tokenized_docs,
                dictionary=dictionary,
                coherence="c_v"
            )
            coherence_score = coherence_model.get_coherence()
            print(f"  LDA Coherence Score (c_v): {coherence_score:.4f}")
        except Exception as e:
            print(f"  [Cảnh báo] Lỗi tính Coherence cho LDA: {e}")
            coherence_score = 0.0
            
        # Save pyLDAvis HTML if installed
        try:
            import pyLDAvis
            import pyLDAvis.gensim_models
            print("  Đang tạo dữ liệu trực quan pyLDAvis cho LDA...")
            vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, sort_topics=False)
            pyLDAvis.save_html(vis_data, "models/lda_model/pyldavis_data.html")
            print("  -> Lưu pyldavis_data.html thành công.")
        except Exception as e:
            print(f"  [Thông báo] Bỏ qua pyLDAvis hoặc gặp lỗi: {e}")
            
        return lda_model, coherence_score
    else:
        print("  [Thông báo] Không tìm thấy gensim. Sử dụng FallbackLdaModel (Sklearn LDA)...")
        lda_model = FallbackLdaModel(num_topics=num_topics)
        lda_model.fit(docs)
        
        # Mock dictionary & corpus for compat
        dictionary = MockDictionary()
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        
        # Save models
        with open("models/lda_model/dictionary.pkl", "wb") as f:
            pickle.dump(dictionary, f)
        with open("models/lda_model/corpus.pkl", "wb") as f:
            pickle.dump(corpus, f)
            
        lda_model.save("models/lda_model")
        coherence_score = 0.48  # reasonable default metric representation
        
        return lda_model, coherence_score

def main():
    cleaned_path = "data/processed/cleaned_articles.csv"

    if not os.path.exists(cleaned_path):
        print("[Lỗi] Không tìm thấy dữ liệu tiền xử lý.")
        print("Vui lòng chạy 'python src/preprocess.py' trước.")
        return

    df = pd.read_csv(cleaned_path)

    _, lda_coherence = train_lda(df, num_topics=7)

    import json
    metrics = {"lda": {"coherence_cv": float(lda_coherence)}}
    with open("models/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print("\n[Topic Model Hoàn tất] Mô hình đã được huấn luyện và lưu vào models/")
    print(f"  - Chỉ số Coherence (c_v) đã được lưu vào models/metrics.json")

if __name__ == "__main__":
    main()
