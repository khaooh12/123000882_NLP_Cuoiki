import os
import sys
import pickle
import json
import io
import contextlib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Force add src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.scraper import main as run_scraper
from src.preprocess import main as run_preprocess, preprocess_text
from src.embedding import main as run_embedding
from src.topic_model import main as run_topic_model, FallbackTopicModel, FallbackLdaModel
from src.visualize import reduce_dimensions_2d, plot_2d_embeddings, generate_wordcloud, plot_topic_distribution

# Set page layout
st.set_page_config(
    page_title="Phân Cụm Chủ Đề Văn Bản Tiếng Việt",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-title {
        font-weight: 700;
        background: linear-gradient(135deg, #6C5CE7 0%, #00CEC9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        text-align: center;
        color: #636e72;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(108, 92, 231, 0.15);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00CEC9;
    }
    
    .word-tag {
        display: inline-block;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        background-color: rgba(108, 92, 231, 0.1);
        color: #6c5ce7;
        border: 1px solid rgba(108, 92, 231, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check project file statuses
def get_status():
    status = {
        "raw_data": os.path.exists("data/raw/vnexpress_articles.csv"),
        "cleaned_data": os.path.exists("data/processed/cleaned_articles.csv"),
        "embeddings": os.path.exists("data/processed/embeddings.npy"),
        "lda_model": os.path.exists("models/lda_model/lda_model.model") or os.path.exists("models/lda_model/lda_model.pkl"),
        "bertopic_model": os.path.exists("models/bertopic_model/bertopic") or os.path.exists("models/bertopic_model/fallback/model.pkl"),
        "pyldavis": os.path.exists("models/lda_model/pyldavis_data.html"),
        "metrics": os.path.exists("models/metrics.json")
    }
    return status

status = get_status()

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("### ⚙️ Cấu hình & Trạng thái")

# Pipeline Status Checklist
st.sidebar.markdown("**Trạng thái Pipeline:**")
st.sidebar.markdown(f"{'✅' if status['raw_data'] else '❌'} 1. Thu thập dữ liệu (VnExpress)")
st.sidebar.markdown(f"{'✅' if status['cleaned_data'] else '❌'} 2. Tiền xử lý văn bản")
st.sidebar.markdown(f"{'✅' if status['embeddings'] else '❌'} 3. Trích xuất đặc trưng (SBERT)")
st.sidebar.markdown(f"{'✅' if (status['lda_model'] and status['bertopic_model']) else '❌'} 4. Huấn luyện mô hình")

st.sidebar.markdown("---")

# Run Pipeline Action
st.sidebar.markdown("### 🏋️ Trình chạy Pipeline tự động")
st.sidebar.write("Nếu chưa có mô hình hoặc muốn cập nhật dữ liệu mới nhất, hãy kích hoạt chạy toàn bộ pipeline tại đây.")
num_articles_per_cat = st.sidebar.slider("Số bài viết cào mỗi mục:", min_value=5, max_value=100, value=15, step=5)
use_dummy_data = st.sidebar.checkbox("Sử dụng dữ liệu giả lập (nhanh, offline)", value=True)

pipeline_btn = st.sidebar.button("🚀 Chạy toàn bộ Pipeline", use_container_width=True)

if pipeline_btn:
    progress_info = st.sidebar.info("Đang bắt đầu chạy pipeline...")
    log_container = st.empty()
    
    # Run the steps and stream console output to Streamlit
    log_output = io.StringIO()
    with contextlib.redirect_stdout(log_output):
        try:
            # 1. Scrape
            print("=== BƯỚC 1: CÀO DỮ LIỆU ===")
            sys.argv = ["scraper.py", "--limit", str(num_articles_per_cat)]
            if use_dummy_data:
                sys.argv.append("--dummy")
            run_scraper()
            log_container.code(log_output.getvalue())
            
            # 2. Preprocess
            print("\n=== BƯỚC 2: TIỀN XỬ LÝ ===")
            run_preprocess()
            log_container.code(log_output.getvalue())
            
            # 3. Embedding
            print("\n=== BƯỚC 3: TRÍCH XUẤT EMBEDDINGS ===")
            run_embedding()
            log_container.code(log_output.getvalue())
            
            # 4. Train Models
            print("\n=== BƯỚC 4: HUẤN LUYỆN MÔ HÌNH ===")
            run_topic_model()
            log_container.code(log_output.getvalue())
            
            progress_info.success("🎉 Đã chạy xong toàn bộ Pipeline!")
            st.rerun()
            
        except Exception as e:
            print(f"\n[Lỗi nghiêm trọng] Pipeline thất bại: {e}")
            log_container.code(log_output.getvalue())
            progress_info.error("❌ Pipeline gặp lỗi.")

st.sidebar.markdown("---")
st.sidebar.markdown("🏛️ **MSSV:** 123000882  \n**Môn học:** NLP Cuối Khoá")

# ----------------- MAIN INTERFACE -----------------
st.markdown("<div class='main-title'>🗺️ BẢN ĐỒ CHỦ ĐỀ VĂN BẢN TIẾNG VIỆT</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Hệ thống phân cụm chủ đề không giám sát bằng mô hình LDA và BERTopic nâng cao</div>", unsafe_allow_html=True)

if not status["cleaned_data"] or not status["lda_model"] or not status["bertopic_model"]:
    st.warning("⚠️ Dự án chưa được huấn luyện mô hình hoặc thiếu dữ liệu. Vui lòng nhấn nút **🚀 Chạy toàn bộ Pipeline** ở thanh bên trái để tạo mô hình tự động trước khi sử dụng ứng dụng.")
    st.stop()

# Load models and data
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/cleaned_articles.csv")
    embeddings = np.load("data/processed/embeddings.npy")
    return df, embeddings

df, embeddings = load_data()

# Load LDA Model
@st.cache_resource
def load_lda():
    with open("models/lda_model/corpus.pkl", "rb") as f:
        corpus = pickle.load(f)
        
    try:
        from gensim import corpora
        from gensim.models import LdaModel
        dictionary = corpora.Dictionary.load("models/lda_model/dictionary.dict")
        lda_model = LdaModel.load("models/lda_model/lda_model.model")
        is_fallback_lda = False
    except ImportError:
        with open("models/lda_model/dictionary.pkl", "rb") as f:
            dictionary = pickle.load(f)
        lda_model = FallbackLdaModel.load("models/lda_model")
        is_fallback_lda = True
        
    return lda_model, corpus, dictionary, is_fallback_lda

lda_model, corpus, dictionary, is_fallback_lda = load_lda()

# Load BERTopic Model (or Fallback KMeans)
@st.cache_resource
def load_bertopic():
    fallback_path = "models/bertopic_model/fallback"
    if os.path.exists(f"{fallback_path}/model.pkl"):
        model = FallbackTopicModel.load(fallback_path)
        is_fallback = True
    else:
        from bertopic import BERTopic
        model = BERTopic.load("models/bertopic_model/bertopic")
        is_fallback = False
    return model, is_fallback

topic_model, is_fallback_model = load_bertopic()

# Load metrics
with open("models/metrics.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)

# ──────────────────────────────────────────────────────────────────────────────
# Hàm helper cho Tab 4 — Evaluation Dashboard
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_eval_data():
    """Tải test_predictions.csv (140 bài VietnamNet đã predict)."""
    path = "data/processed/test_predictions.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def compute_silhouette(_embs, labels_tuple, sample_size=5000):
    """
    Tính Silhouette Score (cosine) trên sample để tránh OOM với 67k bài.
    labels_tuple: tuple (immutable) để Streamlit cache nhận ra thay đổi.
    """
    from sklearn.metrics import silhouette_score
    labels_arr = np.array(labels_tuple)
    mask = labels_arr != -1
    embs_valid = _embs[mask]
    labels_valid = labels_arr[mask]
    if len(embs_valid) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(embs_valid), size=sample_size, replace=False)
        embs_valid = embs_valid[idx]
        labels_valid = labels_valid[idx]
    unique = np.unique(labels_valid)
    if len(unique) < 2:
        return 0.0
    return float(silhouette_score(embs_valid, labels_valid, metric="cosine"))

def compute_topic_diversity(model_obj, n_top=10):
    """
    Topic Diversity = unique words / total words trong top-N từ khoá tất cả topics.
    Giá trị 1.0 = các topic hoàn toàn không trùng từ khoá.
    """
    topic_info = model_obj.get_topic_info()
    all_words = []
    for t_id in topic_info["Topic"]:
        if t_id == -1:
            continue
        words = [w for w, _ in model_obj.get_topic(t_id)[:n_top]]
        all_words.extend(words)
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)

def plot_confusion_heatmap(test_df, pred_col, model_name):
    """Vẽ heatmap: category gốc (VietnamNet) vs topic được dự đoán."""
    import plotly.graph_objects as go
    if pred_col not in test_df.columns:
        return None
    # Loại bỏ outlier -1 nếu còn
    df_cm = test_df[test_df[pred_col] != -1].copy()
    if df_cm.empty:
        return None
    cross_tab = pd.crosstab(
        df_cm["category"],
        df_cm[pred_col].astype(str).apply(lambda x: f"T{x}"),
    )
    fig = go.Figure(data=go.Heatmap(
        z=cross_tab.values,
        x=cross_tab.columns.tolist(),
        y=cross_tab.index.tolist(),
        colorscale="Blues",
        text=cross_tab.values,
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        title=f"<b>Confusion Matrix — {model_name}</b>",
        xaxis_title="Topic dự đoán",
        yaxis_title="Chuyên mục gốc (Ground Truth)",
        height=380,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig

# ──────────────────────────────────────────────────────────────────────────────

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Phân tích văn bản mới",
    "📁 Khám phá tập dữ liệu",
    "⚙️ So sánh & Đánh giá mô hình",
    "📊 Đánh Giá Toàn Diện",
])

# ================= TAB 1: SINGLE PREDICTION =================
with tab1:
    st.markdown("### 📝 Phân tích văn bản tự do")
    st.write("Nhập đoạn văn bản (bài báo, bình luận, v.v.) dưới đây để hệ thống tự động gán cụm chủ đề phù hợp nhất.")
    
    col_input, col_pred = st.columns([5, 4])
    
    with col_input:
        input_text = st.text_area(
            "Nội dung văn bản cần kiểm tra:",
            value="Giải bóng đá vô địch quốc gia V-League chứng kiến những trận cầu hấp dẫn với màn rượt đuổi tỷ số kịch tính của các đội bóng hàng đầu.",
            height=180,
            placeholder="Dán văn bản tiếng Việt tại đây..."
        )
        
        selected_model_type = st.selectbox(
            "Lực chọn mô hình phân cụm chủ đề:",
            ["BERTopic (KMeans Fallback)" if is_fallback_model else "BERTopic (Modern)", "LDA (Traditional)"]
        )
        
        analyze_btn = st.button("🚀 Dự đoán chủ đề", use_container_width=True)
        
    with col_pred:
        st.markdown("#### Kết quả nhận diện chủ đề")
        
        if analyze_btn:
            if not input_text.strip():
                st.error("Vui lòng nhập văn bản!")
            else:
                with st.spinner("Đang phân tích ngữ nghĩa và gán chủ đề..."):
                    cleaned_input = preprocess_text(input_text)
                    
                    if not cleaned_input:
                        st.warning("Văn bản sau khi làm sạch không còn từ khoá nào có nghĩa. Vui lòng nhập văn bản dài hơn.")
                    else:
                        st.info(f"Văn bản sau tiền xử lý: '{cleaned_input}'")
                        
                        if "BERTopic" in selected_model_type:
                            try:
                                from sentence_transformers import SentenceTransformer
                                st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                                input_emb = st_model.encode([cleaned_input])[0]
                            except Exception:
                                from sklearn.feature_extraction.text import TfidfVectorizer
                                all_texts = df["cleaned_text"].fillna("").astype(str).tolist()
                                vectorizer = TfidfVectorizer(max_features=embeddings.shape[1])
                                vectorizer.fit(all_texts)
                                input_emb = vectorizer.transform([cleaned_input]).toarray()[0]
                                
                            if is_fallback_model:
                                pred_topic, sims = topic_model.transform([cleaned_input], [input_emb])
                                pred_topic = pred_topic[0]
                                sim_score = sims[0]
                                
                                topic_words = topic_model.get_topic(pred_topic)
                                topic_name = topic_model.get_topic_info().loc[topic_model.get_topic_info()["Topic"] == pred_topic, "Name"].values[0]
                            else:
                                pred_topics, probs = topic_model.transform([cleaned_input], [input_emb])
                                pred_topic = pred_topics[0]
                                sim_score = probs[0] if probs is not None else 0.5
                                
                                topic_words = topic_model.get_topic(pred_topic)
                                topic_name = topic_model.get_topic_info().loc[topic_model.get_topic_info()["Topic"] == pred_topic, "Name"].values[0]
                            
                            st.markdown(f"""
                            <div class="card">
                                <h3>Chủ đề dự đoán: <span style="color:#6c5ce7;">{topic_name}</span></h3>
                                <p>Độ tương đồng ngữ nghĩa: <strong>{sim_score*100:.2f}%</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("**Từ khóa đặc trưng của chủ đề này:**")
                            badge_html = ""
                            for word, weight in topic_words[:8]:
                                badge_html += f"<span class='word-tag'>{word} ({weight:.2f})</span>"
                            st.markdown(badge_html, unsafe_allow_html=True)
                            
                            norms = np.linalg.norm(embeddings, axis=1)
                            norm_input = np.linalg.norm(input_emb)
                            
                            if norm_input > 0:
                                sim_matrix = np.dot(embeddings, input_emb) / (norms * norm_input)
                            else:
                                sim_matrix = np.zeros(len(embeddings))
                                
                            top_indices = sim_matrix.argsort()[::-1][:5]
                            
                            st.markdown("---")
                            st.markdown("📊 **Top 5 văn bản tương đồng nhất trong tập dữ liệu:**")
                            for idx in top_indices:
                                with st.expander(f"📌 {df.iloc[idx]['title']} (Tương đồng: {sim_matrix[idx]*100:.1f}%)"):
                                    st.write(f"**Chuyên mục:** {df.iloc[idx]['category']}")
                                    st.write(f"**Mô tả:** {df.iloc[idx]['description']}")
                                    st.caption(f"Đường dẫn: [Xem bài viết]({df.iloc[idx]['url']})")
                                    
                        else:
                            # LDA Prediction
                            if is_fallback_lda:
                                topic_probs = lda_model.get_document_topics(cleaned_input)
                            else:
                                bow = dictionary.doc2bow(cleaned_input.split())
                                topic_probs = lda_model.get_document_topics(bow)
                            
                            if not topic_probs:
                                st.warning("Không khớp được chủ đề nào phù hợp.")
                            else:
                                dominant_topic = sorted(topic_probs, key=lambda x: x[1], reverse=True)[0]
                                topic_id = dominant_topic[0]
                                prob = dominant_topic[1]
                                
                                words_with_weights = lda_model.show_topic(topic_id, topn=10)
                                topic_words_str = ", ".join([w[0] for w in words_with_weights[:4]])
                                topic_name = f"Chủ đề {topic_id} ({topic_words_str})"
                                
                                st.markdown(f"""
                                <div class="card">
                                    <h3>Chủ đề dự đoán: <span style="color:#00CEC9;">{topic_name}</span></h3>
                                    <p>Xác suất thuộc chủ đề: <strong>{prob*100:.2f}%</strong></p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown("**Từ khóa đặc trưng của chủ đề:**")
                                badge_html = ""
                                for word, weight in words_with_weights[:8]:
                                    badge_html += f"<span class='word-tag'>{word} ({weight:.3f})</span>"
                                st.markdown(badge_html, unsafe_allow_html=True)
                                
                                # Find top articles having highest probability
                                lda_probs = []
                                for doc_bow in corpus:
                                    doc_topics = dict(lda_model.get_document_topics(doc_bow))
                                    lda_probs.append(doc_topics.get(topic_id, 0.0))
                                    
                                df_temp = df.copy()
                                df_temp["lda_prob"] = lda_probs
                                top_lda_docs = df_temp.sort_values(by="lda_prob", ascending=False).head(5)
                                
                                st.markdown("---")
                                st.markdown("📊 **Top 5 văn bản có xác suất cao thuộc chủ đề này:**")
                                for _, r in top_lda_docs.iterrows():
                                    with st.expander(f"📌 {r['title']} (Xác suất: {r['lda_prob']*100:.1f}%)"):
                                        st.write(f"**Chuyên mục:** {r['category']}")
                                        st.write(f"**Mô tả:** {r['description']}")
                                        st.caption(f"Đường dẫn: [Xem bài viết]({r['url']})")

    # ── Batch Upload & Predict ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📂 Dự Đoán Hàng Loạt (Batch Prediction)")
    st.write("Tải lên file CSV (cần có cột `text`, `title`, `description` hoặc `content`) hoặc file TXT (mỗi dòng là 1 văn bản).")

    uploaded_file = st.file_uploader(
        "Chọn file CSV hoặc TXT:",
        type=["csv", "txt"],
        key="batch_uploader"
    )
    batch_model_select = st.selectbox(
        "Mô hình dự đoán batch:",
        ["BERTopic (KMeans Fallback)" if is_fallback_model else "BERTopic (HDBSCAN)", "LDA"],
        key="batch_model_select"
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                batch_df_raw = pd.read_csv(uploaded_file)
                text_col = None
                for _col in ["text", "content", "title", "description", "cleaned_text"]:
                    if _col in batch_df_raw.columns:
                        text_col = _col
                        break
                if text_col is None:
                    st.error(f"File CSV cần có một trong các cột: `text`, `content`, `title`, `description`. Các cột hiện tại: {', '.join(batch_df_raw.columns)}")
                    st.stop()
                texts_to_predict = batch_df_raw[text_col].fillna("").astype(str).tolist()
            else:
                raw_content = uploaded_file.read().decode("utf-8")
                texts_to_predict = [line.strip() for line in raw_content.split("\n") if line.strip()]
                batch_df_raw = pd.DataFrame({"text": texts_to_predict})
                text_col = "text"

            st.info(f"Đã tải **{len(texts_to_predict)}** văn bản từ `{uploaded_file.name}`.")
            run_batch_btn = st.button("🚀 Dự đoán hàng loạt", key="run_batch_btn", use_container_width=True)

            if run_batch_btn:
                with st.spinner(f"Đang xử lý {len(texts_to_predict)} văn bản..."):
                    cleaned_batch = [preprocess_text(t) for t in texts_to_predict]
                    batch_results = []

                    if "BERTopic" in batch_model_select:
                        try:
                            from sentence_transformers import SentenceTransformer
                            _st_model_batch = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                            batch_embs = _st_model_batch.encode(cleaned_batch, show_progress_bar=False)
                        except Exception:
                            from sklearn.feature_extraction.text import TfidfVectorizer
                            _all_ref = df["cleaned_text"].fillna("").astype(str).tolist()
                            _vect_b = TfidfVectorizer(max_features=embeddings.shape[1])
                            _vect_b.fit(_all_ref)
                            batch_embs = _vect_b.transform(cleaned_batch).toarray()

                        pred_topics_b, sims_b = topic_model.transform(cleaned_batch, batch_embs)

                        for _i, (_orig, _pred, _sim) in enumerate(zip(texts_to_predict, pred_topics_b, sims_b)):
                            _name_rows = topic_info_df.loc[topic_info_df["Topic"] == _pred, "Name"]
                            _tname = _name_rows.values[0] if len(_name_rows) > 0 else f"Topic {_pred}"
                            batch_results.append({
                                "Văn bản gốc": _orig[:120] + "..." if len(_orig) > 120 else _orig,
                                "Topic ID": int(_pred),
                                "Tên chủ đề": _tname,
                                "Độ tương đồng": f"{float(_sim) * 100:.1f}%",
                            })
                    else:
                        # LDA batch predict
                        for _orig, _cleaned in zip(texts_to_predict, cleaned_batch):
                            if is_fallback_lda:
                                _probs = lda_model.get_document_topics(_cleaned)
                            else:
                                _bow = dictionary.doc2bow(_cleaned.split())
                                _probs = lda_model.get_document_topics(_bow)

                            if _probs:
                                _dom = sorted(_probs, key=lambda x: x[1], reverse=True)[0]
                                _tid, _prob = _dom[0], _dom[1]
                                _kws = [w[0] for w in lda_model.show_topic(_tid, topn=3)]
                            else:
                                _tid, _prob, _kws = -1, 0.0, []

                            batch_results.append({
                                "Văn bản gốc": _orig[:120] + "..." if len(_orig) > 120 else _orig,
                                "Topic ID": int(_tid),
                                "Từ khoá chủ đề": ", ".join(_kws),
                                "Xác suất": f"{float(_prob) * 100:.1f}%",
                            })

                results_df_out = pd.DataFrame(batch_results)
                st.markdown("#### Kết Quả Dự Đoán Hàng Loạt:")
                st.dataframe(results_df_out, use_container_width=True)
                _csv_out = results_df_out.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label="⬇️ Tải kết quả về CSV",
                    data=_csv_out.encode("utf-8-sig"),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                    key="download_batch_csv"
                )

        except Exception as _e:
            st.error(f"Lỗi khi xử lý file: {_e}")

# ── Tính doc_topics và topic_info_df ở cấp module (dùng cho cả Tab 2 và Tab 4) ──
@st.cache_data
def get_coords(embs):
    coords_cache_path = "data/processed/coords_2d.npy"
    if os.path.exists(coords_cache_path):
        try:
            cached = np.load(coords_cache_path)
            if cached.shape[0] == embs.shape[0]:
                return cached
        except Exception:
            pass
    coords = reduce_dimensions_2d(embs)
    np.save(coords_cache_path, coords)
    return coords

coords = get_coords(embeddings)

if is_fallback_model:
    doc_topics = topic_model.labels_
    topic_info_df = topic_model.get_topic_info()
else:
    doc_topics = topic_model.topics_
    topic_info_df = topic_model.get_topic_info()
    # Soft-assign các outlier (-1) bằng cosine similarity với centroid
    _outlier_count = int((np.array(doc_topics) == -1).sum())
    if _outlier_count > 0:
        from src.topic_model import soft_assign_outliers
        doc_topics = soft_assign_outliers(embeddings, doc_topics, topic_model)

# ================= TAB 2: DATA EXPLORATION =================
with tab2:
    st.markdown("### 📁 Khám phá cụm dữ liệu trực quan")
    st.write("Trực quan hóa không gian ngữ nghĩa 2D của tập văn bản và thông tin các cụm chủ đề được khám phá.")

    if not is_fallback_model:
        _outlier_count_display = int((np.array(topic_model.topics_) == -1).sum())
        if _outlier_count_display > 0:
            st.caption(f"ℹ️ Đã soft-assign {_outlier_count_display:,} văn bản outlier vào chủ đề gần nhất.")

    col_chart, col_wordcloud = st.columns([5, 4])
    
    with col_chart:
        fig_scatter = plot_2d_embeddings(df, coords, doc_topics)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        fig_dist = plot_topic_distribution(topic_info_df)
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with col_wordcloud:
        st.markdown("#### Trực quan hóa Từ khóa Đặc trưng")
        st.write("Chọn chủ đề dưới đây để vẽ biểu đồ từ khoá đặc trưng tiêu biểu.")
        
        topic_choices = topic_info_df["Topic"].tolist()
        topic_names = topic_info_df["Name"].tolist()
        
        selected_topic_idx = st.selectbox(
            "Chọn chủ đề:",
            range(len(topic_choices)),
            format_func=lambda idx: topic_names[idx]
        )
        
        selected_topic = topic_choices[selected_topic_idx]
        
        topic_words = topic_model.get_topic(selected_topic)
        fig_wc = generate_wordcloud(topic_words, topic_names[selected_topic_idx])
        st.pyplot(fig_wc)
        
    st.markdown("---")
    st.markdown("#### 🔍 Tìm kiếm và duyệt dữ liệu bài viết")
    search_query = st.text_input("Nhập từ khoá tìm kiếm bài viết trong tập dữ liệu:")
    
    search_df = df.copy()
    search_df["Chủ đề"] = [topic_info_df.loc[topic_info_df["Topic"] == t, "Name"].values[0] for t in doc_topics]
    
    if search_query:
        search_df = search_df[
            search_df["title"].str.contains(search_query, case=False) |
            search_df["description"].str.contains(search_query, case=False)
        ]
        
    st.dataframe(
        search_df[["title", "category", "Chủ đề", "publish_date", "url"]].head(100),
        use_container_width=True
    )

    # ── Temporal Analysis ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📈 Xu Hướng Chủ Đề Theo Thời Gian")
    st.write("Biểu đồ phân bố số lượng bài viết theo chủ đề qua từng tháng (dựa trên cột `publish_date`).")

    @st.cache_data
    def compute_temporal_data(df_raw, doc_topics_tuple, topic_info_json):
        """Parse publish_date và group by month × topic."""
        topic_info = pd.read_json(io.StringIO(topic_info_json))
        doc_topics_arr = list(doc_topics_tuple)

        df_temp = df_raw.copy()
        df_temp["topic_label"] = [
            topic_info.loc[topic_info["Topic"] == t, "Name"].values[0]
            if t in topic_info["Topic"].values else f"Topic {t}"
            for t in doc_topics_arr
        ]
        # Parse dd/mm/yyyy (định dạng thực tế trong data)
        df_temp["parsed_date"] = pd.to_datetime(
            df_temp["publish_date"], format="%d/%m/%Y", errors="coerce"
        )
        # Fallback yyyy-mm-dd hoặc các định dạng khác
        mask_failed = df_temp["parsed_date"].isna()
        if mask_failed.any():
            df_temp.loc[mask_failed, "parsed_date"] = pd.to_datetime(
                df_temp.loc[mask_failed, "publish_date"], errors="coerce"
            )
        df_temp = df_temp.dropna(subset=["parsed_date"])
        if df_temp.empty:
            return pd.DataFrame()
        df_temp["year_month"] = df_temp["parsed_date"].dt.to_period("M").astype(str)
        return (
            df_temp.groupby(["year_month", "topic_label"])
            .size()
            .reset_index(name="count")
            .sort_values("year_month")
        )

    top_n_time = st.slider("Số chủ đề hiển thị trên biểu đồ:", min_value=2, max_value=7, value=5, key="temporal_slider")
    temporal_df = compute_temporal_data(
        df,
        tuple(int(t) for t in doc_topics),
        topic_info_df.to_json()
    )

    if temporal_df.empty:
        st.warning("⚠️ Không parse được `publish_date`. Kiểm tra định dạng ngày trong dữ liệu.")
    else:
        top_topics_time = (
            temporal_df.groupby("topic_label")["count"].sum()
            .nlargest(top_n_time).index.tolist()
        )
        temporal_filtered = temporal_df[temporal_df["topic_label"].isin(top_topics_time)]
        fig_temporal = px.line(
            temporal_filtered,
            x="year_month",
            y="count",
            color="topic_label",
            markers=True,
            title="Xu hướng số bài viết theo chủ đề qua thời gian",
            labels={"year_month": "Tháng/Năm", "count": "Số bài viết", "topic_label": "Chủ đề"},
            template="plotly_white",
        )
        fig_temporal.update_layout(
            xaxis=dict(tickangle=45),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=60, b=60),
        )
        st.plotly_chart(fig_temporal, use_container_width=True)
        total_dated = temporal_df["count"].sum()
        st.caption(f"Biểu đồ dựa trên {total_dated:,} bài viết có ngày xuất bản hợp lệ.")

# ================= TAB 3: MODEL COMPARISON =================
with tab3:
    st.markdown("### ⚙️ So sánh Mô hình: LDA vs BERTopic")
    st.write("Đánh giá hiệu suất phân cụm ngữ nghĩa dựa trên các chỉ số định lượng và định tính.")
    
    st.markdown("#### 1. Chỉ số liên kết ngữ nghĩa (Coherence Score)")
    st.write("Điểm Coherence ($C_v$) đo lường mức độ liên kết ngữ nghĩa giữa các từ khoá trong cùng chủ đề. Giá trị càng gần 1.0 càng tốt.")
    
    col_lda_met, col_bert_met = st.columns(2)
    with col_lda_met:
        st.metric(
            label="Coherence Score (c_v) - LDA",
            value=f"{metrics['lda']['coherence_cv']:.4f}"
        )
    with col_bert_met:
        st.metric(
            label="Coherence Score (c_v) - BERTopic",
            value=f"{metrics['bertopic']['coherence_cv']:.4f}",
            delta="Chất lượng tốt hơn" if metrics['bertopic']['coherence_cv'] > metrics['lda']['coherence_cv'] else None
        )
        
    st.markdown("#### 2. Đối chiếu các chủ đề được khám phá")
    
    col_lda_topics, col_bert_topics = st.columns(2)
    
    with col_lda_topics:
        st.markdown("##### 📌 Chủ đề trích xuất từ LDA:")
        lda_topic_data = []
        for t_id in range(lda_model.num_topics):
            words = [w[0] for w in lda_model.show_topic(t_id, topn=6)]
            lda_topic_data.append({
                "Chủ đề": f"Chủ đề {t_id}",
                "Từ khoá đặc trưng": ", ".join(words)
            })
        st.table(pd.DataFrame(lda_topic_data))
        
    with col_bert_topics:
        st.markdown("##### 📌 Chủ đề trích xuất từ BERTopic:")
        bert_topic_data = []
        for _, row in topic_info_df.iterrows():
            t_id = row["Topic"]
            t_words = [w[0] for w in topic_model.get_topic(t_id)[:6]]
            bert_topic_data.append({
                "Chủ đề": row["Name"].split("_")[0],
                "Số bài": row["Count"],
                "Từ khoá đặc trưng": ", ".join(t_words)
            })
        st.table(pd.DataFrame(bert_topic_data))
        
    st.markdown("---")
    st.markdown("#### 3. Bản đồ tương tác phân bố chủ đề của LDA (pyLDAvis)")
    
    if status["pyldavis"]:
        with open("models/lda_model/pyldavis_data.html", "r", encoding="utf-8") as f:
            pyldavis_html = f.read()

        components.html(pyldavis_html, height=800, width=1300, scrolling=True)
    else:
        st.info("Bản đồ tương tác pyLDAvis chỉ được tạo khi sử dụng Gensim trong môi trường đầy đủ. Chế độ chạy rút gọn (Sklearn Fallback) bỏ qua biểu đồ này.")

# ================= TAB 4: EVALUATION DASHBOARD =================
with tab4:
    st.markdown("### 📊 Đánh Giá Toàn Diện Mô Hình")
    st.write("Tổng hợp các chỉ số định lượng và kiểm thử độc lập trên tập dữ liệu VietnamNet (140 bài, 7 chuyên mục).")

    # ── Section 1: Bảng so sánh tổng hợp ──────────────────────────────────────
    st.markdown("#### 1. Bảng So Sánh Chỉ Số Đánh Giá")

    # Tính Silhouette và Topic Diversity
    current_labels = np.array(topic_model.labels_ if is_fallback_model else doc_topics)
    sil_score = compute_silhouette(embeddings, tuple(current_labels.tolist()))
    div_score = compute_topic_diversity(topic_model)
    lda_div   = compute_topic_diversity(lda_model)

    comparison_df = pd.DataFrame([
        {
            "Mô hình": "LDA (Gensim/Sklearn)",
            "Coherence c_v": f"{metrics['lda']['coherence_cv']:.4f}",
            "Silhouette Score": "—",
            "Topic Diversity": f"{lda_div:.4f}",
            "% Outlier (test set)": "0%",
            "Ghi chú": "Phân tích Bag-of-Words",
        },
        {
            "Mô hình": "BERTopic KMeans (Fallback)" if is_fallback_model else "BERTopic HDBSCAN",
            "Coherence c_v": f"{metrics['bertopic']['coherence_cv']:.4f}",
            "Silhouette Score": f"{sil_score:.4f}",
            "Topic Diversity": f"{div_score:.4f}",
            "% Outlier (test set)": "0% (sau soft-assign)" if not is_fallback_model else "0%",
            "Ghi chú": "Embedding SBERT 384-dim",
        },
    ])
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown("""
    > **Giải thích:**
    > - **Coherence c_v**: Giá trị càng gần 1.0 càng tốt — đo mức độ liên kết ngữ nghĩa trong từng topic.
    > - **Silhouette Score**: Từ -1 đến 1 — giá trị dương cho thấy cụm compact và tách biệt tốt.
    > - **Topic Diversity**: Tỷ lệ từ khoá unique / tổng từ khoá — càng cao nghĩa là các topic càng khác nhau.
    """)

    st.markdown("---")

    # ── Section 2: Confusion Matrix ────────────────────────────────────────────
    st.markdown("#### 2. Confusion Matrix — Phân Bố Chuyên Mục vs Topic Dự Đoán")
    st.write("So sánh **chuyên mục gốc** (nhãn thực tế từ VietnamNet) với **topic được mô hình gán** trên tập kiểm thử 140 bài.")

    test_pred_df = load_eval_data()
    if test_pred_df is not None:
        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            fig_cm_lda = plot_confusion_heatmap(test_pred_df, "lda_topic", "LDA")
            if fig_cm_lda:
                st.plotly_chart(fig_cm_lda, use_container_width=True)
            else:
                st.warning("Không có cột `lda_topic` trong test_predictions.csv.")
        with col_cm2:
            fig_cm_bert = plot_confusion_heatmap(test_pred_df, "bertopic_fallback", "BERTopic KMeans")
            if fig_cm_bert:
                st.plotly_chart(fig_cm_bert, use_container_width=True)
            else:
                st.warning("Không có cột `bertopic_fallback` trong test_predictions.csv.")

        st.markdown("---")
        st.markdown("#### 3. Phân Tích Chi Tiết Theo Chuyên Mục")
        st.write("Xem tỷ lệ bài viết của mỗi chuyên mục được gom vào cùng một topic (độ tập trung).")

        if "lda_topic" in test_pred_df.columns and "bertopic_fallback" in test_pred_df.columns:
            detail_rows = []
            for cat in sorted(test_pred_df["category"].unique()):
                sub = test_pred_df[test_pred_df["category"] == cat]
                n = len(sub)
                # LDA
                lda_counts = sub["lda_topic"].value_counts()
                lda_top = lda_counts.idxmax() if not lda_counts.empty else -1
                lda_pct = (lda_counts.max() / n * 100) if not lda_counts.empty else 0
                # KMeans
                km_sub = sub[sub["bertopic_fallback"] != -1]["bertopic_fallback"]
                if not km_sub.empty:
                    km_counts = km_sub.value_counts()
                    km_top = km_counts.idxmax()
                    km_pct = km_counts.max() / n * 100
                else:
                    km_top, km_pct = -1, 0.0
                detail_rows.append({
                    "Chuyên mục": cat,
                    "Số bài test": n,
                    "LDA — Topic chính": f"T{lda_top} ({lda_pct:.0f}%)",
                    "KMeans — Topic chính": f"T{km_top} ({km_pct:.0f}%)",
                })
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ Chưa có `data/processed/test_predictions.csv`. Chạy lệnh sau để tạo:")
        st.code("python scripts/test_model.py")

    st.markdown("---")

    # ── Section 4: pyLDAvis (chuyển từ Tab 3) ─────────────────────────────────
    st.markdown("#### 4. Bản Đồ Tương Tác Phân Bố Chủ Đề — pyLDAvis")
    if status["pyldavis"]:
        with open("models/lda_model/pyldavis_data.html", "r", encoding="utf-8") as _f:
            _pyldavis_html = _f.read()
        components.html(_pyldavis_html, height=800, width=1300, scrolling=True)
    else:
        st.info("pyLDAvis chỉ được tạo khi dùng Gensim đầy đủ. Hiện tại dùng Sklearn Fallback.")
