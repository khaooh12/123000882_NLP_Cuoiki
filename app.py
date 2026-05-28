import os
import sys
import html as _html
import pandas as pd
import numpy as np
import streamlit as st

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.preprocess import preprocess_text
from src.topic_model import FallbackTopicModel
from src.visualize import reduce_dimensions_2d, plot_upload_scatter, generate_wordcloud

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phân Cụm Chủ Đề Văn Bản Tiếng Việt",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .main-title {
        font-weight: 700;
        background: linear-gradient(135deg, #6C5CE7 0%, #00CEC9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle { text-align: center; color: #636e72; font-size: 1.1rem; margin-bottom: 2rem; }
    .word-tag {
        display: inline-block;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        background-color: rgba(108,92,231,0.1);
        color: #6c5ce7;
        border: 1px solid rgba(108,92,231,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Cấu hình")
n_topics = st.sidebar.slider("Số chủ đề:", min_value=3, max_value=12, value=7, step=1)
st.sidebar.markdown("---")
st.sidebar.markdown("🏛️ **MSSV:** 123000882  \n**Môn học:** NLP Cuối Khoá")

# ── Page title ─────────────────────────────────────────────────────────────────
st.markdown("<div class='main-title'>🗺️ BẢN ĐỒ CHỦ ĐỀ VĂN BẢN TIẾNG VIỆT</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Demo phân cụm chủ đề không giám sát · KMeans + TF-IDF</div>", unsafe_allow_html=True)

# ── Encode helper ──────────────────────────────────────────────────────────────
def encode_texts(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vect = TfidfVectorizer(max_features=384)
    return vect.fit_transform(texts).toarray()

# ── Pipeline ───────────────────────────────────────────────────────────────────
def run_pipeline(texts, n_topics):
    cleaned = [preprocess_text(t) for t in texts]
    valid_pairs = [(c, o) for c, o in zip(cleaned, texts) if c.strip()]
    if not valid_pairs:
        return None
    cleaned_valid, originals_valid = zip(*valid_pairs)
    cleaned_valid = list(cleaned_valid)
    originals_valid = list(originals_valid)
    filtered_out = len(texts) - len(cleaned_valid)

    embs = encode_texts(cleaned_valid)
    actual_k = min(n_topics, max(2, len(cleaned_valid) // 3))
    model = FallbackTopicModel(n_topics=actual_k)
    labels, _ = model.fit_transform(cleaned_valid, embs)
    coords = reduce_dimensions_2d(embs)

    return {
        "originals": originals_valid,
        "embs": embs,
        "labels": labels,
        "model": model,
        "coords": coords,
        "filtered_out": filtered_out,
    }

# ── Input section ──────────────────────────────────────────────────────────────
st.markdown("### 📥 Nhập Dữ Liệu")
tab_upload, tab_text = st.tabs(["📂 Upload File", "✏️ Nhập Văn Bản"])

with tab_upload:
    st.write(
        "Tải lên file **CSV** (cần cột `text`, `title`, `description` hoặc `content`) "
        "hoặc file **TXT** (mỗi dòng là 1 văn bản)."
    )
    uploaded = st.file_uploader("Chọn file CSV hoặc TXT:", type=["csv", "txt"])
    run_file_btn = st.button("🚀 Phân Tích File", use_container_width=True, disabled=(uploaded is None))

    if uploaded is not None and run_file_btn:
        try:
            if uploaded.name.endswith(".csv"):
                raw_df = pd.read_csv(uploaded)
                text_col = next(
                    (c for c in ["text", "content", "title", "description", "cleaned_text"]
                     if c in raw_df.columns), None
                )
                if text_col is None:
                    st.error(
                        f"File CSV phải có một trong các cột: text, content, title, description. "
                        f"Các cột hiện tại: {', '.join(raw_df.columns)}"
                    )
                    st.stop()
                raw_texts = raw_df[text_col].fillna("").astype(str).tolist()
            else:
                content = uploaded.read().decode("utf-8", errors="replace")
                raw_texts = [ln.strip() for ln in content.split("\n") if ln.strip()]

            if len(raw_texts) < 10:
                st.warning("Cần ít nhất **10 văn bản** để phân cụm. Vui lòng upload file lớn hơn.")
                st.stop()

            if len(raw_texts) > 2000:
                st.warning(f"File có {len(raw_texts):,} văn bản — chỉ xử lý **2,000 văn bản đầu** để đảm bảo hiệu năng.")
                raw_texts = raw_texts[:2000]

            with st.spinner("Đang xử lý: tiền xử lý → mã hóa TF-IDF → phân cụm KMeans → giảm chiều 2D..."):
                result = run_pipeline(raw_texts, n_topics)

            if result is None:
                st.error("Toàn bộ văn bản sau tiền xử lý đều rỗng. Vui lòng kiểm tra nội dung file.")
                st.stop()

            st.session_state["result"] = result
            st.session_state["file_name"] = uploaded.name

        except Exception as e:
            st.error(f"Lỗi khi xử lý file: {e}")

with tab_text:
    st.write(
        "Dán hoặc gõ trực tiếp đoạn văn bản. Mỗi **dòng** là một câu/đoạn văn."
    )
    pasted = st.text_area(
        "Nhập văn bản (mỗi dòng = 1 câu/đoạn):",
        height=220,
        placeholder="Ví dụ:\nHội đồng quản trị thông qua kế hoạch mở rộng thị trường.\nĐội tuyển Việt Nam giành chiến thắng 2-0 trước đối thủ.\n...",
    )
    run_text_btn = st.button("🚀 Phân Tích Văn Bản", use_container_width=True, disabled=not pasted.strip())

    if pasted.strip() and run_text_btn:
        raw_texts = [ln.strip() for ln in pasted.splitlines() if ln.strip()]
        with st.spinner("Đang xử lý: tiền xử lý → mã hóa TF-IDF → phân cụm KMeans → giảm chiều 2D..."):
            result = run_pipeline(raw_texts, n_topics)

        if result is None:
            st.error("Toàn bộ văn bản sau tiền xử lý đều rỗng. Vui lòng kiểm tra nội dung.")
            st.stop()

        st.session_state["result"] = result
        st.session_state["file_name"] = f"văn bản ({len(raw_texts)} dòng)"

# ── Hiển thị kết quả ───────────────────────────────────────────────────────────
if "result" in st.session_state:
    r = st.session_state["result"]
    fname = st.session_state.get("file_name", "")
    originals, embs, labels, model, coords, filtered_out = (
        r["originals"], r["embs"], r["labels"], r["model"], r["coords"], r["filtered_out"]
    )
    topic_info = model.get_topic_info()
    tid_to_name = dict(zip(topic_info["Topic"], topic_info["Name"]))

    if filtered_out > 0:
        st.caption(f"ℹ️ Đã lọc bỏ {filtered_out} văn bản rỗng sau tiền xử lý.")
    st.success(f"Đã phân cụm **{len(originals):,}** văn bản từ `{fname}` thành **{len(topic_info)}** chủ đề.")

    col_left, col_right = st.columns([4, 5])

    # ── Cột trái: danh sách chủ đề ──────────────────────────────────────────────
    with col_left:
        st.markdown("#### 📋 Danh Sách Chủ Đề")
        table_rows = []
        for _, row in topic_info.iterrows():
            kws = ", ".join([w for w, _ in model.get_topic(row["Topic"])[:5]])
            table_rows.append({
                "Chủ đề": row["Name"],
                "Số bài": int(row["Count"]),
                "Từ khoá tiêu biểu": kws,
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        # Download
        result_rows = [
            {
                "Văn bản gốc": orig[:200],
                "Topic ID": int(lbl),
                "Tên chủ đề": tid_to_name.get(int(lbl), f"Topic {lbl}"),
                "Từ khoá": ", ".join([w for w, _ in model.get_topic(int(lbl))[:5]]),
            }
            for orig, lbl in zip(originals, labels)
        ]
        csv_bytes = pd.DataFrame(result_rows).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "⬇️ Tải kết quả CSV",
            data=csv_bytes,
            file_name="clusters.csv",
            mime="text/csv",
        )

    # ── Cột phải: scatter plot + từ khoá ────────────────────────────────────────
    with col_right:
        st.markdown("#### 🗺️ Scatter Plot 2D — Màu sắc theo cụm")
        fig_scatter = plot_upload_scatter(originals, coords, labels, topic_info)
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("#### 🔑 Từ Khoá Đặc Trưng Mỗi Chủ Đề")
        topic_choices = topic_info["Topic"].tolist()
        topic_names = topic_info["Name"].tolist()
        sel_idx = st.selectbox(
            "Chọn chủ đề:",
            range(len(topic_choices)),
            format_func=lambda i: topic_names[i],
        )
        topic_words = model.get_topic(topic_choices[sel_idx])
        badge_html = "".join(
            f"<span class='word-tag'>{w} ({s:.3f})</span>"
            for w, s in topic_words[:12]
        )
        st.markdown(badge_html, unsafe_allow_html=True)
        fig_wc = generate_wordcloud(topic_words, topic_names[sel_idx])
        st.pyplot(fig_wc)

    # ── Văn bản gốc tô màu theo chủ đề ─────────────────────────────────────────
    _TOPIC_COLORS = [
        "#FFE0B2", "#C8E6C9", "#BBDEFB", "#F8BBD9", "#E1BEE7",
        "#B2EBF2", "#FFF9C4", "#FFCCBC", "#D7CCC8", "#CFD8DC",
        "#DCEDC8", "#F0F4C3",
    ]
    _BORDER_COLORS = [
        "#FB8C00", "#43A047", "#1E88E5", "#E91E63", "#8E24AA",
        "#00ACC1", "#F9A825", "#F4511E", "#6D4C41", "#546E7A",
        "#7CB342", "#C0CA33",
    ]
    tid_to_color = {
        int(row["Topic"]): _TOPIC_COLORS[i % len(_TOPIC_COLORS)]
        for i, (_, row) in enumerate(topic_info.iterrows())
    }
    tid_to_border = {
        int(row["Topic"]): _BORDER_COLORS[i % len(_BORDER_COLORS)]
        for i, (_, row) in enumerate(topic_info.iterrows())
    }

    st.markdown("---")
    st.markdown("#### 🎨 Văn Bản Gốc Tô Màu Theo Chủ Đề")

    legend_html = "<div style='display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px;'>"
    for _, row in topic_info.iterrows():
        color = tid_to_color[int(row["Topic"])]
        border = tid_to_border[int(row["Topic"])]
        legend_html += (
            f"<span style='background:{color};border-left:4px solid {border};"
            f"padding:4px 12px;border-radius:4px;font-size:0.82rem;font-weight:600;'>"
            f"{row['Name']}</span>"
        )
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)

    blocks_html = ""
    for text, label in zip(originals, labels):
        color = tid_to_color.get(int(label), "#F5F5F5")
        border = tid_to_border.get(int(label), "#BDBDBD")
        topic_name = tid_to_name.get(int(label), f"Topic {label}")
        safe_text = _html.escape(str(text))
        blocks_html += (
            f"<div style='background:{color};padding:10px 14px;margin:5px 0;"
            f"border-radius:6px;border-left:4px solid {border};line-height:1.7;'>"
            f"<span style='font-size:0.72rem;color:#444;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:0.06em;'>{topic_name}</span><br>"
            f"<span style='font-size:0.95rem;color:#212121;'>{safe_text}</span>"
            f"</div>"
        )
    st.markdown(blocks_html, unsafe_allow_html=True)

else:
    st.info("Hãy upload file hoặc nhập văn bản để bắt đầu phân cụm.")
