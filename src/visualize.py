import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import os

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False


def _find_vietnamese_font():
    """Tìm font hỗ trợ tiếng Việt có trên hệ thống (Windows / Linux / macOS)."""
    candidates = [
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        # Linux (Ubuntu/Debian)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        # macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None  # WordCloud dùng font mặc định nếu None


def reduce_dimensions_2d(embeddings):
    """Giảm chiều dữ liệu về 2D bằng UMAP hoặc TSNE/PCA dự phòng"""
    print("  [Visualization] Đang thực hiện giảm chiều đặc trưng về 2D...")
    try:
        from umap import UMAP
        reducer = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='cosine', random_state=42)
        coords = reducer.fit_transform(embeddings)
        print("  -> Giảm chiều bằng UMAP thành công.")
        return coords
    except Exception as e:
        print(f"  [Visualization] Lỗi hoặc không import được UMAP ({e}). Chuyển sang TSNE dự phòng...")
        try:
            from sklearn.manifold import TSNE
            perp = min(30, max(5, len(embeddings) // 5))
            reducer = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
            coords = reducer.fit_transform(embeddings)
            print("  -> Giảm chiều bằng t-SNE thành công.")
            return coords
        except Exception as tsne_err:
            print(f"  [Visualization] Lỗi t-SNE ({tsne_err}). Chuyển sang PCA dự phòng...")
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
            coords = reducer.fit_transform(embeddings)
            print("  -> Giảm chiều bằng PCA thành công.")
            return coords

def plot_2d_embeddings(df, coords, topic_labels):
    """Vẽ biểu đồ Plotly Scatter 2D tương tác đại diện cho các tài liệu và chủ đề"""
    plot_df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "Chuyên mục gốc": df["category"],
        "Tiêu đề": df["title"],
        "Chủ đề dự đoán": [f"Chủ đề {t}" for t in topic_labels],
        "Tóm tắt": df["description"].apply(lambda x: x[:100] + "..." if isinstance(x, str) and len(x) > 100 else x)
    })
    
    plot_df = plot_df.sort_values(by="Chủ đề dự đoán")
    
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="Chủ đề dự đoán",
        symbol="Chuyên mục gốc",
        hover_name="Tiêu đề",
        hover_data={"x": False, "y": False, "Chủ đề dự đoán": True, "Chuyên mục gốc": True, "Tóm tắt": True},
        title="Bản đồ phân bố văn bản (Không gian đặc trưng 2D)",
        labels={"Chủ đề dự đoán": "Chủ đề", "Chuyên mục gốc": "Chuyên mục gốc"},
        template="plotly_white"
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def generate_wordcloud(topic_words, topic_name):
    """Tạo biểu đồ Word Cloud (hoặc Bar chart nếu thiếu wordcloud) cho chủ đề"""
    word_freq = {word: score for word, score in topic_words if score > 0}
    
    if not word_freq:
        word_freq = {"trống": 1.0}
        
    if HAS_WORDCLOUD:
        font_path = _find_vietnamese_font()
        wc_kwargs = dict(
            background_color="white",
            width=800,
            height=400,
            max_words=50,
            colormap="plasma",
            prefer_horizontal=0.7
        )
        if font_path:
            wc_kwargs["font_path"] = font_path
        wc = WordCloud(**wc_kwargs)
        wc.generate_from_frequencies(word_freq)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud của {topic_name}", fontsize=16, pad=10)
        plt.tight_layout()
        return fig
    else:
        # Fallback to horizontal bar chart in matplotlib
        fig, ax = plt.subplots(figsize=(10, 5))
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1])[-15:]
        words = [w[0] for w in sorted_words]
        scores = [w[1] for w in sorted_words]
        
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(words)))
        bars = ax.barh(words, scores, color=colors, edgecolor='none')
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + (max(scores)*0.01), bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                    va='center', ha='left', fontsize=9, color='#2d3748')
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e0')
        ax.spines['bottom'].set_color('#cbd5e0')
        ax.tick_params(colors='#4a5568')
        ax.set_title(f"Từ khóa đặc trưng tiêu biểu - {topic_name}", fontsize=14, pad=15, color='#2d3748', weight='bold')
        plt.tight_layout()
        return fig

def plot_upload_scatter(texts, coords, labels, topic_info_df):
    """Scatter plot 2D cho corpus được upload (không yêu cầu cột category)."""
    topic_id_to_name = dict(zip(topic_info_df["Topic"], topic_info_df["Name"]))
    plot_df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "Chủ đề": [topic_id_to_name.get(int(t), f"Topic {t}") for t in labels],
        "Văn bản": [str(t)[:100] + "..." if len(str(t)) > 100 else str(t) for t in texts],
    })
    plot_df = plot_df.sort_values("Chủ đề")
    fig = px.scatter(
        plot_df,
        x="x", y="y",
        color="Chủ đề",
        hover_name="Văn bản",
        hover_data={"x": False, "y": False, "Chủ đề": True},
        title="Bản đồ phân bố văn bản đã upload (2D)",
        template="plotly_white",
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=0.5, color="DarkSlateGrey")))
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_topic_distribution(topic_counts_df):
    """Vẽ biểu đồ hình cột thống kê số lượng văn bản trong mỗi chủ đề"""
    topic_counts_df = topic_counts_df.sort_values(by="Count", ascending=False)
    
    fig = px.bar(
        topic_counts_df,
        x="Name",
        y="Count",
        color="Count",
        color_continuous_scale="Viridis",
        title="Phân bố số lượng văn bản theo chủ đề",
        labels={"Name": "Tên chủ đề", "Count": "Số lượng bài viết"},
        text="Count",
        template="plotly_white"
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis=dict(tickangle=30),
        margin=dict(l=20, r=20, t=50, b=20),
        coloraxis_showscale=False
    )
    return fig
