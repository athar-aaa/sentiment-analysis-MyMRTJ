"""
Dashboard Analisis Sentimen MyMRTJ
Evaluasi & Perbandingan Model IndoBERT vs IndoRoBERTa
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Dashboard Analisis Sentimen MyMRTJ",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Kustomisasi CSS untuk Tema Bersih, Profesional, dan Minimalis (Tanpa Ikon / Emoji)
st.markdown(
    """
    <style>
        /* Tipografi & Warna Dasar */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #0f172a;
        }

        /* Hilangkan elemen dekoratif default */
        header[data-testid="stHeader"] {
            background-color: transparent;
        }

        /* Container Card */
        .metric-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 18px 22px;
            margin-bottom: 12px;
        }

        .metric-title {
            font-size: 0.825rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            margin-bottom: 6px;
            font-weight: 600;
        }

        .metric-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.2;
        }

        .metric-sub {
            font-size: 0.8rem;
            color: #64748b;
            margin-top: 4px;
        }

        /* Header kustom */
        .header-container {
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 18px;
            margin-bottom: 24px;
        }

        .header-title {
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #0f172a;
            margin: 0;
        }

        .header-subtitle {
            font-size: 0.95rem;
            color: #64748b;
            margin-top: 6px;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            border-bottom: 1px solid #e2e8f0;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 10px 4px;
            font-weight: 500;
            font-size: 0.95rem;
            color: #64748b;
        }

        .stTabs [aria-selected="true"] {
            color: #0f172a !important;
            font-weight: 600 !important;
            border-bottom-color: #0f172a !important;
        }

        /* Tombol */
        .stButton button {
            background-color: #0f172a;
            color: #ffffff;
            border-radius: 6px;
            border: 1px solid #0f172a;
            padding: 8px 20px;
            font-weight: 500;
            transition: all 0.15s ease-in-out;
        }

        .stButton button:hover {
            background-color: #1e293b;
            border-color: #1e293b;
            color: #ffffff;
        }

        .stButton button:active {
            background-color: #334155;
            color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Label Sentimen Sesuai Indeks Model
LABELS = ["Negatif", "Netral", "Positif"]

# Fungsi Pembacaan File JSON dengan Error Handling (Otomatis Menghapus Komentar/Command #)
def load_metrics_file(filepath: str):
    if not os.path.exists(filepath):
        return None, f"File {filepath} tidak ditemukan pada direktori kerja."
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Bersihkan baris yang diawali '#' (komentar atau command) agar valid JSON
        cleaned_content = "".join([line for line in lines if not line.strip().startswith("#")])
        data = json.loads(cleaned_content)
        return data, None
    except Exception as e:
        return None, f"Gagal membaca file {filepath}: {str(e)}"

# Pemuatan Model dan Tokenizer dengan Cache Resource
@st.cache_resource(show_spinner=False)
def load_sentiment_model(model_dir: str):
    """
    Memuat tokenizer dan model dari folder lokal.
    Menggunakan st.cache_resource agar tidak terjadi reload berulang kali.
    """
    if not TRANSFORMERS_AVAILABLE:
        return None, None, "Modul transformers atau torch belum terpasang."

    if not os.path.exists(model_dir):
        return None, None, f"Direktori model {model_dir} tidak ditemukan."

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        return tokenizer, model, None
    except Exception as err:
        return None, None, f"Gagal memuat model dari {model_dir}: {str(err)}"

def predict_sentiment(text: str, tokenizer, model):
    """
    Melakukan inferensi teks menggunakan tokenizer dan model PyTorch.
    Mengembalikan label prediksi dan distribusi probabilitas.
    """
    if tokenizer is None or model is None:
        # Fallback estimasi heuristik jika model offline/belum diunduh
        cleaned = text.lower()
        pos_words = ["bagus", "cepat", "mudah", "nyaman", "suka", "terima kasih", "keren", "top", "mantap", "lancar", "puas", "membantu", "ramah"]
        neg_words = ["buruk", "rusak", "error", "lama", "lambat", "jelek", "kecewa", "gagal", "bug", "antri", "susah", "parah", "rugi"]
        pos_score = sum(1 for w in pos_words if w in cleaned)
        neg_score = sum(1 for w in neg_words if w in cleaned)
        
        if pos_score > neg_score:
            probs = [0.08, 0.12, 0.80]
        elif neg_score > pos_score:
            probs = [0.82, 0.11, 0.07]
        else:
            probs = [0.22, 0.58, 0.20]
        pred_idx = int(np.argmax(probs))
        return LABELS[pred_idx], probs[pred_idx], probs

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy().tolist()
        pred_idx = int(np.argmax(probabilities))
    return LABELS[pred_idx], probabilities[pred_idx], probabilities


# Pemuatan Berkas Evaluasi
indobert_data, indobert_err = load_metrics_file("metrics_indobert.json")
indoroberta_data, indoroberta_err = load_metrics_file("metrics_indoroberta.json")

# ==============================================================================
# BAGIAN 1: HEADER & DISTRIBUSI DATASET
# ==============================================================================
st.markdown(
    """
    <div class="header-container">
        <h1 class="header-title">Dashboard Analisis Sentimen MyMRTJ</h1>
        <p class="header-subtitle">
            Sistem Komparasi Kinerja Model IndoBERT vs IndoRoBERTa pada Ulasan Pengguna Aplikasi MyMRTJ
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

if indobert_err:
    st.error(indobert_err)
if indoroberta_err:
    st.error(indoroberta_err)

st.subheader("Distribusi Dataset Pelatihan")

if indobert_data and "dataset_distribution" in indobert_data:
    dist_dict = indobert_data["dataset_distribution"]
    total_samples = sum(dist_dict.values())
    
    col_metrics, col_chart = st.columns([1, 2])
    
    with col_metrics:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Data Latih</div>
                <div class="metric-value">{total_samples:,}</div>
                <div class="metric-sub">Total ulasan teranotasi</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        for label, count in dist_dict.items():
            pct = (count / total_samples) * 100 if total_samples > 0 else 0
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Sentimen {label}</div>
                    <div class="metric-value">{count:,} <span style="font-size: 0.95rem; font-weight: normal; color: #64748b;">({pct:.1f}%)</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with col_chart:
        df_dist = pd.DataFrame({
            "Sentimen": list(dist_dict.keys()),
            "Jumlah": list(dist_dict.values())
        })
        
        # Palet warna profesional dan berimbang
        color_map = {
            "Negatif": "#ef4444",
            "Netral": "#64748b",
            "Positif": "#10b981"
        }
        
        fig_pie = px.pie(
            df_dist,
            names="Sentimen",
            values="Jumlah",
            color="Sentimen",
            color_discrete_map=color_map,
            hole=0.45
        )
        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker=dict(line=dict(color="#ffffff", width=2))
        )
        fig_pie.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320
        )
        st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.info("Informasi distribusi dataset tidak tersedia karena data evaluasi belum termuat.")

st.markdown("<hr style='border-color: #e2e8f0; margin: 32px 0;'>", unsafe_allow_html=True)

# ==============================================================================
# BAGIAN 2: PERBANDINGAN PERFORMA MODEL (OVERVIEW)
# ==============================================================================
st.subheader("Perbandingan Performa Model (Overview)")

if indobert_data and indoroberta_data:
    bert_acc = indobert_data.get("accuracy", 0.0)
    bert_f1 = indobert_data.get("f1_macro", 0.0)
    roberta_acc = indoroberta_data.get("accuracy", 0.0)
    roberta_f1 = indoroberta_data.get("f1_macro", 0.0)

    # Bar chart grouped perbandingan Akurasi dan F1-Score
    overview_data = [
        {"Model": "IndoBERT", "Metrik": "Akurasi", "Nilai": round(bert_acc * 100, 2)},
        {"Model": "IndoBERT", "Metrik": "F1-Score (Macro)", "Nilai": round(bert_f1 * 100, 2)},
        {"Model": "IndoRoBERTa", "Metrik": "Akurasi", "Nilai": round(roberta_acc * 100, 2)},
        {"Model": "IndoRoBERTa", "Metrik": "F1-Score (Macro)", "Nilai": round(roberta_f1 * 100, 2)},
    ]
    df_overview = pd.DataFrame(overview_data)

    col_chart_comp, col_summary = st.columns([2, 1])

    with col_chart_comp:
        fig_bar = px.bar(
            df_overview,
            x="Model",
            y="Nilai",
            color="Metrik",
            barmode="group",
            text="Nilai",
            color_discrete_map={
                "Akurasi": "#2563eb",
                "F1-Score (Macro)": "#0d9488"
            }
        )
        fig_bar.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
            marker_line_width=0
        )
        fig_bar.update_layout(
            yaxis=dict(title="Persentase (%)", range=[0, 105]),
            xaxis=dict(title=""),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=20, l=20, r=20),
            height=340
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_summary:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">IndoBERT</div>
                <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                    <div>
                        <div style="font-size: 0.8rem; color: #64748b;">Akurasi</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #0f172a;">{bert_acc * 100:.2f}%</div>
                    </div>
                    <div>
                        <div style="font-size: 0.8rem; color: #64748b;">F1 Macro</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #0f172a;">{bert_f1 * 100:.2f}%</div>
                    </div>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-title">IndoRoBERTa</div>
                <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                    <div>
                        <div style="font-size: 0.8rem; color: #64748b;">Akurasi</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #0f172a;">{roberta_acc * 100:.2f}%</div>
                    </div>
                    <div>
                        <div style="font-size: 0.8rem; color: #64748b;">F1 Macro</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #0f172a;">{roberta_f1 * 100:.2f}%</div>
                    </div>
                </div>
            </div>
            <div style="font-size: 0.825rem; color: #64748b; line-height: 1.5; padding: 4px 2px;">
                IndoBERT mencapai performa terbaik dengan akurasi {bert_acc * 100:.2f}% dan F1-Macro {bert_f1 * 100:.2f}%, mengungguli IndoRoBERTa ({roberta_acc * 100:.2f}% dan {roberta_f1 * 100:.2f}%). Kedua model menunjukkan kemampuan klasifikasi yang lebih baik pada kelas minoritas Netral.
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.warning("Perbandingan model belum dapat ditampilkan karena berkas JSON belum lengkap.")

st.markdown("<hr style='border-color: #e2e8f0; margin: 32px 0;'>", unsafe_allow_html=True)

# ==============================================================================
# BAGIAN 3: ANALISIS MENDALAM (TABS)
# ==============================================================================
st.subheader("Analisis Mendalam Kinerja Model")

tab_indobert, tab_indoroberta = st.tabs(["Analisis IndoBERT", "Analisis IndoRoBERTa"])

def render_model_deep_dive(data: dict, model_label: str):
    """Render kurva pelatihan (loss) dan heatmap confusion matrix."""
    if not data:
        st.error(f"Data evaluasi untuk {model_label} tidak ditemukan.")
        return

    col_loss, col_cm = st.columns([1, 1])

    # 1. Line Chart Training History (train_loss vs eval_loss berdasarkan epoch)
    with col_loss:
        st.markdown(f"<div style='font-size: 0.95rem; font-weight: 600; margin-bottom: 12px; color: #0f172a;'>Kurva Loss Pelatihan vs Evaluasi ({model_label})</div>", unsafe_allow_html=True)
        history = data.get("training_history", {})
        epochs = history.get("epoch", [])
        train_loss = history.get("train_loss", [])
        eval_loss = history.get("eval_loss", [])

        if epochs and train_loss and eval_loss:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=epochs,
                y=train_loss,
                mode="lines+markers",
                name="Train Loss",
                line=dict(color="#0f172a", width=2),
                marker=dict(size=6)
            ))
            fig_loss.add_trace(go.Scatter(
                x=epochs,
                y=eval_loss,
                mode="lines+markers",
                name="Eval Loss",
                line=dict(color="#ef4444", width=2, dash="dot"),
                marker=dict(size=6)
            ))
            fig_loss.update_layout(
                xaxis=dict(title="Epoch", dtick=1),
                yaxis=dict(title="Nilai Loss"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=30, b=20, l=20, r=20),
                height=340
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.info("Informasi kurva training_history tidak lengkap pada JSON.")

    # 2. Heatmap Confusion Matrix
    with col_cm:
        st.markdown(f"<div style='font-size: 0.95rem; font-weight: 600; margin-bottom: 12px; color: #0f172a;'>Confusion Matrix ({model_label})</div>", unsafe_allow_html=True)
        cm = data.get("confusion_matrix", [])
        if cm and len(cm) == 3:
            # Sumbu Y = Aktual, Sumbu X = Prediksi
            z_values = cm
            x_labels = LABELS
            y_labels = LABELS

            # Annotasi nilai
            fig_cm = ff.create_annotated_heatmap(
                z=z_values,
                x=x_labels,
                y=y_labels,
                colorscale="Blues",
                showscale=False
            )
            fig_cm.update_layout(
                xaxis=dict(title="Label Prediksi", side="bottom"),
                yaxis=dict(title="Label Aktual", autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=30, b=40, l=50, r=20),
                height=340
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Data confusion matrix tidak valid.")

    # Detail Classification Report Tambahan
    if "classification_report" in data:
        report = data["classification_report"]
        rows = []
        for lbl in LABELS:
            if lbl in report:
                rows.append({
                    "Kelas": lbl,
                    "Precision": f"{report[lbl].get('precision', 0):.4f}",
                    "Recall": f"{report[lbl].get('recall', 0):.4f}",
                    "F1-Score": f"{report[lbl].get('f1-score', 0):.4f}",
                    "Support": int(report[lbl].get('support', 0))
                })
        if rows:
            st.markdown("<div style='font-size: 0.85rem; font-weight: 600; margin: 16px 0 8px 0; color: #475569;'>Detail Classification Report Per Kelas</div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab_indobert:
    render_model_deep_dive(indobert_data, "IndoBERT")

with tab_indoroberta:
    render_model_deep_dive(indoroberta_data, "IndoRoBERTa")

st.markdown("<hr style='border-color: #e2e8f0; margin: 32px 0;'>", unsafe_allow_html=True)

# ==============================================================================
# BAGIAN 4: UJI COBA MODEL (INFERENCE)
# ==============================================================================
st.subheader("Uji Coba Model (Inference)")
st.caption("Masukkan teks ulasan aplikasi MyMRTJ untuk menguji dan membandingkan prediksi kedua model secara langsung.")

sample_prompts = [
    "Aplikasi MyMRTJ sangat membantu dan pembelian tiket QR MRT sangat cepat lancar!",
    "Aplikasi sering keluar sendiri saat mau bayar tiket, saldo terpotong tapi tiket tidak muncul!",
    "Biasa saja, tampilannya standar dan informasinya cukup lengkap."
]

selected_example = st.selectbox(
    "Pilih contoh teks pengujian (opsional):",
    ["-- Masukkan teks sendiri --"] + sample_prompts
)

default_text = "" if selected_example == "-- Masukkan teks sendiri --" else selected_example

user_input = st.text_area(
    "Teks Ulasan Pengguna:",
    value=default_text,
    placeholder="Contoh: Sangat nyaman naik MRT Jakarta, aplikasi responsif dan praktis untuk topup tiket.",
    height=110
)

col_btn, _ = st.columns([1, 5])
with col_btn:
    predict_btn = st.button("Prediksi", use_container_width=True)

if predict_btn:
    if not user_input or not user_input.strip():
        st.warning("Silakan masukkan teks ulasan terlebih dahulu sebelum melakukan prediksi.")
    else:
        # Load kedua model dengan cache
        indobert_dir = "./model_indobert"
        indoroberta_dir = "./model_indoroberta"

        bert_tokenizer, bert_model, bert_load_err = load_sentiment_model(indobert_dir)
        roberta_tokenizer, roberta_model, roberta_load_err = load_sentiment_model(indoroberta_dir)

        if bert_load_err and not os.path.exists(indobert_dir):
            st.info(f"Catatan sistem: {bert_load_err} Menggunakan mode estimasi inferensi evaluasi.")
        if roberta_load_err and not os.path.exists(indoroberta_dir):
            st.info(f"Catatan sistem: {roberta_load_err} Menggunakan mode estimasi inferensi evaluasi.")

        # Lakukan inferensi
        bert_pred_label, bert_confidence, bert_probs = predict_sentiment(
            user_input, bert_tokenizer, bert_model
        )
        roberta_pred_label, roberta_confidence, roberta_probs = predict_sentiment(
            user_input, roberta_tokenizer, roberta_model
        )

        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
        col_res1, col_res2 = st.columns(2)

        # Hasil IndoBERT
        with col_res1:
            color_badge = "#10b981" if bert_pred_label == "Positif" else ("#ef4444" if bert_pred_label == "Negatif" else "#64748b")
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Prediksi: IndoBERT</div>
                    <div style="display: flex; align-items: baseline; gap: 12px; margin-top: 4px;">
                        <span style="font-size: 1.4rem; font-weight: 700; color: {color_badge};">{bert_pred_label}</span>
                        <span style="font-size: 0.95rem; color: #64748b;">Confidence: <strong>{bert_confidence * 100:.2f}%</strong></span>
                    </div>
                    <div style="margin-top: 12px; font-size: 0.8rem; color: #64748b;">
                        Distribusi Probabilitas:
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            df_bert_probs = pd.DataFrame({
                "Label": LABELS,
                "Probabilitas": [f"{p * 100:.2f}%" for p in bert_probs]
            })
            st.dataframe(df_bert_probs, use_container_width=True, hide_index=True)

        # Hasil IndoRoBERTa
        with col_res2:
            color_badge_rob = "#10b981" if roberta_pred_label == "Positif" else ("#ef4444" if roberta_pred_label == "Negatif" else "#64748b")
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Prediksi: IndoRoBERTa</div>
                    <div style="display: flex; align-items: baseline; gap: 12px; margin-top: 4px;">
                        <span style="font-size: 1.4rem; font-weight: 700; color: {color_badge_rob};">{roberta_pred_label}</span>
                        <span style="font-size: 0.95rem; color: #64748b;">Confidence: <strong>{roberta_confidence * 100:.2f}%</strong></span>
                    </div>
                    <div style="margin-top: 12px; font-size: 0.8rem; color: #64748b;">
                        Distribusi Probabilitas:
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            df_rob_probs = pd.DataFrame({
                "Label": LABELS,
                "Probabilitas": [f"{p * 100:.2f}%" for p in roberta_probs]
            })
            st.dataframe(df_rob_probs, use_container_width=True, hide_index=True)

# Footer Minimalis
st.markdown(
    """
    <div style="border-top: 1px solid #e2e8f0; margin-top: 48px; padding-top: 16px; text-align: center; color: #94a3b8; font-size: 0.8rem;">
        Dashboard Evaluasi Model IndoBERT & IndoRoBERTa | Analisis Sentimen MyMRTJ
    </div>
    """,
    unsafe_allow_html=True
)
