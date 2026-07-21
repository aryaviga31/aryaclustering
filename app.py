import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Konfigurasi Halaman Utama
st.set_page_config(
    page_title="Analisis Clustering Cacat Produk",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏭 Analisis Clustering Cacat Produk Industri Manufaktur")
st.markdown("Aplikasi interaktif untuk segmentasi cacat produksi berbasis *Machine Learning* (K-Means Clustering).")

# 2. Fungsi Memuat Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("defects_data.csv")
        severity_mapping = {'Minor': 1, 'Moderate': 2, 'Critical': 3}
        df['severity_score'] = df['severity'].map(severity_mapping)
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is not None:
    # 3. Pengaturan Sidebar (Model Parameter)
    st.sidebar.header("Pengaturan Model")
    optimal_k = st.sidebar.slider("Pilih Jumlah Klaster (K):", min_value=2, max_value=5, value=3)

    # 4. Pemrosesan Machine Learning (K-Means)
    X = df[['repair_cost', 'severity_score']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans_model = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    df['cluster_kmeans'] = kmeans_model.fit_predict(X_scaled)

    # Hitung profil rata-rata untuk analisis otomatis
    profil_kmeans = df.groupby('cluster_kmeans')[['repair_cost', 'severity_score']].mean()
    profil_kmeans['jumlah_sampel'] = df['cluster_kmeans'].value_counts()

    # 5. Pembuatan Tab Dashboard
    tab1, tab2, tab3 = st.tabs(["📊 Visualisasi & Data", "💡 Interpretasi & Insight Bisnis", "🛠️ Prediksi Data Baru"])

    with tab1:
        st.header("Sebaran Klaster Cacat Manufaktur")
        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(10, 6.5))
            sns.scatterplot(
                x=df['repair_cost'],
                y=df['severity_score'],
                hue=df['cluster_kmeans'],
                palette='Set1',
                s=100,
                alpha=0.85,
                edgecolor='black',
                linewidth=0.5,
                ax=ax
            )
            ax.set_title(f'Hasil Segmentasi Kasus Cacat Menggunakan K-Means (K={optimal_k})', fontsize=14, fontweight='bold')
            ax.set_xlabel('Biaya Perbaikan Produk ($ / Repair Cost)', fontsize=12)
            ax.set_ylabel('Tingkat Keparahan Cacat (Severity Score)', fontsize=12)
            ax.set_yticks([1, 2, 3])
            ax.set_yticklabels(['1 (Minor)', '2 (Moderate)', '3 (Critical)'])
            ax.legend(title='Klaster K-Means', loc='upper left')
            st.pyplot(fig)

        with col2:
            st.subheader("Statistik Rata-rata Tiap Klaster")
            st.dataframe(profil_kmeans[['repair_cost', 'severity_score']].style.format({"repair_cost": "${:.2f}", "severity_score": "{:.2f}"}))

            st.subheader("Jumlah Sampel per Klaster")
            st.bar_chart(df['cluster_kmeans'].value_counts())

        st.subheader("Sampel Data Hasil Clustering")
        st.dataframe(df[['defect_id', 'product_id', 'defect_type', 'severity', 'repair_cost', 'cluster_kmeans']].head(20))

    with tab2:
        st.header("💡 Interpretasi & Insight Bisnis Mendalam")
        st.markdown(f"Berikut adalah hasil ekstrak karakteristik mendalam dari **{optimal_k} kelompok (klaster)** cacat produk yang terbentuk secara otomatis:")

        # Loop dinamis untuk menjabarkan karakteristik klaster secara rinci
        for cluster_id, row in profil_kmeans.iterrows():
            cost = row['repair_cost']
            sev = row['severity_score']
            count = row['jumlah_sampel']

            # Menentukan label kategori berdasarkan data statistik aktual
            if sev >= 2.3 and cost >= 200:
                kategori = "🔴 **KATEGORI KRITIS & BIAYA TINGGI (High-Risk Financial Damage)**"
                insight = "Kelompok ini adalah ancaman terbesar bagi profitabilitas dan reputasi perusahaan. Cacat bersifat fatal dan perbaikannya sangat mahal."
                rekomendasi = "Hentikan jalur perakitan terkait sementara untuk kalibrasi ulang komponen presisi, evaluasi vendor pemasok bahan baku utama, dan perketat lolos QC awal."
            elif sev >= 2.1:
                kategori = "🟠 **KATEGORI KEPARAHAN TINGGI - BIAYA MENENGAH (Operational Issues)**"
                insight = "Cacat pada kelompok ini berdampak buruk bagi fungsi produk, namun biaya penanganannya masih masuk dalam batas toleransi standar."
                rekomendasi = "Tingkatkan intensitas perawatan mesin (*preventive maintenance*) dan lakukan audit berkala terhadap kepatuhan SOP operator di lapangan."
            elif cost >= 150:
                kategori = "🟡 **KATEGORI BIAYA TINGGI - KEPARAHAN RINGAN (Inefficient Repair Cost)**"
                insight = "Tingkat keparahan produk sebenarnya tidak fatal (kosmetik/minor), namun biaya penanganan atau penggantian partnya tidak efisien."
                rekomendasi = "Tinjau ulang kontrak dengan penyedia jasa perbaikan luar atau cari komponen substitusi lokal yang lebih murah tanpa menurunkan kualitas standar."
            else:
                kategori = "🟢 **KATEGORI AMAN (Low-Cost & Minor Defects)**"
                insight = "Kelompok cacat ringan dengan biaya perbaikan yang sangat murah. Karakteristik ini wajar terjadi dalam batas toleransi produksi massal."
                rekomendasi = "Cukup lakukan pemantauan berkala menggunakan grafik kendali kualitas statistik (SPC) dan lakukan inspeksi visual reguler di akhir lini."

            st.subheader(f"Klaster {cluster_id}: {kategori}")

            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            metrics_col1.metric("Rata-rata Biaya Perbaikan", f"${cost:.2f}")
            metrics_col2.metric("Skor Keparahan (1-3)", f"{sev:.2f}")
            metrics_col3.metric("Total Temuan Kasus", f"{int(count)} Produk")

            st.markdown(f"**Analisis Masalah:** {insight}")
            st.markdown(f"**Rekomendasi Strategis:** {rekomendasi}")
            st.markdown("---")

    with tab3:
        st.header("🛠️ Input Data Cacat Baru untuk Prediksi Klaster")
        st.markdown("Gunakan formulir ini untuk memprediksi masuk ke kelompok mana data temuan cacat baru di pabrik.")

        input_cost = st.number_input("Masukkan Biaya Perbaikan ($):", min_value=0.0, value=50.0, step=5.0)
        input_severity = st.selectbox("Pilih Tingkat Keparahan:", ['Minor', 'Moderate', 'Critical'])

        severity_mapped = {'Minor': 1, 'Moderate': 2, 'Critical': 3}[input_severity]

        if st.button("Prediksi Klaster"):
            new_data = np.array([[input_cost, severity_mapped]])
            new_data_scaled = scaler.transform(new_data)
            pred_cluster = kmeans_model.predict(new_data_scaled)[0]
            st.success(f"Hasil Analisis: Data cacat baru tersebut resmi dikelompokkan ke dalam **Klaster {pred_cluster}**")
else:
    st.error("Gagal memuat data! File 'defects_data.csv' tidak ditemukan. Harap letakkan file CSV tersebut di direktori yang sama dengan app.py.")
