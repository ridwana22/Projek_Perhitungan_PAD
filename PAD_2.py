# ==========================================================================================
# Aplikasi Analisis Clustering dan Klasifikasi Menggunakan Streamlit
# ==========================================================================================
# Kode ini membangun aplikasi interaktif berbasis Streamlit untuk melakukan:
# 1. K-Means Clustering — Mengelompokkan data berdasarkan kesamaan fitur numerik. 
#     Disertai visualisasi PCA 2D, perhitungan metrik evaluasi (Silhouette & Davies-Bouldin Score),
#     serta opsi ekspor hasil clustering ke file CSV.
#
# 2. Model Klasifikasi — Menggunakan dua algoritma:
#     • K-Nearest Neighbors (KNN): dapat memilih algoritma pencarian tetangga ('auto', 'kd_tree', 'ball_tree', 'brute'),
#       serta mendukung GridSearchCV untuk tuning hyperparameter.
#     • Random Forest Classifier: dapat menyesuaikan jumlah pohon (n_estimators) dan kedalaman maksimum (max_depth).
#
# 3. Pencarian Data Terdekat — Memungkinkan pengguna mencari data serupa berdasarkan input manual 
#     (misalnya berdasarkan nama pemda dan tahun) dengan dua pendekatan:
#       • KNN: mencari tetangga terdekat berdasarkan jarak fitur. (LOGIKA DIPERBAIKI)
#       • Random Forest: menampilkan data lain dalam cluster yang sama.
#
# 4. Fitur Tambahan:
#     - Deteksi dan penghapusan missing values.
#     - Pemilihan metode normalisasi (MinMaxScaler / StandardScaler).
#     - Visualisasi Elbow Method untuk menentukan jumlah cluster optimal.
#     - Tabel hasil, metrik akurasi, confusion matrix, dan laporan klasifikasi otomatis.
#
# Tujuan utama:
# Membangun pipeline lengkap dari preprocessing → clustering → klasifikasi → analisis hasil 
# secara interaktif dalam satu aplikasi yang mudah digunakan.
# ==========================================================================================

# Import Library yang digunakan
import io
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report,
    silhouette_score, davies_bouldin_score
)
import warnings
warnings.filterwarnings("ignore")

# Fungsi scaling untuk fitur
def scale_features(X, method='MinMaxScaler'):
    if method == 'MinMaxScaler':
        scaler = MinMaxScaler() # Untuk mengskala fitur antara 0 dan 1
    else:
        scaler = StandardScaler() # Untuk mengskala fitur dengan mean=0 dan std=1
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

st.set_page_config(layout="wide") # Mengatur layout agar lebih luas
st.title('K-Means Clustering & KNN/Random Forest Classification')
st.subheader('🔍 Analisis Clustering dan Model Prediktif')

uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel file", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Baca file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file) # untuk Baca file CSV
    else:
        df = pd.read_excel(uploaded_file) # untuk Baca file Excel

    st.success("✅ File berhasil diunggah!")
    st.write("### Preview Dataset:")
    st.dataframe(df.head())

    # Cek Missing Values
    if df.isnull().sum().sum() > 0: # Cek apakah ada missing values 
        st.warning("⚠️ Dataset mengandung missing values. Baris dengan missing values akan dihapus.")
        df.dropna(inplace=True) # Hapus baris dengan missing values
    else:
        st.info("Dataset tidak mengandung missing values.")

    # Info Dataset
    col1, col2 = st.columns(2)
    
    with col1:
        buffer = io.StringIO() # Menyimpan output info ke buffer
        df.info(buf=buffer) # Dapatkan info dataset
        st.write("### Informasi Dataset:")
        st.text(buffer.getvalue())

    with col2:
        st.write("### Statistik Deskriptif:")
        st.write(df.describe()) # Statistik deskriptif dari dataset

    # Untuk clustering, pilih kolom numerik
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist() # Kolom numerik saja
    if len(numeric_cols) < 2:
        st.warning("Dataset harus memiliki minimal 2 kolom numerik untuk clustering.")
    else:
        selected_columns = st.multiselect("Pilih Kolom Numerik untuk Clustering:", numeric_cols, default=numeric_cols[:2])

        # Pastikan ada minimal 2 kolom yang dipilih
        if len(selected_columns) >= 2:
            X = df[selected_columns].dropna() # Hapus baris dengan missing values pada kolom terpilih

            # Normalisasi
            scale_method = st.sidebar.radio("Pilih Metode Normalisasi", ["MinMaxScaler", "StandardScaler"])
            # gunakan fungsi untuk mendapatkan X_scaled dan scaler yang sudah di-fit
            X_scaled, scaler = scale_features(X, method=scale_method)

            st.write("### Data Setelah Normalisasi (Sampel):")
            n_sample = min(5, X_scaled.shape[0]) # Tampilkan maksimal 5 sampel
            st.dataframe(pd.DataFrame(X_scaled, columns=selected_columns).sample(n=n_sample))
            
            # Kolom untuk Elbow dan Clustering
            col_elbow, col_clustering = st.columns(2)

            with col_elbow:
                st.write("#### 📉 Elbow Method (Menentukan K Optimal)")
                # Elbow method dipakai untuk membantu menentukan jumlah cluster (k) yang baik pada K‑Means dengan melihat trade‑off antara jumlah cluster dan inertia
                inertia = []
                K = range(1, 11)
                for k in K:
                    # Menggunakan n_init=10 untuk menghilangkan warning.
                    kmeans_ = KMeans(n_clusters=k, random_state=42, n_init=10) 
                    kmeans_.fit(X_scaled)
                    inertia.append(kmeans_.inertia_)

                fig_elbow = plt.figure(figsize=(6, 4))
                plt.plot(K, inertia, 'bo-')
                plt.xlabel('Jumlah Cluster (k)')
                plt.ylabel('Inertia')
                plt.title('Elbow Method')
                plt.grid(True)
                st.pyplot(fig_elbow)
            
            with col_clustering:
                st.write("#### 📊 Hasil K-Means Clustering")
                # Untuk menentukan jumlah cluster (k) secara manual atau default
                check_k = st.sidebar.checkbox("Atur jumlah Cluster (K) manual", value=False)
                if check_k:
                    cluster_range = int(st.sidebar.number_input("Masukkan jumlah cluster (K)", 2, 10, 3, 1))
                else:
                    cluster_range = 3

                # K-means 
                # Untuk melakukan proses pembentukan kelompok (clustering) 
                # pada data yang telah dinormalisasi, lalu menambahkan hasil 
                # label cluster ke dataframe utama 
                try:
                    kmeans = KMeans(n_clusters=int(cluster_range), random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(X_scaled)

                    df_filtered = df.loc[X.index].copy()
                    df_filtered['Cluster'] = cluster_labels

                    # Visualisasi PCA
                    # Menampilkan hasil clustering dalam bentuk visual 2D
                    # menggunakan PCA, agar kita bisa melihat pola, sebaran, 
                    # dan pemisahan antar cluster secara jelas 
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    fig_pca = plt.figure(figsize=(6, 4))
                    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='Set1', style=cluster_labels)
                    plt.title("Visualisasi Clustering (PCA 2D)")
                    plt.xlabel("PCA 1")
                    plt.ylabel("PCA 2")
                    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    st.pyplot(fig_pca)

                    # Menilai seberapa baik hasil clustering
                    sil_score = silhouette_score(X_scaled, cluster_labels)
                    db_score = davies_bouldin_score(X_scaled, cluster_labels)
                    st.markdown(f"**Metrik Evaluasi (K={cluster_range}):**")
                    st.markdown(f"**• Silhouette Score:** `{sil_score:.3f}` (Semakin tinggi, semakin baik)")
                    st.markdown(f"**• Davies-Bouldin Score:** `{db_score:.3f}` (Semakin rendah, semakin baik)")
                
                except ValueError as e:
                    st.error(f"Error saat Clustering/PCA: {e}. Pastikan jumlah cluster (K) tidak melebihi jumlah data unik.")
                    # PERBAIKAN: Mengganti 'return' dengan 'st.stop()'
                    st.stop() 

            st.write("### Data Hasil Clustering (Preview):")
            st.dataframe(df_filtered.head())
            
            # Simpan data
            # Membuat fitur ekspor hasil clustering ke file CSV dan
            # menyediakan tombol unduhan langsung di aplikasi Streamlit. 
            st.markdown("---")
            csv_data = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                "💾 Unduh Data Clustered (CSV)", 
                csv_data, 
                "DataClustered.csv", 
                "text/csv",
                key='download_clustered_data'
            )
            st.markdown("---")


            # -----------------------------
            # Bagian KNN dan Random Forest
            # -----------------------------
            st.write("## 🔍 Model Prediktif Berdasarkan Hasil Clustering")
            model_choice = st.radio("Pilih Model yang Akan Diterapkan:", ["KNN", "Random Forest"])

            # Menyiapkan data sebelum pelatihan model, memisahkan fitur dan label (Cluster)
            X_clf = X_scaled
            y_clf = df_filtered['Cluster']
            X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

            # Inisialisasi model di luar blok if/else agar bisa diakses di bagian pencarian
            knn = None
            rf = None
            knn_k = 5 # Default
            knn_algo = 'auto' # Default
            
            clf_col1, clf_col2 = st.columns(2)

            if model_choice == "KNN":
                with clf_col1:
                    st.subheader("🔹 K-Nearest Neighbors (KNN)")
                    knn_k = st.sidebar.slider("Jumlah Tetangga (K)", 1, 20, 5, key='knn_k_slider')
                    knn_k = int(knn_k)
                    knn_algo = st.sidebar.selectbox("Pilih Algoritma KNN", ['auto', 'kd_tree', 'ball_tree', 'brute'], key='knn_algo_select')

                    knn = KNeighborsClassifier(n_neighbors=knn_k, algorithm=knn_algo)
                    knn.fit(X_train, y_train)
                    y_pred = knn.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    st.write(f"**Akurasi KNN ({knn_algo}):** `{accuracy:.3f}`")
                    st.text(classification_report(y_test, y_pred))

                with clf_col2:
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
                    plt.title('Confusion Matrix KNN')
                    st.pyplot(fig_cm)
                    
                    if st.checkbox("Lakukan Hyperparameter Tuning (GridSearchCV)", key='knn_tune'):
                        param_grid = {
                            'n_neighbors': range(1, 11),
                            'algorithm': ['auto', 'kd_tree', 'ball_tree', 'brute']
                        }
                        grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3)
                        grid_knn.fit(X_train, y_train)
                        st.write(f"**Best Params:** {grid_knn.best_params_}")
                        st.write(f"**Best Score (CV):** `{grid_knn.best_score_:.3f}`")

            else:
                with clf_col1:
                    st.subheader("🌲 Random Forest Classifier")
                    n_estimators = st.sidebar.slider("Jumlah Pohon (n_estimators)", 10, 200, 100, key='rf_n_estimators')
                    max_depth = st.sidebar.slider("Kedalaman Maksimum (max_depth)", 2, 20, 5, key='rf_max_depth')

                    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                    rf.fit(X_train, y_train)
                    y_pred = rf.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    st.write(f"**Akurasi Random Forest:** `{accuracy:.3f}`")
                    st.text(classification_report(y_test, y_pred))

                with clf_col2:
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False)
                    plt.title('Confusion Matrix Random Forest')
                    st.pyplot(fig_cm)

            # -----------------------------
            # Cari Data Terdekat
            # -----------------------------
            st.write("## 📍 Cari Data Terdekat Berdasarkan Input Manual")

            if (model_choice == "KNN" and knn is None) or (model_choice == "Random Forest" and rf is None):
                st.warning("Silakan jalankan model klasifikasi di atas terlebih dahulu.")
            elif "namapemda" in df.columns and "tahun" in df.columns:
                selected_year = st.selectbox("Pilih Tahun", sorted(df_filtered['tahun'].unique()))
                df_year_filtered = df_filtered[df_filtered['tahun'] == selected_year]

                if df_year_filtered.empty:
                    st.warning(f"Tidak ada data 'namapemda' untuk tahun {selected_year}.")
                else:
                    selected_namapemda = st.selectbox("Pilih Nama Pemda", sorted(df_year_filtered['namapemda'].unique()))

                    if st.button("Cari Data Terdekat"):
                        # Ambil baris input dari tahun yang dipilih
                        selected_row = df_year_filtered[df_year_filtered['namapemda'] == selected_namapemda].iloc[0]
                        input_index = selected_row.name

                        # Ambil fitur input dan scaling menggunakan scaler global
                        feature_row = selected_row[selected_columns].to_frame().T
                        scaled_input = scaler.transform(feature_row)
                        display_cols = ['namapemda', 'tahun'] + selected_columns

                        st.write("### 📌 Data Input yang Digunakan:")
                        input_display = selected_row[display_cols].to_frame().T
                        st.dataframe(input_display)

                        if model_choice == "KNN":
                            # === LOGIKA KNN BARU: FIT HANYA PADA DATA TAHUN YANG SAMA ===
                            
                            # 1. Ambil data FITUR dan LABEL hanya untuk tahun yang dipilih.
                            df_year_features = df_filtered[df_filtered['tahun'] == selected_year]
                            
                            # 2. Skala fitur tahun yang sama menggunakan scaler global
                            X_year_filtered_scaled = scaler.transform(df_year_features[selected_columns])

                            # 3. Inisialisasi/fit KNN HANYA pada data fitur tahun yang sama.
                            knn_full_year = KNeighborsClassifier(n_neighbors=knn_k, algorithm=knn_algo)
                            knn_full_year.fit(X_year_filtered_scaled, df_year_features['Cluster'])
                            
                            # 4. Cari tetangga terdekat dari input di dalam subset tahun yang sama.
                            n_nbrs = min(int(knn_k) + 1, len(X_year_filtered_scaled))
                            distances, indices = knn_full_year.kneighbors(scaled_input, n_neighbors=n_nbrs)

                            # 5. Ambil indeks data asli yang sesuai dari df_year_features
                            original_indices = df_year_features.iloc[indices[0]].index.tolist()

                            nearest_neighbors = df_filtered.loc[original_indices].copy()
                            nearest_neighbors['Jarak'] = distances[0]
                            
                            # Hapus baris input sendiri
                            nearest_neighbors = nearest_neighbors[nearest_neighbors.index != input_index].head(knn_k)

                            cols_to_show = [c for c in display_cols if c in nearest_neighbors.columns] + ['Cluster', 'Jarak']
                            cols_to_show = list(dict.fromkeys(cols_to_show))
                            nearest_neighbors = nearest_neighbors[cols_to_show]

                            st.write(f"### 🔹 Hasil Pencarian Tetangga Terdekat (KNN, Tahun {selected_year})")
                            if nearest_neighbors.empty:
                                st.warning(f"Tidak ada data terdekat lain yang ditemukan di tahun {selected_year} selain data input itu sendiri (K={knn_k} mungkin terlalu kecil atau datanya unik).")
                            else:
                                st.dataframe(nearest_neighbors)

                        else:
                            # Random Forest: tampilkan data lain dalam cluster sama dari tahun yang sama
                            predicted_cluster = rf.predict(scaled_input)[0]
                            st.info(f"Data input diprediksi berada di Cluster: **{predicted_cluster}**")
                            
                            nearest_neighbors_same_year = df_filtered[
                                (df_filtered['Cluster'] == predicted_cluster) &
                                (df_filtered['tahun'] == selected_year)
                            ].copy()
                            nearest_neighbors_same_year = nearest_neighbors_same_year.drop(input_index, errors='ignore')

                            n_show = min(5, len(nearest_neighbors_same_year))
                            
                            if n_show > 0:
                                nearest_neighbors_same_year = nearest_neighbors_same_year.sample(n_show)
                            
                            cols_to_show = [c for c in display_cols if c in nearest_neighbors_same_year.columns] + ['Cluster']
                            cols_to_show = list(dict.fromkeys(cols_to_show))
                            nearest_neighbors = nearest_neighbors_same_year[cols_to_show]

                            st.write(f"### 🌲 Hasil Pencarian Berdasarkan Cluster Sama (Random Forest, Tahun {selected_year})")
                            if nearest_neighbors.empty:
                                st.warning(f"Data ini adalah satu-satunya entitas di Cluster {predicted_cluster} pada tahun {selected_year}.")
                            else:
                                st.dataframe(nearest_neighbors)

            else:
                st.warning("Dataset tidak memiliki kolom 'namapemda' dan/atau 'tahun'.")
        else:
            st.warning("Silakan pilih minimal 2 kolom numerik untuk memulai analisis.")
