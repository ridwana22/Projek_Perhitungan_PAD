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
#       • KNN: mencari tetangga terdekat berdasarkan jarak fitur.
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

# Fungsi scaling untuk fitu
def scale_features(X, method='MinMaxScaler'):
    if method == 'MinMaxScaler':
        scaler = MinMaxScaler() # Untuk mengskala fitur antara 0 dan 1
    else:
        scaler = StandardScaler() # Untuk mengskala fitur dengan mean=0 dan std=1
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

st.title('K-Means Clustering & KNN/Random Forest Classification')
st.subheader('🔍 Analisis Clustering dan Model Prediktif')

uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel file", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Baca file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file) # Baca file CSV
    else:
        df = pd.read_excel(uploaded_file) # Baca file Excel

    st.success("✅ File berhasil diunggah!")
    st.write("### Preview Dataset:")
    st.dataframe(df.head())

    # Cek Missing Values
    if df.isnull().sum().sum() > 0:
        st.warning("⚠️ Dataset mengandung missing values. Baris dengan missing values akan dihapus.")
        df.dropna(inplace=True)
    else:
        st.info("Dataset tidak mengandung missing values.")

    # Info Dataset
    buffer = io.StringIO() # Menyimpan output info ke buffer
    df.info(buf=buffer) # Dapatkan info dataset
    st.write("### Informasi Dataset:")
    st.text(buffer.getvalue())

    st.write("### Statistik Deskriptif:")
    st.write(df.describe()) # Statistik deskriptif dari dataset

    # Untuk clustering, pilih kolom numerik
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
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

            st.write("### Data Setelah Normalisasi:")
            n_sample = min(5, X_scaled.shape[0]) # Tampilkan maksimal 5 sampel
            st.dataframe(pd.DataFrame(X_scaled, columns=selected_columns).sample(n=n_sample))

            # Elbow method dipakai untuk membantu menentukan jumlah cluster (k) yang baik pada K‑Means dengan melihat trade‑off antara jumlah cluster dan inertia
            inertia = []
            K = range(1, 11)
            for k in K:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertia.append(kmeans.inertia_)

            fig_elbow = plt.figure(figsize=(8, 5))
            plt.plot(K, inertia, 'bo-')
            plt.xlabel('Jumlah Cluster (k)')
            plt.ylabel('Inertia')
            plt.title('Elbow Method')
            plt.grid(True)
            st.pyplot(fig_elbow)

            # Untuk menentukan jumlah cluster (k) secara manual atau default
            check_k = st.sidebar.checkbox("Atur jumlah Cluster (K) manual", value=False)
            if check_k:
                cluster_range = int(st.sidebar.number_input("Masukkan jumlah cluster (K)", 2, 10, 3, 1))
            else:
                cluster_range = 3

            # K-means 
            # Untuk melakukan proses pembentukan kelompok (clustering) 
            #pada data yang telah dinormalisasi, lalu menambahkan hasil 
            #label cluster ke dataframe utama 
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
            fig_pca = plt.figure(figsize=(8, 6))
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='Set1', style=cluster_labels)
            plt.title("Visualisasi Clustering (PCA 2D)")
            plt.xlabel("PCA 1")
            plt.ylabel("PCA 2")
            plt.legend(title='Cluster')
            st.pyplot(fig_pca)

            st.write("### Data dengan Label Cluster:")
            st.dataframe(df_filtered.head())


            # Menilai seberapa baik hasil clustering K-Means yang telah dilakukan, berdasarkan dua metrik utama:
            # Silhouette Score → semakin tinggi, semakin baik.
            # Davies–Bouldin Score → semakin rendah, semakin baik. 
            sil_score = silhouette_score(X_scaled, cluster_labels)
            db_score = davies_bouldin_score(X_scaled, cluster_labels)
            st.write(f"**Silhouette Score:** {sil_score:.3f}")
            st.write(f"**Davies-Bouldin Score:** {db_score:.3f}")

            # Simpan data
            # Membuat fitur ekspor hasil clustering ke file CSV dan
            # menyediakan tombol unduhan langsung di aplikasi Streamlit. 
            if st.button("💾 Simpan Data Clustered"):
                csv_data = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button("Unduh DataClustered.csv", csv_data, "DataClustered.csv", "text/csv")
                st.success("Data berhasil disimpan sebagai 'DataClustered.csv'")


            # -----------------------------
            # Bagian KNN dan Random Forest
            # -----------------------------
            st.write("## 🔍 Model Prediktif Berdasarkan Hasil Clustering")
            model_choice = st.radio("Pilih Model yang Akan Diterapkan:", ["KNN", "Random Forest"])

            # Menyiapkan data sebelum pelatihan model KNN classifier, 
            # dengan cara memisahkan fitur (X_scaled) dan label (Cluster), 
            # lalu membaginya menjadi data latih dan data uji. 
            X_knn = X_scaled
            y_knn = df_filtered['Cluster']
            X_train, X_test, y_train, y_test = train_test_split(X_knn, y_knn, test_size=0.2, random_state=42)

            # Inisialisasi model di luar blok if/else agar bisa diakses di bagian pencarian
            knn = None
            rf = None
            knn_k = 5 # Default
            knn_algo = 'auto' # Default

            # Untuk menguji seberapa bagus model KNN (K-Nearest Neighbors) 
            # dalam memprediksi hasil pengelompokan (cluster) dari data yang 
            # sudah diolah sebelumnya. Juga bisa melihat akurasi modelnya, 
            # melihat hasil prediksi dalam bentuk visual, dan mencoba mencari 
            # pengaturan terbaik otomatis (tuning) untuk KNN. """
            if model_choice == "KNN":
                st.subheader("🔹 K-Nearest Neighbors (KNN)")
                knn_k = st.sidebar.slider("Jumlah Tetangga (K)", 1, 15, 5)
                knn_k = int(knn_k)
                knn_algo = st.sidebar.selectbox("Pilih Algoritma KNN", ['auto', 'kd_tree', 'ball_tree', 'brute'])

                knn = KNeighborsClassifier(n_neighbors=knn_k, algorithm=knn_algo)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"**Akurasi KNN ({knn_algo}):** {accuracy:.3f}")
                st.text(classification_report(y_test, y_pred))

                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title('Confusion Matrix KNN')
                st.pyplot(fig_cm)

                if st.checkbox("Lakukan Hyperparameter Tuning (GridSearchCV)"):
                    param_grid = {
                        'n_neighbors': range(1, 11),
                        'algorithm': ['auto', 'kd_tree', 'ball_tree', 'brute']
                    }
                    grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3)
                    grid_knn.fit(X_train, y_train)
                    st.write(f"**Best Params:** {grid_knn.best_params_}")
                    st.write(f"**Best Score (CV):** {grid_knn.best_score_:.3f}")

            else:
                # Kode ini digunakan untuk melatih dan menilai model Random Forest 
                # Classifier, yaitu salah satu algoritma Machine Learning yang digunakan 
                # untuk klasifikasi (membedakan atau mengelompokkan data). Kalau Random Forest 
                # bekerja dengan menggabungkan banyak pohon keputusan (decision tree) agar 
                # hasilnya lebih stabil dan akurat. """
                st.subheader("🌲 Random Forest Classifier")
                n_estimators = st.sidebar.slider("Jumlah Pohon (n_estimators)", 10, 200, 100)
                max_depth = st.sidebar.slider("Kedalaman Maksimum (max_depth)", 2, 20, 5)

                rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"**Akurasi Random Forest:** {accuracy:.3f}")
                st.text(classification_report(y_test, y_pred))

                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
                plt.title('Confusion Matrix Random Forest')
                st.pyplot(fig_cm)

            # -----------------------------
            # Cari Data Terdekat (MODIFIKASI DIMULAI DI SINI)
            # -----------------------------
            
            # Untuk mencari data terdekat berdasarkan input manual dari user
            st.write("## 📍 Cari Data Terdekat Berdasarkan Input Manual")
            # Pastikan model sudah terinisialisasi
            if (model_choice == "KNN" and knn is None) or (model_choice == "Random Forest" and rf is None): # Cek model apa yang dipilih
                 st.warning("Silakan jalankan model klasifikasi di atas terlebih dahulu.")

            # Cek apakah kolom 'namapemda' dan 'tahun' ada di dataset
            elif "namapemda" in df.columns and "tahun" in df.columns:
                selected_year = st.selectbox("Pilih Tahun", sorted(df['tahun'].unique()))
                df_year_filtered = df_filtered[df_filtered['tahun'] == selected_year]
                
                # Cek apakah data filter untuk tahun tersebut ada
                if df_year_filtered.empty:
                    st.warning(f"Tidak ada data 'namapemda' untuk tahun {selected_year} dalam dataset hasil clustering.")
                else:
                    selected_namapemda = st.selectbox("Pilih Nama Pemda", sorted(df_year_filtered['namapemda'].unique()))

                    # Tombol untuk mencari data terdekat
                    if st.button("Cari Data Terdekat"):
                        selected_row = df_year_filtered[df_year_filtered['namapemda'] == selected_namapemda].iloc[0]
                        input_index = selected_row.name # Simpan index asli baris input

                        # Pastikan kolom yang dipilih (X) di-index ulang agar urutan fiturnya sama
                        feature_row = selected_row[selected_columns].to_frame().T
                        # gunakan scaler yang sudah di-fit pada X, jangan fit ulang pada satu baris
                        scaled_input = scaler.transform(feature_row)

                        display_cols = ['namapemda', 'tahun'] + selected_columns  # kolom yang akan ditampilkan

                        if model_choice == "KNN":
                            # --- PERBAIKAN KNN: Latih KNN pada SELURUH data ter-skala ---
                            knn_full = KNeighborsClassifier(n_neighbors=knn_k, algorithm=knn_algo)
                            knn_full.fit(X_scaled, df_filtered['Cluster']) 

                            # minta 1 neighbor ekstra supaya bisa menghapus baris input itu sendiri
                            n_nbrs = min(int(knn_k) + 1, len(X_scaled))
                            distances, indices = knn_full.kneighbors(scaled_input, n_neighbors=n_nbrs)
                            
                            # Indeks yang dihasilkan adalah indeks dari X_scaled, yang sesuai dengan df_filtered
                            nearest_neighbors_global = df_filtered.iloc[indices[0]].copy()
                            nearest_neighbors_global['Jarak'] = distances[0]
                            
                            # Hapus baris input itu sendiri dari hasil (jika ada), lalu ambil knn_k teratas
                            nearest_neighbors_global = nearest_neighbors_global[nearest_neighbors_global.index != input_index]
                            nearest_neighbors_global = nearest_neighbors_global.head(knn_k)

                            # Tentukan kolom yang akan ditampilkan
                            cols_to_show = [c for c in display_cols if c in nearest_neighbors_global.columns] + ['Cluster', 'Jarak']
                            nearest_neighbors = nearest_neighbors_global[cols_to_show]

                            st.write(f"### 🔹 Hasil Pencarian Tetangga Terdekat (KNN - {knn_algo})")
                            st.dataframe(nearest_neighbors)

                        else:
                            # --- PERBAIKAN Random Forest: Hapus random_state untuk sampel ---
                            predicted_cluster = rf.predict(scaled_input)[0]
                            nearest_neighbors_global = df_filtered[df_filtered['Cluster'] == predicted_cluster].copy()
                            
                            # Hapus baris input itu sendiri dari daftar
                            nearest_neighbors_global = nearest_neighbors_global.drop(input_index, errors='ignore')
                            
                            n_show = min(5, len(nearest_neighbors_global))
                            nearest_neighbors_global = nearest_neighbors_global.sample(n_show) 

                            cols_to_show = [c for c in display_cols if c in nearest_neighbors_global.columns] + ['Cluster']
                            # Dedup kolom dan tampilkan
                            cols_to_show = list(dict.fromkeys(cols_to_show))
                            nearest_neighbors = nearest_neighbors_global[cols_to_show]

                            st.write(f"### 🌲 Hasil Pencarian Berdasarkan Cluster Sama (Random Forest)")
                            st.dataframe(nearest_neighbors)
            else:
                st.warning("Dataset tidak memiliki kolom 'namapemda' dan/atau 'tahun' yang diperlukan untuk fitur ini.")