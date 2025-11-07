# Code dari Zefanya magang TR Data dan Potensi PDRD Periode II

import io
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import random
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, silhouette_score, davies_bouldin_score, ConfusionMatrixDisplay, classification_report

import warnings
warnings.filterwarnings("ignore")

def scale_features(X, method='MinMaxScaler'):
    if method == 'MinMaxScaler':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    return scaler.fit_transform(X)

st.title('K-Means Clustering')
st.subheader('Data Clustering')
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel file", type=["xlsx", "csv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ File berhasil diunggah!")
    st.write("### Preview Dataset:")
    st.dataframe(df.head())
    
    if df.isnull().sum().sum() > 0:
        df.dropna(inplace=True)
    else:
        st.info("Dataset tidak mengandung missing values.")

    buffer = io.StringIO()
    df.info(buf=buffer) 
    info_str = buffer.getvalue()
    
    st.write("### Informasi Dataset Setelah Missing Value Dihapus:")
    st.text(info_str)
    
    st.write("### Statistik Deskriptif:")
    st.write(df.describe())
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Dataset harus memiliki minimal 2 kolom numerik untuk clustering.")
        
    else:
        st.write("### Pilih Kolom untuk Clustering:")
        selected_columns = st.multiselect("Kolom Numerik", numeric_cols, default=numeric_cols[:2])

        if len(selected_columns) >= 2:
            X = df[selected_columns].dropna()
            
            scale_method = st.sidebar.radio("Pilih Metode Normalisasi", ["MinMaxScaler", "StandardScaler"])

            if scale_method == "MinMaxScaler":
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            
            X_scaled = scale_features(X, method=scale_method)
            st.write("### Data Setelah Normalisasi:")
            st.dataframe(pd.DataFrame(X_scaled, columns=selected_columns).sample(n=5)) 
            st.write('### Elbow Method untuk Jumlah Cluster')

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
            for i, value in enumerate(inertia):
                plt.text(K[i], value + 0.1, f'{value:.2f}', ha='center')
            plt.legend(['Inertia'])
            plt.tight_layout()
            st.pyplot(fig_elbow)
            
            cluster_range = st.sidebar.slider("Pilih Jumlah Cluster (K)", min_value=2, max_value=10, value=3)

            kmeans = KMeans(n_clusters=cluster_range, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)

            df_filtered = df.loc[X.index].copy()
            df_filtered['Cluster'] = cluster_labels

            silhouette_scores = []
            Davies_Bouldin_Index = []

            for k in range(2, 11):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                silhouette_scores.append(silhouette_score(X_scaled, labels))
                Davies_Bouldin_Index.append(davies_bouldin_score(X_scaled, labels))

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(range(2, 11), silhouette_scores, 'bx-')
            ax.axvline(cluster_range, color='red', linestyle='--', label=f'K yang dipilih = {cluster_range}')

            for i, value in enumerate(silhouette_scores):
                ax.text(range(2, 11)[i], value + 0.01, f'{value:.2f}', ha='center')

            ax.set_title("Silhouette Score untuk Tiap K")
            ax.set_xlabel("Jumlah Cluster")
            ax.set_ylabel("Silhouette Score")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(range(2, 11), Davies_Bouldin_Index, 'bx-')
            ax.axvline(cluster_range, color='red', linestyle='--', label=f'K yang dipilih = {cluster_range}')

            for i, value in enumerate(Davies_Bouldin_Index):
                ax.text(range(2, 11)[i], value + 0.01, f'{value:.2f}', ha='center')

            ax.set_title("Davies Bouldin Index untuk Tiap K")
            ax.set_xlabel("Jumlah Cluster")
            ax.set_ylabel("DBI Score")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

            st.write("### Data dengan Label Cluster:")
            st.dataframe(df_filtered.head())

            try:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)

                fig_pca = plt.figure(figsize=(8, 6))
                sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='Set1', style=cluster_labels)

                plt.title("Visualisasi Clustering (PCA 2D)")
                plt.xlabel("PCA 1")
                plt.ylabel("PCA 2")
                plt.legend(title='Cluster')
                plt.tight_layout()
                st.pyplot(fig_pca)

            except Exception as e:
                st.error(f"❌ Gagal membuat visualisasi PCA: {e}")

            if len(set(cluster_labels)) > 1:
                sil_score = silhouette_score(X_scaled, cluster_labels)
                db_score = davies_bouldin_score(X_scaled, cluster_labels)
                st.write(f"**Silhouette Score:** {sil_score:.3f}")
                st.write(f"**Davies-Bouldin Score:** {db_score:.3f}")
            else:
                st.warning("Silhouette dan DB Score tidak bisa dihitung karena hanya ada 1 cluster.")

        else:
            st.info("Pilih minimal 2 kolom numerik untuk visualisasi clustering.")
        
        st.write("### Simpan Data dengan Label Cluster:")
        
        if 'df_filtered' in locals() and st.button("Simpan Data"):
            csv_data = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button("Unduh DataClustered.csv", csv_data, "DataClustered.csv", "text/csv")
            st.success("Data berhasil disimpan sebagai 'DataClustered.csv'")
            st.write("Anda dapat mengunduh file ini untuk analisis lebih lanjut.")
        else:
            st.warning("Silakan pilih kolom numerik untuk clustering dan tentukan jumlah cluster sebelum menyimpan data.")

    st.write("### Terapkan K-Nearest Neighbors (KNN)")
    st.caption("KNN dilakukan terhadap label hasil clustering (bukan label asli), sehingga akurasi hanya sebagai simulasi prediktif.")

    apply_knn = st.radio("Apakah Anda ingin menerapkan K-Nearest Neighbors?", ("Tidak", "Ya"))

    if apply_knn == "Ya":
        st.write("### Pilih Nilai K untuk KNN:")
        knn_k = st.sidebar.slider("Jumlah Tetangga (K) untuk KNN", min_value=1, max_value=15, value=5)
        if 'df_filtered' not in locals() or 'Cluster' not in df_filtered.columns:
            st.warning("⚠️ Anda harus menjalankan proses clustering terlebih dahulu sebelum menerapkan KNN.")
            
        else:
            X_knn = X_scaled 
            y_knn = df_filtered['Cluster'] 

            X_train, X_test, y_train, y_test = train_test_split(X_knn, y_knn, test_size=0.2, random_state=42)

            knn = KNeighborsClassifier(n_neighbors=knn_k)
            knn.fit(X_train, y_train)

            y_pred = knn.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred)

            st.write(f"**Akurasi KNN:** {accuracy:.3f}")
            st.write("**Laporan Klasifikasi KNN:**")
            st.text(classification_rep)

            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(8, 6))

            unique_labels = np.unique(np.concatenate((y_test, y_pred)))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
            plt.title('Confusion Matrix KNN')
            plt.xlabel('Prediksi')
            plt.ylabel('Aktual')
            st.pyplot(fig_cm)
            
        st.write("### Prediksi Cluster untuk Input Manual")

        if 'df_filtered' in locals():
            # Pastikan kolom 'namapemda' dan 'tahun' ada
            if 'namapemda' in df_filtered.columns and 'tahun' in df_filtered.columns:

                # Pilih tahun
                selected_year = st.selectbox(
                    "Pilih Tahun:",
                    sorted(df_filtered['tahun'].unique())
                )

                # Filter data sesuai tahun
                df_year_filtered = df_filtered[df_filtered['tahun'] == selected_year]

                # Pilih Pemda
                selected_namapemda = st.selectbox(
                    "Pilih Pemda:",
                    sorted(df_year_filtered['namapemda'].unique())
                )

                if st.button("Cari Data Terdekat"):
                    # Ambil baris data berdasarkan namapemda dan tahun
                    selected_row = df_year_filtered[df_year_filtered['namapemda'] == selected_namapemda].iloc[0]

                    # Ambil hanya kolom fitur yang dipakai clustering
                    feature_row = selected_row[selected_columns].to_frame().T

                    # Scale data input
                    scaled_input = scale_features(feature_row, method=scale_method)

                    # Cari tetangga terdekat
                    distances, indices = knn.kneighbors(scaled_input, n_neighbors=knn_k)

                    # Kolom tambahan untuk ditampilkan
                    additional_cols = ['provinsi', 'tahun', 'namapemda']
                    available_additional = [col for col in additional_cols if col in df_filtered.columns]
                    display_cols = available_additional + selected_columns

                    # Data input yang dipilih
                    input_display = pd.DataFrame([selected_row[display_cols]])

                    # Tetangga terdekat (dari data tahun yang sama)
                    nearest_neighbors_global = df_filtered.iloc[indices[0]].copy()
                    jarak_global = distances[0]

                    # Filter sesuai tahun
                    mask_tahun = nearest_neighbors_global['tahun'] == selected_year
                    nearest_neighbors = nearest_neighbors_global[mask_tahun].copy()

                    # Ambil jarak yang sesuai
                    nearest_neighbors['Jarak'] = np.array(jarak_global)[mask_tahun]

                    nearest_neighbors = nearest_neighbors[display_cols + ['Jarak']]

                    # Tampilkan hasil
                    st.write(f"### 📌 Data {selected_namapemda} Tahun {selected_year} (Input)")
                    st.dataframe(input_display)

                    st.write(f"### 🔍 Data Terdekat untuk {selected_namapemda} di Tahun {selected_year}")
                    st.dataframe(nearest_neighbors)

            else:
                st.error("Kolom 'namapemda' atau 'tahun' tidak ditemukan di data.")



        