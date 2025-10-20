# Import library
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

# Fungsi scaling
def scale_features(X, method='MinMaxScaler'):
    if method == 'MinMaxScaler': 
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    return scaler.fit_transform(X)

st.title('K-Means Clustering & KNN/Random Forest Classification')
st.subheader('🔍 Analisis Clustering dan Model Prediktif')

uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel file", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Baca file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ File berhasil diunggah!")
    st.write("### Preview Dataset:")
    st.dataframe(df.head())

    if df.isnull().sum().sum() > 0:
        st.warning("⚠️ Dataset mengandung missing values. Baris dengan missing values akan dihapus.")
        df.dropna(inplace=True)
    else:
        st.info("Dataset tidak mengandung missing values.")

    # Info Dataset
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.write("### Informasi Dataset:")
    st.text(buffer.getvalue())

    st.write("### Statistik Deskriptif:")
    st.write(df.describe())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Dataset harus memiliki minimal 2 kolom numerik untuk clustering.")
    else:
        selected_columns = st.multiselect("Pilih Kolom Numerik untuk Clustering:", numeric_cols, default=numeric_cols[:2])

        if len(selected_columns) >= 2:
            X = df[selected_columns].dropna()

            # Normalisasi
            scale_method = st.sidebar.radio("Pilih Metode Normalisasi", ["MinMaxScaler", "StandardScaler"])
            X_scaled = scale_features(X, method=scale_method)
            st.write("### Data Setelah Normalisasi:")
            st.dataframe(pd.DataFrame(X_scaled, columns=selected_columns).sample(n=5))

            # Elbow Method
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

            # Jumlah cluster manual
            check_k = st.sidebar.checkbox("Atur jumlah Cluster (K) manual", value=False)
            if check_k:
                cluster_range = st.sidebar.number_input("Masukkan jumlah cluster (K)", 2, 10, 3, 1)
            else:
                cluster_range = 3

            # KMeans
            kmeans = KMeans(n_clusters=cluster_range, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)

            df_filtered = df.loc[X.index].copy()
            df_filtered['Cluster'] = cluster_labels

            # Visualisasi PCA
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

            sil_score = silhouette_score(X_scaled, cluster_labels)
            db_score = davies_bouldin_score(X_scaled, cluster_labels)
            st.write(f"**Silhouette Score:** {sil_score:.3f}")
            st.write(f"**Davies-Bouldin Score:** {db_score:.3f}")

            # Simpan data
            if st.button("💾 Simpan Data Clustered"):
                csv_data = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button("Unduh DataClustered.csv", csv_data, "DataClustered.csv", "text/csv")
                st.success("Data berhasil disimpan sebagai 'DataClustered.csv'")

            # -----------------------------
            # Bagian KNN dan Random Forest
            # -----------------------------
            st.write("## 🔍 Model Prediktif Berdasarkan Hasil Clustering")
            model_choice = st.radio("Pilih Model yang Akan Diterapkan:", ["KNN", "Random Forest"])

            X_knn = X_scaled
            y_knn = df_filtered['Cluster']
            X_train, X_test, y_train, y_test = train_test_split(X_knn, y_knn, test_size=0.2, random_state=42)

            if model_choice == "KNN":
                st.subheader("🔹 K-Nearest Neighbors (KNN)")
                knn_k = st.sidebar.slider("Jumlah Tetangga (K)", 1, 15, 5)
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
            # Cari Data Terdekat
            # -----------------------------
            st.write("## 📍 Cari Data Terdekat Berdasarkan Input Manual")
            if "namapemda" in df.columns and "tahun" in df.columns:
                selected_year = st.selectbox("Pilih Tahun", sorted(df['tahun'].unique()))
                df_year_filtered = df_filtered[df_filtered['tahun'] == selected_year]
                selected_namapemda = st.selectbox("Pilih Nama Pemda", sorted(df_year_filtered['namapemda'].unique()))

                if st.button("Cari Data Terdekat"):
                    selected_row = df_year_filtered[df_year_filtered['namapemda'] == selected_namapemda].iloc[0]
                    feature_row = selected_row[selected_columns].to_frame().T
                    scaled_input = scale_features(feature_row, method=scale_method)

                    display_cols = ['namapemda', 'tahun'] + selected_columns  # kolom yang akan ditampilkan

                    if model_choice == "KNN":
                        distances, indices = knn.kneighbors(scaled_input, n_neighbors=knn_k)
                        nearest_neighbors_global = df_filtered.iloc[indices[0]].copy()
                        nearest_neighbors_global['Jarak'] = distances[0]
                        # Pastikan kolom ada sebelum memilih
                        cols_to_show = [c for c in display_cols if c in nearest_neighbors_global.columns] + ['Jarak']
                        nearest_neighbors = nearest_neighbors_global[cols_to_show]

                        st.write(f"### 🔹 Hasil Pencarian Tetangga Terdekat (KNN - {knn_algo})")
                        st.dataframe(nearest_neighbors)

                    else:
                        predicted_cluster = rf.predict(scaled_input)[0]
                        nearest_neighbors_global = df_filtered[df_filtered['Cluster'] == predicted_cluster].copy()
                        n_show = min(5, len(nearest_neighbors_global))
                        nearest_neighbors_global = nearest_neighbors_global.sample(n_show, random_state=42)
                        cols_to_show = [c for c in display_cols if c in nearest_neighbors_global.columns] + selected_columns
                        # jika selected_columns sudah termasuk, dedup kolom
                        cols_to_show = list(dict.fromkeys(cols_to_show))
                        nearest_neighbors = nearest_neighbors_global[cols_to_show]

                        st.write(f"### 🌲 Hasil Pencarian Berdasarkan Cluster Sama (Random Forest)")
                        st.dataframe(nearest_neighbors)
