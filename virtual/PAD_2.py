import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import scipy.cluster.hierarchy as shc

# Set page configuration
st.set_page_config(page_title="Analisis Clustering", layout="wide")

# --- Function Definitions ---
def scale_features(data, scaler):
    """Scales features using a pre-fitted scaler."""
    return scaler.transform(data)

def plot_dendrogram(X_scaled, method):
    """Creates and displays a dendrogram for Hierarchical Clustering."""
    st.write("### Dendrogram")
    st.info("Dendrogram membantu memvisualisasikan bagaimana cluster digabungkan secara hierarkis. Gunakan ini untuk membantu memilih jumlah cluster.")
    fig = plt.figure(figsize=(12, 8))
    dend = shc.dendrogram(shc.linkage(X_scaled, method=method))
    plt.title(f'Dendrogram (Metode: {method})')
    plt.xlabel('Data Points')
    plt.ylabel('Jarak Euclidean')
    st.pyplot(fig)

def display_cluster_profiling(df, selected_columns):
    """Displays automatic cluster profiling statistics."""
    st.write("### Profiling Cluster Otomatis")
    st.info("Tabel ini menunjukkan karakteristik rata-rata, median, dan standar deviasi untuk setiap fitur di dalam masing-masing cluster. Ini membantu Anda memahami 'identitas' setiap cluster.")
    
    profiling_metrics = ['mean', 'median', 'std']
    cluster_profile = df.groupby('Cluster')[selected_columns].agg(profiling_metrics)
    st.dataframe(cluster_profile.style.background_gradient(cmap='viridis'))


def display_feature_importance(X, labels, selected_columns):
    """Trains a RandomForest to determine feature importance for clustering."""
    st.write("### Tingkat Kepentingan Fitur (Feature Importance)")
    st.info("Fitur yang lebih tinggi di sini adalah yang paling berpengaruh dalam memisahkan data ke dalam cluster yang berbeda.")
    
    # Handle case with only one cluster or noise cluster
    if len(np.unique(labels)) < 2:
        st.warning("Feature importance tidak dapat dihitung karena hanya ada satu cluster yang terbentuk.")
        return

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, labels)
    
    importances = pd.DataFrame({
        'Fitur': selected_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Fitur', data=importances)
    plt.title('Feature Importance untuk Pembentukan Cluster')
    st.pyplot(fig)

# --- Streamlit UI ---

st.title("🚀 Analisis Clustering Lanjutan & Prediksi")

# Initialize session state to store model and scaler
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.scaler = None
    st.session_state.selected_columns = None
    st.session_state.predictor = None
    st.session_state.algorithm = None

# Sidebar for controls
st.sidebar.header("⚙️ Konfigurasi Analisis")
uploaded_file = st.sidebar.file_uploader("1. Unggah File CSV / Excel Anda", type=["csv", "xlsx"])

# Main app logic
if uploaded_file is None:
    st.info("Selamat datang! Silakan unggah file data Anda untuk memulai analisis clustering.")
else:
    # Read data
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    st.success("✅ File berhasil diunggah!")

    # --- MAIN TABS ---
    tab1, tab2 = st.tabs(["📊 Analisis & Clustering", "🔮 Prediksi pada Data Baru"])

    with tab1:
        st.header("Langkah 1: Eksplorasi & Filter Data")

        # --- Dynamic Data Filtering ---
        with st.expander("Filter Data Dinamis & Preview"):
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            filtered_df = df.copy()
            if categorical_cols:
                st.write("**Filter berdasarkan kolom kategorikal:**")
                for col in categorical_cols:
                    unique_vals = ['ALL'] + sorted(df[col].unique().tolist())
                    selected_val = st.selectbox(f"Filter berdasarkan '{col}':", unique_vals)
                    if selected_val != 'ALL':
                        filtered_df = filtered_df[filtered_df[col] == selected_val]
            else:
                st.write("Tidak ada kolom kategorikal untuk difilter.")
            
            st.write("### Preview Data (setelah difilter):")
            st.dataframe(filtered_df.head())

        st.header("Langkah 2: Konfigurasi Pra-pemrosesan & Algoritma")
        
        # --- Preprocessing Options ---
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Pra-pemrosesan")
            missing_value_option = st.radio(
                "Penanganan Missing Values:",
                ('Hapus Baris (dropna)', 'Isi dengan Mean', 'Isi dengan Median', 'Isi dengan Modus'),
                help="Pilih cara menangani data yang hilang."
            )
            scale_method = st.radio(
                "Metode Normalisasi:",
                ('MinMaxScaler', 'StandardScaler'),
                help="MinMaxScaler menskalakan data ke rentang [0,1]. StandardScaler menskalakan data ke mean=0, std=1."
            )
        
        with col2:
            st.write("#### Algoritma Clustering")
            algorithm = st.selectbox(
                "Pilih Algoritma:",
                ('K-Means', 'DBSCAN', 'Hierarchical Clustering', 'Gaussian Mixture (GMM)')
            )

        # --- Feature Selection ---
        numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
        st.write("#### Pemilihan Fitur")
        selected_columns = st.multiselect(
            "Pilih kolom numerik untuk clustering:",
            numeric_cols,
            default=numeric_cols[:2] if len(numeric_cols) > 1 else numeric_cols
        )

        if len(selected_columns) < 2:
            st.warning("Pilih minimal 2 kolom numerik untuk melanjutkan.")
            st.stop()

        # --- Data Preparation ---
        X = filtered_df[selected_columns].copy()

        # Handle missing values
        if X.isnull().sum().sum() > 0:
            if missing_value_option == 'Hapus Baris (dropna)':
                original_index = X.index
                X.dropna(inplace=True)
                df_processed = filtered_df.loc[X.index].copy()
            else:
                imputer_strategy = {
                    'Isi dengan Mean': 'mean',
                    'Isi dengan Median': 'median',
                    'Isi dengan Modus': 'most_frequent'
                }[missing_value_option]
                imputer = SimpleImputer(strategy=imputer_strategy)
                X[:] = imputer.fit_transform(X)
                df_processed = filtered_df.copy()
        else:
            df_processed = filtered_df.copy()

        # Scaling
        scaler = MinMaxScaler() if scale_method == 'MinMaxScaler' else StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        st.header("Langkah 3: Jalankan Clustering & Analisis Hasil")

        if st.button(f"🚀 Jalankan {algorithm} Clustering", type="primary"):
            st.session_state.scaler = scaler
            st.session_state.selected_columns = selected_columns
            st.session_state.algorithm = algorithm
            
            labels = []
            model = None

            # --- ALGORITHM-SPECIFIC UI & EXECUTION ---
            if algorithm == 'K-Means':
                with st.sidebar:
                    st.header("Parameter K-Means")
                    n_clusters = st.slider("Jumlah Cluster (K)", 2, 15, 3)
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = model.fit_predict(X_scaled)
                st.session_state.model = model
            
            elif algorithm == 'DBSCAN':
                with st.sidebar:
                    st.header("Parameter DBSCAN")
                    eps = st.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1, help="Jarak maksimum antara dua sampel untuk dianggap sebagai tetangga.")
                    min_samples = st.slider("Min Samples", 1, 20, 5, help="Jumlah sampel minimum dalam suatu lingkungan agar dianggap sebagai titik inti.")
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X_scaled)
                st.session_state.model = model

            elif algorithm == 'Hierarchical Clustering':
                with st.sidebar:
                    st.header("Parameter Hierarchical")
                    n_clusters = st.slider("Jumlah Cluster", 2, 15, 3)
                    linkage_method = st.selectbox("Metode Linkage", ['ward', 'complete', 'average', 'single'])
                plot_dendrogram(X_scaled, linkage_method)
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
                labels = model.fit_predict(X_scaled)
                st.session_state.model = model

            elif algorithm == 'Gaussian Mixture (GMM)':
                with st.sidebar:
                    st.header("Parameter GMM")
                    n_components = st.slider("Jumlah Komponen (Cluster)", 2, 15, 3)
                model = GaussianMixture(n_components=n_components, random_state=42)
                labels = model.fit_predict(X_scaled)
                st.session_state.model = model

            # --- Post-Clustering Analysis ---
            df_processed['Cluster'] = labels
            
            # Create a predictive model for all algorithms
            # For K-Means and GMM, it's the model itself. For others, we train a KNN.
            if algorithm in ['DBSCAN', 'Hierarchical Clustering']:
                 st.session_state.predictor = KNeighborsClassifier(n_neighbors=5).fit(X_scaled, labels)
            else:
                 st.session_state.predictor = model

            st.write("### Hasil Clustering")
            st.dataframe(df_processed.head())
            
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            st.write(f"**Jumlah cluster yang ditemukan:** {n_clusters_found}")
            if -1 in labels:
                 noise_points = np.sum(labels == -1)
                 st.write(f"**Jumlah titik noise (outlier):** {noise_points}")

            # --- Visualizations & Interpretations ---
            col_viz1, col_viz2 = st.columns(2)
            with col_viz1:
                st.write("#### Visualisasi Cluster (PCA)")
                try:
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    df_pca = pd.DataFrame(X_pca, columns=['PCA 1', 'PCA 2'])
                    df_pca['Cluster'] = labels
                    
                    fig = plt.figure(figsize=(8, 6))
                    sns.scatterplot(data=df_pca, x='PCA 1', y='PCA 2', hue='Cluster', palette='viridis', style='Cluster', s=50)
                    plt.title('Visualisasi 2D Hasil Clustering')
                    plt.legend(title='Cluster')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Gagal membuat visualisasi PCA: {e}")
            
            with col_viz2:
                st.write("#### Metrik Evaluasi")
                if n_clusters_found > 1:
                    sil_score = silhouette_score(X_scaled, labels)
                    db_score = davies_bouldin_score(X_scaled, labels)
                    st.metric("Silhouette Score", f"{sil_score:.3f}", help="Mendekati 1 lebih baik. Mengukur seberapa mirip objek dengan clusternya sendiri dibandingkan cluster lain.")
                    st.metric("Davies-Bouldin Score", f"{db_score:.3f}", help="Mendekati 0 lebih baik. Mengukur rata-rata kemiripan antar cluster.")
                else:
                    st.warning("Metrik evaluasi tidak dapat dihitung karena kurang dari 2 cluster ditemukan.")
            
            st.markdown("---")
            display_cluster_profiling(df_processed, selected_columns)
            
            st.markdown("---")
            display_feature_importance(X_scaled, labels, selected_columns)

    with tab2:
        st.header("Prediksi Cluster untuk Data Baru")
        if st.session_state.predictor is None:
            st.warning("⚠️ Anda harus menjalankan proses clustering di tab 'Analisis & Clustering' terlebih dahulu sebelum dapat melakukan prediksi.")
        else:
            st.info(f"Model **{st.session_state.algorithm}** dan scaler siap digunakan. Unggah data baru untuk prediksi.")
            
            new_uploaded_file = st.file_uploader("Unggah File Data Baru", type=["csv", "xlsx"], key="new_file")
            
            if new_uploaded_file is not None:
                try:
                    if new_uploaded_file.name.endswith('.csv'):
                        new_df = pd.read_csv(new_uploaded_file)
                    else:
                        new_df = pd.read_excel(new_uploaded_file)
                    
                    st.write("### Preview Data Baru")
                    st.dataframe(new_df.head())
                    
                    # Preprocess new data
                    new_X = new_df[st.session_state.selected_columns].copy()
                    
                    # Ensure no missing values in new data
                    if new_X.isnull().sum().sum() > 0:
                        st.warning("Data baru mengandung missing values. Mengisi dengan nilai mean dari data training.")
                        imputer = SimpleImputer(strategy='mean')
                        # Fit on old data, transform new data
                        imputer.fit(X) 
                        new_X[:] = imputer.transform(new_X)

                    new_X_scaled = st.session_state.scaler.transform(new_X)
                    
                    # Predict
                    predictions = st.session_state.predictor.predict(new_X_scaled)
                    new_df['Predicted_Cluster'] = predictions
                    
                    st.write("### Hasil Prediksi")
                    st.dataframe(new_df)
                    
                except KeyError:
                    st.error(f"Error: Pastikan file data baru Anda memiliki kolom yang sama dengan data training: {st.session_state.selected_columns}")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat prediksi: {e}")
