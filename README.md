# Aplikasi K-Means Clustering & Model Prediktif (KNN dan Random Forest)

Aplikasi berbasis **Streamlit** untuk melakukan **analisis clustering** menggunakan **K-Means** dan membangun **model prediktif** menggunakan **K-Nearest Neighbors (KNN)** serta **Random Forest**.  
Aplikasi ini memungkinkan pengguna untuk mengunggah dataset sendiri, melakukan pra-pemrosesan data, visualisasi hasil clustering, dan membuat model klasifikasi berbasis hasil clustering.

---

## 🚀 **Fitur Utama**

✅ Upload file **CSV atau Excel (.xlsx)**  
✅ Deteksi dan penanganan **missing values**  
✅ Normalisasi data menggunakan **MinMaxScaler** atau **StandardScaler**  
✅ Clustering otomatis dan manual dengan **K-Means**  
✅ **Elbow Method** untuk menentukan jumlah cluster optimal  
✅ Visualisasi hasil clustering dalam bentuk **PCA 2D Plot**  
✅ Perhitungan metrik kualitas cluster:
   - **Silhouette Score**
   - **Davies-Bouldin Score**  
✅ Model prediktif:
   - **KNN (K-Nearest Neighbors)** dengan pilihan algoritma: `auto`, `kd_tree`, `ball_tree`, `brute`
   - **Random Forest** dengan pengaturan jumlah pohon dan kedalaman maksimum  
✅ **Hyperparameter tuning** otomatis dengan **GridSearchCV**  
✅ Fitur **pencarian data terdekat** berdasarkan nama daerah (`namapemda`) dan tahun (`tahun`)  
✅ Download hasil clustering dalam format **CSV**

---

## 🧠 **Teknologi yang Digunakan**

- **Python 3.9+**
- **Streamlit**
- **scikit-learn**
- **pandas**
- **numpy**
- **seaborn**
- **matplotlib**

---

## ⚙️ **Langkah Instalasi**

1. Clone repositori ini:
   ```bash
   git clone https://github.com/ridwana22/Projek_Perhitungan_PAD.git
   cd PerhitunganPAD
````

2. Buat dan aktifkan virtual environment (disarankan):

   ```bash
   python -m venv venv
   venv\Scripts\activate      # Untuk Windows
   source venv/bin/activate   # Untuk Mac/Linux
   ```

3. Instal semua dependensi:

   ```bash
   pip install -r requirements.txt
   ```

4. Jalankan aplikasi Streamlit:

   ```bash
   streamlit run PAD_2.py
   ```

---

## 🧮 **Cara Menggunakan Aplikasi**

1. Jalankan aplikasi Streamlit seperti perintah di atas.
2. Unggah dataset berformat **CSV** atau **Excel (.xlsx)**.
3. Pilih kolom numerik untuk analisis clustering.
4. Pilih metode normalisasi dan jumlah cluster (opsional).
5. Lihat hasil visualisasi dan metrik evaluasi cluster.
6. Pilih model prediktif yang diinginkan (**KNN** atau **Random Forest**).
7. Jika menggunakan KNN, dapat dilakukan tuning parameter otomatis menggunakan **GridSearchCV**.
8. Gunakan fitur **Cari Data Terdekat** untuk melihat data serupa berdasarkan input.
9. Unduh hasil data yang telah dikelompokkan (clustered data) dalam format CSV.

---


