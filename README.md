# 🧠 K-Means Clustering & KNN/Random Forest Classification App

Aplikasi berbasis **Streamlit** untuk melakukan **analisis clustering dan klasifikasi** pada dataset numerik.
Dilengkapi dengan visualisasi, evaluasi model, serta fitur pencarian data terdekat berdasarkan input pengguna.

---

## 🚀 Fitur Utama

### 🔹 1. Upload & Eksplorasi Dataset

* Mendukung format **CSV** dan **Excel (.xlsx)**
* Menampilkan:

  * Preview data
  * Informasi struktur dataset (`df.info()`)
  * Statistik deskriptif (`df.describe()`)
* Otomatis mendeteksi dan menghapus **missing values**

---

### 🔹 2. Clustering dengan K-Means

* Pilih kolom numerik untuk proses clustering
* Pilihan **normalisasi data**:

  * `MinMaxScaler` (skala 0–1)
  * `StandardScaler` (mean=0, std=1)
* Dilengkapi visualisasi:

  * **Elbow Method** → bantu menentukan jumlah cluster optimal
  * **PCA 2D Visualization** → menampilkan hasil clustering dalam dua dimensi
* Menampilkan:

  * `Silhouette Score`
  * `Davies-Bouldin Score`
* Dapat mengunduh hasil clustering dalam format `.csv`

---

### 🔹 3. Model Prediktif (KNN & Random Forest)

Setelah proses clustering, aplikasi memungkinkan pembuatan model prediktif menggunakan hasil cluster sebagai target (`label`):

#### 🧩 K-Nearest Neighbors (KNN)

* Pilih jumlah tetangga (`K`)
* Pilih algoritma:

  * `auto`
  * `kd_tree`
  * `ball_tree`
  * `brute`
* Evaluasi model:

  * Akurasi
  * Confusion Matrix
  * Classification Report
* Opsi **Hyperparameter Tuning (GridSearchCV)** untuk mencari parameter terbaik

#### 🌲 Random Forest Classifier

* Atur jumlah pohon (`n_estimators`)
* Atur kedalaman maksimum (`max_depth`)
* Menampilkan:

  * Akurasi model
  * Confusion Matrix
  * Classification Report

---

### 🔹 4. 📍 Pencarian Data Terdekat

Fitur ini memungkinkan pengguna mencari data paling mirip berdasarkan **input manual (Pemda & Tahun)**:

* Untuk **KNN**, menampilkan tetangga terdekat berdasarkan jarak antar data yang telah di-normalisasi.
* Untuk **Random Forest**, menampilkan beberapa data acak dalam **cluster yang sama**.

---

## ⚙️ Cara Menjalankan Aplikasi

### 1️⃣ Clone Repository

```bash
git clone https://github.com/ridwana22/Projek_Perhitungan_PAD.git
cd nama-repo
```

### 2️⃣ Buat Virtual Environment (Opsional)

```bash
python -m venv venv
source venv/bin/activate   # Untuk Mac/Linux
venv\Scripts\activate      # Untuk Windows
```

### 3️⃣ Instal Dependensi

```bash
pip install -r requirements.txt
```

### 4️⃣ Jalankan Aplikasi

```bash
streamlit run app.py
```

---
## 📊 Output Utama

* **Visualisasi Elbow Method**
* **PCA Clustering Plot (2D)**
* **Confusion Matrix (KNN & Random Forest)**
* **Hasil Clustering dan Prediksi dalam format tabel**
* **File hasil clustering (`DataClustered.csv`)**

---

## 💡 Catatan Teknis

* Pastikan dataset memiliki minimal **dua kolom numerik** untuk dapat dilakukan clustering.
* Kolom `namapemda` dan `tahun` dibutuhkan untuk fitur **pencarian data terdekat**.
* Model KNN dan Random Forest dilatih ulang berdasarkan hasil clustering sebelumnya.

---

