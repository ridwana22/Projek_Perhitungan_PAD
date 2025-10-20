# K-Means Clustering & KNN/Random Forest Classification App 📊

Aplikasi interaktif Streamlit ini memungkinkan pengguna untuk melakukan analisis *Clustering* menggunakan **K-Means** dan membangun model prediktif klasifikasi menggunakan **K-Nearest Neighbors (KNN)** atau **Random Forest** berdasarkan hasil *clustering*.

## Fitur Utama ✨

1.  **Unggah Data Interaktif:** Mendukung unggah file dalam format CSV atau Excel (`.csv`, `.xlsx`).
2.  **Pra-pemrosesan Data:** Otomatis menangani *missing values* (menghapus baris) dan menampilkan informasi dasar serta statistik deskriptif.
3.  **Normalisasi:** Pilihan antara **MinMaxScaler** dan **StandardScaler**.
4.  **K-Means Clustering:**
      * Visualisasi *Elbow Method* untuk membantu menentukan jumlah cluster.
      * Pengaturan jumlah cluster ($K$) secara manual.
      * Visualisasi hasil *clustering* menggunakan **Principal Component Analysis (PCA) 2D**.
      * Evaluasi *clustering* menggunakan **Silhouette Score** dan **Davies-Bouldin Score**.
5.  **Model Klasifikasi:**
      * Pilihan model: **KNN** atau **Random Forest** untuk mengklasifikasikan data baru ke dalam cluster yang sudah terbentuk.
      * Menampilkan **Akurasi**, **Classification Report**, dan **Confusion Matrix**.
      * Fitur opsional **Hyperparameter Tuning (GridSearchCV)** untuk KNN.
6.  **Pencarian Data Terdekat:** Fitur interaktif untuk mencari data terdekat berdasarkan input manual:
      * **KNN:** Menemukan $K$ tetangga terdekat di seluruh dataset.
      * **Random Forest:** Menemukan sampel acak dari data yang termasuk dalam cluster prediksi yang sama.

## Persyaratan (Requirements) 🛠️

Pastikan Anda telah menginstal pustaka Python berikut:

```bash
pandas
numpy
streamlit
scikit-learn
matplotlib
seaborn
openpyxl # Jika Anda menggunakan file .xlsx
```

Anda dapat menginstalnya melalui pip:

```bash
pip install pandas numpy streamlit scikit-learn matplotlib seaborn openpyxl
```

## Cara Menjalankan Aplikasi ▶️

1.  Simpan kode aplikasi di atas ke dalam sebuah file, misalnya `app.py`.
2.  Buka terminal atau command prompt dan arahkan ke direktori tempat Anda menyimpan file tersebut.
3.  Jalankan perintah Streamlit:

<!-- end list -->

```bash
streamlit run app.py
```

4.  Aplikasi akan terbuka secara otomatis di *browser* web Anda.

## Alur Penggunaan (Workflow) ⚙️

1.  **Unggah File:** Gunakan *sidebar* untuk mengunggah file CSV atau Excel Anda.
2.  **Pilih Kolom:** Pilih kolom numerik yang akan digunakan untuk proses *clustering*.
3.  **Pilih Metode Normalisasi:** Tentukan `MinMaxScaler` atau `StandardScaler` di *sidebar*.
4.  **Tentukan $K$:** Lihat grafik *Elbow Method* dan tentukan jumlah cluster yang diinginkan, baik secara otomatis atau manual di *sidebar*.
5.  **Pilih Model:** Pilih antara **KNN** atau **Random Forest** di bagian model prediktif. Atur *hyperparameter* yang relevan (misalnya, $K$ untuk KNN atau `n_estimators` untuk Random Forest) di *sidebar*.
6.  **Analisis Hasil:** Tinjau metrik evaluasi *clustering* dan model klasifikasi.
7.  **Cari Data Terdekat:** Gunakan bagian terakhir untuk menginput `namapemda` dan `tahun` (jika kolom ini ada) untuk menemukan data yang serupa di cluster yang sama atau merupakan tetangga terdekat.
