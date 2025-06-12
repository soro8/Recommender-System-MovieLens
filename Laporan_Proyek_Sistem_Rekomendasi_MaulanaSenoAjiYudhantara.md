# Laporan Proyek Akhir: Membuat Model Sistem Rekomendasi Film

- **Nama:** Maulana Seno Aji Yudhantara
- **Email:** senoaji115@gmail.com
- **ID Dicoding:** bang_aji
- **Cohort ID Coding Camp:** MC117D5Y1789

## 1. Project Overview

Sistem rekomendasi adalah salah satu aplikasi paling populer dari *machine learning* di dunia industri saat ini. Platform seperti Netflix, Spotify, dan Amazon sangat bergantung pada sistem ini untuk memberikan pengalaman yang dipersonalisasi kepada pengguna. Dengan merekomendasikan konten atau produk yang relevan, perusahaan dapat secara signifikan meningkatkan keterlibatan, kepuasan, dan loyalitas pelanggan.

Proyek ini menjadi penting karena di tengah ledakan informasi dan banyaknya pilihan film yang tersedia, pengguna seringkali mengalami *choice paralysis* atau kesulitan dalam menemukan film yang benar-benar sesuai dengan selera mereka. Sistem rekomendasi yang efektif tidak hanya membantu pengguna menemukan film baru yang kemungkinan besar akan mereka sukai, tetapi juga membantu platform untuk memaksimalkan konsumsi konten pada katalog mereka.

## 2. Business Understanding

### 2.1. Problem Statement

-   Bagaimana cara membangun sebuah sistem yang dapat memberikan rekomendasi film yang relevan dan dipersonalisasi kepada setiap pengguna?
-   Bagaimana cara membantu pengguna untuk menemukan film-film baru (*serendipity*) yang mungkin tidak akan mereka temukan sendiri, namun sesuai dengan pola selera mereka?

### 2.2. Goals

-   Membangun sebuah model sistem rekomendasi yang dapat merekomendasikan film berdasarkan kemiripan konten (dalam hal ini, genre).
-   Membangun sebuah model sistem rekomendasi yang dapat merekomendasikan film berdasarkan riwayat rating dari pengguna lain (pola kolaboratif).
-   Mengevaluasi performa dari kedua model tersebut menggunakan metrik yang sesuai.

### 2.3. Solution Approach

Untuk mencapai tujuan tersebut, saya akan mengembangkan dua jenis model sistem rekomendasi untuk dibandingkan:

1.  **Content-Based Filtering**: Pendekatan ini dipilih karena dapat memberikan rekomendasi yang transparan dan mudah dijelaskan (misalnya, "Anda direkomendasikan film ini karena Anda menyukai film bergenre serupa"). Model ini akan dibangun dengan teknik **TF-IDF Vectorizer** untuk memproses fitur genre dan **Cosine Similarity** untuk menghitung kemiripan antar film.
2.  **Collaborative Filtering**: Pendekatan ini dipilih karena kemampuannya untuk menemukan pola yang kompleks dan menghasilkan rekomendasi yang bersifat *serendipitous*. Model ini akan dibangun menggunakan pendekatan **Deep Learning** dengan arsitektur *neural network* yang memanfaatkan lapisan *Embedding* untuk mempelajari fitur laten dari pengguna dan film.

## 3. Data Understanding

### 3.1. Sumber Data

Dataset yang digunakan dalam proyek ini adalah **"MovieLens Latest Datasets (small)"** yang bersumber dari platform Kaggle, disediakan oleh GroupLens.
-   **Tautan:** [https://www.kaggle.com/datasets/grouplens/movielens-latest-small](https://www.kaggle.com/datasets/grouplens/movielens-latest-small)

Dataset ini terdiri dari 100,836 rating dari 610 pengguna untuk 9,742 film.

### 3.2. Penjelasan Variabel

Dataset utama yang digunakan adalah `movies.csv` dan `ratings.csv`.
-   **movies.csv**:
    -   `movieId`: ID unik untuk setiap film.
    -   `title`: Judul film beserta tahun rilis.
    -   `genres`: Genre film yang dipisahkan oleh karakter `|`.
-   **ratings.csv**:
    -   `userId`: ID unik untuk setiap pengguna.
    -   `movieId`: ID film yang diberi rating.
    -   `rating`: Rating yang diberikan pengguna (skala 0.5 - 5.0).
    -   `timestamp`: Waktu pemberian rating.

### 3.3. Exploratory Data Analysis (EDA)

Pada tahap EDA, beberapa visualisasi dibuat untuk mendapatkan wawasan dari data.

![Distribusi Jumlah Rating Film](images/Distribusi%20Jumlah%20Rating%20Film.png)

*Gambar 1. Distribusi rating menunjukkan bahwa pengguna cenderung memberikan rating tinggi (3.0 - 5.0).*

![Jumlah Film per Genre](images/Jumlah%20Film%20per%20Genre.png)

*Gambar 2. Genre Drama dan Comedy adalah yang paling dominan dalam dataset.*

![Top 10 Film dengan Jumlah Rating Terbanyak](images/Top%2010%20Film%20dengan%20Jumlah%20Rating%20Terbanyak.png)

*Gambar 3. Film-film klasik dan populer seperti Forrest Gump dan Shawshank Redemption adalah yang paling banyak menerima rating.*

**Rangkuman EDA:**
-   Dataset bersih dan tidak memiliki nilai yang hilang.
-   Data rating menunjukkan kecenderungan pengguna memberikan penilaian positif.
-   Genre Drama dan Comedy mendominasi, yang perlu diperhatikan agar rekomendasi tidak bias ke genre tersebut.
-   Dataset ini sangat cocok untuk kedua pendekatan, dengan `genres` untuk Content-Based dan data interaksi `userId-movieId-rating` untuk Collaborative Filtering.

## 4. Data Preparation

Proses persiapan data dilakukan secara terpisah untuk kedua model.

### 4.1. Persiapan Data untuk Content-Based Filtering

Tujuan utama di sini adalah mengubah fitur `genres` menjadi vektor numerik yang dapat diukur kemiripannya.
-   **Pembersihan Data Genre**: Data dengan genre `(no genres listed)` dihapus karena tidak memberikan informasi.
-   **TF-IDF Vectorization**: Teknik TF-IDF diterapkan pada kolom `genres`. **Alasan**: TF-IDF mampu memberikan bobot yang lebih tinggi pada genre yang lebih "unik" untuk sebuah film, sehingga representasi vektornya menjadi lebih bermakna dibandingkan hanya menghitung frekuensi kemunculan.

### 4.2. Persiapan Data untuk Collaborative Filtering

Tujuan di sini adalah mempersiapkan data interaksi pengguna untuk model Deep Learning.
-   **Encoding Fitur**: `userId` dan `movieId` diubah menjadi indeks integer yang berurutan (mulai dari 0). **Alasan**: Lapisan *Embedding* pada Keras/TensorFlow memerlukan input berupa indeks integer untuk dapat memetakan setiap ID ke vektor latennya.
-   **Normalisasi Rating**: Nilai `rating` dinormalisasi ke dalam rentang [0, 1]. **Alasan**: Model *neural network*, terutama dengan fungsi aktivasi sigmoid di akhir, bekerja paling baik dengan nilai target dalam rentang ini, yang membantu proses konvergensi saat pelatihan.
-   **Pembagian Data**: Data dibagi menjadi 80% data latih dan 20% data validasi untuk melatih dan mengevaluasi model secara adil.

## 5. Modeling and Result

Dua model dikembangkan sesuai dengan pendekatan yang telah direncanakan.

### 5.1. Content-Based Filtering

-   **Cara Kerja**: Model ini dibangun dengan menghitung **Cosine Similarity** antar semua film berdasarkan matriks TF-IDF dari genre mereka. Cosine Similarity mengukur sudut antara dua vektor, di mana nilai 1 berarti identik dan 0 berarti tidak ada kemiripan.
-   **Kelebihan**: Tidak memerlukan data pengguna lain (mengatasi *cold start* untuk item baru), dan rekomendasinya transparan.
-   **Kekurangan**: Terbatas pada fitur yang ada (kurang *serendipity*) dan bisa menjadi terlalu spesifik (jika pengguna hanya suka satu genre, rekomendasinya akan monoton).
-   **Hasil Rekomendasi (Top 5 untuk "Toy Story (1995)")**:
    | No. | Judul Film Rekomendasi |
    |:---:|:---|
    | 1 | Antz (1998) |
    | 2 | Toy Story 2 (1999) |
    | 3 | Adventures of Rocky and Bullwinkle, The (2000) |
    | 4 | Emperor's New Groove, The (2000) |
    | 5 | Monsters, Inc. (2001) |

### 5.2. Collaborative Filtering

-   **Cara Kerja**: Model ini menggunakan arsitektur *neural network* dengan lapisan *Embedding*. Model belajar merepresentasikan setiap pengguna dan film sebagai vektor fitur laten. Prediksi rating dilakukan dengan menghitung *dot product* antara vektor pengguna dan vektor film, yang menandakan tingkat kecocokan keduanya.
-   **Kelebihan**: Mampu menemukan pola yang kompleks dan tak terduga (*serendipity*), serta tidak bergantung pada fitur item.
-   **Kekurangan**: Mengalami masalah *cold start* (tidak bisa memberi rekomendasi untuk pengguna/item baru) dan hasilnya kurang bisa diinterpretasikan (*black box*).
-   **Hasil Rekomendasi (Top 5 untuk Pengguna Acak ID 160)**:
    | No. | Judul Film Rekomendasi |
    |:---:|:---|
    | 1 | It Happened One Night (1934) |
    | 2 | Casablanca (1942) |
    | 3 | Sunset Blvd. (a.k.a. Sunset Boulevard) (1950) |
    | 4 | Rebecca (1940) |
    | 5 | Notorious (1946) |

## 6. Evaluation

### 6.1. Evaluasi Content-Based Filtering

Evaluasi dilakukan secara **kualitatif**. Dengan melihat hasil rekomendasi untuk "Toy Story (1995)", model memberikan film-film animasi lain yang genrenya sangat mirip. Hal ini menunjukkan model bekerja sesuai harapan dan logis.

### 6.2. Evaluasi Collaborative Filtering

Evaluasi model ini menggunakan metrik **Root Mean Squared Error (RMSE)**. RMSE mengukur rata-rata magnitudo kesalahan antara rating yang diprediksi dan rating sebenarnya. Nilai yang lebih rendah menandakan model lebih akurat.

Formula RMSE: $$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

![Metrik Pelatihan Model](images/Model%20Metrics.png)

*Gambar 4. Grafik metrik RMSE selama proses pelatihan.*

Dari grafik, terlihat nilai RMSE untuk data latih dan validasi sama-sama menurun dan stabil, menunjukkan model belajar dengan baik tanpa *overfitting*. Nilai akhir **val_root_mean_squared_error** yang stabil di sekitar **0.205** menunjukkan tingkat kesalahan prediksi rating yang rendah.

## 7. Daftar Referensi
Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to Recommender Systems Handbook. In *Recommender Systems Handbook* (pp. 1-35). Springer, Boston, MA.