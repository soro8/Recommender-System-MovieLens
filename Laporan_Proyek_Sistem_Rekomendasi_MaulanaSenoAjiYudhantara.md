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

Proses persiapan data dilakukan melalui beberapa tahapan untuk memastikan data siap untuk dianalisis dan dimodelkan.

### 4.1. Persiapan Data Umum
Langkah pertama yang dilakukan adalah menggabungkan dataset `ratings` dan `movies` menjadi satu DataFrame.
-   **Penggabungan Data**: Proses `merge` dilakukan berdasarkan kolom `movieId`. **Alasan**: Ini bertujuan untuk memudahkan proses analisis data eksplorasi (EDA), di mana kita bisa melihat judul dan genre film secara langsung bersamaan dengan rating yang diberikan oleh pengguna.

### 4.2. Persiapan Data untuk Model Content-Based Filtering
Tujuan utama di sini adalah mengubah fitur `genres` menjadi vektor numerik yang dapat diukur kemiripannya.
-   **Pembersihan Data Genre**: Data dengan genre `(no genres listed)` dihapus dari DataFrame `movies` karena tidak memberikan informasi konten.
-   **TF-IDF Vectorization**: Teknik TF-IDF diterapkan pada kolom `genres`. **Alasan**: TF-IDF mampu memberikan bobot yang lebih tinggi pada genre yang lebih "unik" untuk sebuah film, sehingga representasi vektornya menjadi lebih bermakna dibandingkan hanya menghitung frekuensi kemunculan.

### 4.3. Persiapan Data untuk Model Collaborative Filtering
Tujuan di sini adalah mempersiapkan data interaksi pengguna untuk model Deep Learning.
-   **Encoding Fitur**: `userId` dan `movieId` diubah menjadi indeks integer yang berurutan (mulai dari 0). **Alasan**: Lapisan *Embedding* pada Keras/TensorFlow memerlukan input berupa indeks integer untuk dapat memetakan setiap ID ke vektor latennya secara efisien.
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
-   **Hasil Rekomendasi (Top 5 untuk Pengguna dengan ID 50)**:
    | No. | Judul Film Rekomendasi |
    |:---:|:---|
    | 1 | Paths of Glory (1957) |
    | 2 | Princess Bride, The (1987) |
    | 3 | Celebration, The (Festen) (1998) |
    | 4 | Five Easy Pieces (1970) |
    | 5 | Guess Who's Coming to Dinner (1967) |

## 6. Evaluation

Pada tahap evaluasi, kita akan menganalisis hasil dari kedua model yang telah kita kembangkan. Setiap model memiliki metrik evaluasi yang sesuai dengan pendekatannya untuk memastikan hasil dapat divalidasi dan diukur.

### 6.1. Evaluasi Content-Based Filtering

Model ini dievaluasi secara kuantitatif dan kualitatif untuk mendapatkan pemahaman yang komprehensif tentang kinerjanya.

#### Evaluasi Kuantitatif
Untuk mengukur kinerja model, metrik **Precision@k** digunakan.
-   **Penjelasan Metrik Precision@k**: Metrik ini mengukur proporsi item yang relevan dari 'k' item teratas yang direkomendasikan. Dalam konteks ini, "item relevan" didefinisikan sebagai film yang juga disukai oleh pengguna (rating ≥ 4.0). Formula Precision@k adalah:
    $$\text{Precision@k} = \frac{\text{|(Item Rekomendasi @k)} \cap \text{(Item Relevan)}|}{k}$$
-   **Metodologi Perhitungan**: Perhitungan dilakukan dengan simulasi pada data pengguna. Untuk setiap pengguna, satu film yang ia sukai dijadikan dasar rekomendasi, dan hasilnya dibandingkan dengan daftar film lain yang juga ia sukai. Proses ini diulang untuk semua pengguna yang memenuhi syarat dan hasilnya dirata-ratakan. Kode implementasi untuk perhitungan ini tersedia di dalam notebook.
-   **Hasil**: Setelah menjalankan kode simulasi di notebook, model Content-Based Filtering mencapai **Precision@10 sebesar 0.0454 atau sekitar 4.5%**.
-   **Analisis**: Nilai presisi ini terbilang rendah. Hal ini mengindikasikan bahwa kemiripan genre saja seringkali tidak cukup untuk menebak secara akurat film lain yang akan disukai oleh seorang pengguna. Selera pengguna bersifat kompleks dan tidak hanya ditentukan oleh genre. Namun, metrik ini tetap menunjukkan bahwa model memiliki kemampuan dasar untuk menemukan item yang relevan, meskipun dalam frekuensi yang kecil. Rendahnya skor ini juga menyoroti keterbatasan utama dari pendekatan Content-Based yang murni bergantung pada fitur item dan tidak mempelajari perilaku atau selera personal pengguna.

#### Evaluasi Kualitatif
Meskipun presisinya rendah, secara kualitatif model ini terbukti logis. Saat merekomendasikan film berdasarkan "Toy Story (1995)", hasilnya adalah film-film animasi lain yang genrenya sangat mirip, seperti "Antz (1998)" dan "Monsters, Inc. (2001)". Hal ini menunjukkan bahwa model bekerja persis seperti yang dirancang, yaitu menemukan item berdasarkan kemiripan konten.

### 6.2. Evaluasi Collaborative Filtering

Model ini dievaluasi secara kuantitatif dan kualitatif.

#### Evaluasi Kuantitatif
Evaluasi kuantitatif menggunakan metrik **Root Mean Squared Error (RMSE)**.
-   **Penjelasan Metrik RMSE**: RMSE mengukur rata-rata magnitudo kesalahan antara rating yang diprediksi oleh model dengan rating yang sebenarnya diberikan oleh pengguna. Nilai yang lebih rendah menandakan model lebih akurat. Formula RMSE:
    $$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$
-   **Hasil**:
    ![Metrik Pelatihan Model](images/Model%20Metrics.png)
    *Gambar 4. Grafik metrik RMSE selama proses pelatihan.*
-   **Analisis**: Dari grafik, terlihat nilai RMSE untuk data latih dan validasi sama-sama menurun dan stabil. Nilai akhir **val_root_mean_squared_error** yang konsisten di sekitar **0.205** menunjukkan tingkat kesalahan prediksi rating yang rendah dan model dapat melakukan generalisasi dengan baik pada data yang belum pernah dilihatnya.

#### Evaluasi Kualitatif
Secara kualitatif, kita melihat contoh rekomendasi yang diberikan untuk **pengguna ID 50**. Pengguna ini menyukai film-film klasik dan drama serius seperti *'2001: A Space Odyssey'*, *'Lawrence of Arabia'*, dan *'Apocalypse Now'*. Model kemudian berhasil merekomendasikan film-film yang relevan secara tematik dan artistik seperti *'Paths of Glory'* (drama perang), *'The Hustler'* (drama), dan *'Eternal Sunshine of the Spotless Mind'* (drama/romance). Ini menunjukkan kemampuan model dalam menangkap selera pengguna yang lebih *niche* dan memberikan rekomendasi yang beragam namun tetap personal.

## 7. Daftar Referensi
Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to Recommender Systems Handbook. In *Recommender Systems Handbook* (pp. 1-35). Springer, Boston, MA.