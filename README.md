# Proyek Akhir: Sistem Rekomendasi Film

Repository ini berisi proyek machine learning untuk membangun sistem rekomendasi film menggunakan dua pendekatan: **Content-Based Filtering** dan **Collaborative Filtering**. Proyek ini merupakan submission akhir untuk modul **Machine Learning Terapan** dalam program **Coding Camp 2025 by DBS Foundation**.

| Keterangan | Informasi |
| :--- | :--- |
| **Nama Lengkap** | Maulana Seno Aji Yudhantara |
| **Cohort ID** | MC117D5Y1789 |
| **Mentor** | Yeftha Joshua Ezekiel |

---

## 📝 Deskripsi Proyek

Tujuan dari proyek ini adalah membangun sistem rekomendasi yang mampu memberikan rekomendasi film yang relevan dan dipersonalisasi kepada pengguna. Sistem ini dirancang untuk membantu pengguna menavigasi katalog film yang besar dan menemukan judul-judul baru yang kemungkinan besar akan mereka sukai.

## 🚀 Latar Belakang

Di era digital dengan melimpahnya konten media, pengguna seringkali dihadapkan pada masalah *choice paralysis*. Sistem rekomendasi menjadi solusi krusial bagi platform seperti Netflix dan Spotify untuk meningkatkan pengalaman pengguna. Program **Coding Camp 2025 by DBS Foundation** berkolaborasi dengan **Dicoding** memberikan kesempatan untuk mengaplikasikan ilmu *machine learning* dalam menyelesaikan masalah nyata ini.

## 📁 Struktur Repository

Repository ini terdiri dari beberapa file dan folder utama:

-   `[RecomSystem]_Submission_Akhir_MLT_MaulanaSenoAjiYudhantara.ipynb`: Notebook Jupyter yang berisi seluruh proses analisis dan pemodelan kedua model secara detail.
-   `Laporan_Proyek_Sistem_Rekomendasi_MaulanaSenoAjiYudhantara.md`: Laporan lengkap dalam format Markdown yang menjelaskan setiap tahapan proyek.
-   `recommender_system_script.py`: Skrip Python bersih yang mengimplementasikan model Content-Based Filtering dan dapat dijalankan melalui terminal.
-   `movies.csv` & `ratings.csv`: Dataset mentah dari MovieLens yang digunakan dalam proyek.
-   `images/`: Folder yang berisi semua gambar dan visualisasi yang digunakan dalam laporan.
-   `requirements.txt`: Daftar library Python yang dibutuhkan untuk menjalankan proyek ini.

## 🛠️ Alur Kerja Proyek

Proyek ini dikerjakan dengan mengikuti alur kerja *data science* standar:
1.  **Data Understanding**: Memahami karakteristik dataset MovieLens dan melakukan analisis data eksplorasi (EDA) untuk menemukan wawasan awal.
2.  **Data Preparation**: Melakukan persiapan data secara terpisah untuk kedua model:
    -   **Content-Based**: Memproses fitur `genres` menggunakan TF-IDF Vectorizer.
    -   **Collaborative Filtering**: Melakukan encoding pada `userId` dan `movieId`, serta normalisasi `rating` untuk model Deep Learning.
3.  **Modeling**: Membangun dan melatih dua model:
    -   **Content-Based Filtering** dengan Cosine Similarity.
    -   **Collaborative Filtering** dengan arsitektur Deep Learning menggunakan Keras.
4.  **Evaluation**: Mengevaluasi kedua model. Model Content-Based dievaluasi secara kualitatif, sementara model Collaborative Filtering dievaluasi secara kuantitatif dengan metrik RMSE.

## 📊 Hasil

Proyek ini berhasil menghasilkan dua model fungsional:
-   **Model Content-Based** mampu memberikan rekomendasi yang logis berdasarkan kemiripan genre.
-   **Model Collaborative Filtering** berhasil mempelajari preferensi laten pengguna dengan baik, ditunjukkan oleh nilai **RMSE yang rendah dan stabil (~0.205)** pada data validasi.

## 📄 Laporan Lengkap

Untuk analisis yang lebih mendalam mengenai setiap tahapan, arsitektur model, justifikasi, dan visualisasi, silakan merujuk ke laporan lengkap proyek:
-   **[Laporan_Proyek_Sistem_Rekomendasi_MaulanaSenoAjiYudhantara.md](Laporan_Proyek_Sistem_Rekomendasi_MaulanaSenoAjiYudhantara.md)**

## 🚀 Cara Menjalankan Proyek

Untuk menjalankan proyek ini di lingkungan lokal Anda, ikuti langkah-langkah berikut:

1.  **Clone repository ini:**
    ```bash
    git clone https://github.com/bangaji313/Recommender-System-MovieLens.git
    cd Recommender-System-MovieLens
    ```

2.  **(Opsional tapi direkomendasikan) Buat virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
    ```

3.  **Install semua library yang dibutuhkan:**
    ```bash
    pip install -r requirements.txt
    ```
    *Catatan: Jika Anda belum memiliki file `requirements.txt`, buat dengan menjalankan `pip freeze > requirements.txt` di terminal Anda setelah semua library terpasang.*

4.  **Jalankan Notebook atau Skrip:**
    -   Untuk melihat analisis detail dan model Deep Learning, buka dan jalankan file `.ipynb` menggunakan Jupyter Notebook atau Jupyter Lab.
    -   Untuk mencoba model Content-Based Filtering secara langsung, eksekusi skrip Python dari terminal:
        ```bash
        python recommender_system_script.py --movie_title "Toy Story (1995)"
        ```
