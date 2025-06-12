# ===================================================================================
# Proyek Akhir Machine Learning Terapan: Sistem Rekomendasi Film
# Nama: Maulana Seno Aji Yudhantara
# Email: senoaji115@gmail.com
# ID Dicoding: bang_aji
# Cohort ID Coding Camp: MC117D5Y1789
# File: recommender_system_script.py
#
# Deskripsi:
# Skrip ini mengimplementasikan sistem rekomendasi film menggunakan pendekatan
# Content-Based Filtering. Skrip akan menerima input judul film melalui
# command line dan memberikan 10 film lain yang paling mirip berdasarkan genre.
#
# Cara Menjalankan:
# python recommender_system_script.py --movie_title "Judul Film Anda"
# Contoh:
# python recommender_system_script.py --movie_title "Toy Story (1995)"
# ===================================================================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse

def load_data(file_path='movies.csv'):
    """Memuat dataset movies dari file CSV."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' tidak ditemukan.")
        return None

def prepare_data(movies_df):
    """Mempersiapkan data dengan membersihkan genre dan membuat matriks TF-IDF."""
    # Menghilangkan genre '(no genres listed)'
    movies_df['genres'] = movies_df['genres'].replace('(no genres listed)', '')
    
    # Inisialisasi TfidfVectorizer
    tfidf = TfidfVectorizer()
    
    # Melakukan fit dan transformasi pada data genres
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
    
    return tfidf_matrix, movies_df

def get_recommendations(title, movies_df, tfidf_matrix, k=10):
    """Memberikan rekomendasi film berdasarkan kemiripan konten."""
    
    # Membuat series dari judul film untuk mapping index
    indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()
    
    # Cek apakah judul film ada di dalam dataset
    if title not in indices:
        return f"Film dengan judul '{title}' tidak ditemukan dalam dataset."

    # Mengambil index dari film yang sesuai dengan judul
    idx = indices[title]

    # Menghitung cosine similarity on-the-fly
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)

    # Mengambil skor kemiripan film tersebut dengan semua film lain
    sim_scores = list(enumerate(cosine_sim[0]))

    # Mengurutkan film berdasarkan skor kemiripan
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Mengambil skor dari k film paling mirip (dimulai dari 1 untuk mengabaikan film itu sendiri)
    sim_scores = sim_scores[1:k+1]

    # Mengambil index film
    movie_indices = [i[0] for i in sim_scores]

    # Mengembalikan top k film paling mirip
    return movies_df['title'].iloc[movie_indices]

def main():
    """Fungsi utama untuk menjalankan skrip dari command line."""
    
    # Setup Argumen Parser
    parser = argparse.ArgumentParser(description="Sistem Rekomendasi Film Berbasis Konten")
    parser.add_argument('--movie_title', type=str, required=True, help='Judul film sebagai acuan rekomendasi (Contoh: "Toy Story (1995)")')
    
    args = parser.parse_args()
    
    # Proses Utama
    movies_data = load_data()
    
    if movies_data is not None:
        tfidf_matrix, movies_data = prepare_data(movies_data)
        recommendations = get_recommendations(args.movie_title, movies_data, tfidf_matrix)
        
        print("\n" + "="*50)
        print(f"REKOMENDASI FILM UNTUK: '{args.movie_title}'")
        print("="*50)
        
        if isinstance(recommendations, str):
            print(recommendations)
        else:
            for i, movie in enumerate(recommendations):
                print(f"{i+1}. {movie}")
        
        print("="*50)

if __name__ == '__main__':
    main()