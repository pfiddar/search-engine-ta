from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import nltk
import string
import math
import os, pymysql
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_caching import Cache

app = Flask(__name__, template_folder='templates')
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Configure MySQL
def get_connection():
    return pymysql.connect(
        host=os.getenv("MYSQLHOST"),
        port=int(os.getenv("MYSQLPORT", 10724)),
        user=os.getenv("MYSQLUSER"),
        password=os.getenv("MYSQLPASSWORD"),
        database=os.getenv("MYSQLDATABASE"),
        cursorclass=pymysql.cursors.Cursor
    )

conn = get_connection()

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Stemmer and stopwords
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

def safe_stem(word):
    try:
        return stemmer.stem(word)
    except Exception:
        return word

def ensure_connection():
    global conn
    try:
        conn.ping(reconnect=True)  # kalau masih nyambung aman
    except:
        conn = get_db_connection()  # kalau putus, bikin koneksi baru
    return conn


@cache.memoize(timeout=250)  # Cache results for 5 minutes
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [safe_stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search_ta():
    return render_template('search.html')

@app.route('/hasil_search', methods=['GET', 'POST'])
def hasil_search_ta():
    if request.method == 'POST':
        title = request.form['judul']
        return redirect(url_for('hasil_search_ta', judul=title))
    
    title = request.args.get('judul')
    if not title:
        return redirect(url_for('search_ta'))

    # Load dataset
    df = pd.read_csv('dataset_ta.csv')

    # Check if the documents table is empty
    conn = ensure_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]

        if doc_count == 0:
            # Store documents in the database only if the table is empty
            try:
                for index, row in df.iterrows():
                    cursor.execute("INSERT INTO documents (judul, penulis, tahun, deskripsi, tautan, kata_kunci) VALUES (%s, %s, %s, %s, %s, %s)",
                                   (row['judul'], row['penulis'], row['tahun'], row['deskripsi'], row['tautan'], row['kata_kunci']))
                conn.commit()
            except Exception as e:
                conn.rollback()
                print(f"Error inserting documents: {e}")

    # Preprocess the search title
    preprocessed_title = preprocess_text(title)

    @cache.memoize(timeout=300)  # Cache results for 5 minutes
    def calculate_similarity(df, preprocessed_title):
        # TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf = vectorizer.fit_transform(df['deskripsi'].apply(preprocess_text))

        conn = ensure_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM word_document")
                cursor.execute("DELETE FROM words")
                
                feature_names = vectorizer.get_feature_names_out()
                for idx, word in enumerate(feature_names):
                    cursor.execute("INSERT INTO words (word) VALUES (%s)", (word,))
                    word_id = cursor.lastrowid
                    for doc_idx, score in enumerate(tfidf[:, idx].toarray()):
                        if score > 0:
                            cursor.execute("INSERT INTO word_document (word_id, document_id, tfidf_score) VALUES (%s, %s, %s)",
                                           (word_id, doc_idx + 1, score[0]))
                conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"Error indexing words: {e}")

        title_vector = vectorizer.transform([preprocessed_title])

        # Calculate similarity scores
        scores = cosine_similarity(title_vector, tfidf)[0]

        # Tambahkan bobot ekstra berdasarkan umpan balik relevansi
        conn = ensure_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT document_id FROM relevance_feedback WHERE query = %s", (preprocessed_title,))
                relevant_docs = cursor.fetchall()
                relevant_docs = [doc[0] for doc in relevant_docs]
                for doc_id in relevant_docs:
                    scores[doc_id - 1] *= 1.5  
        except Exception as e:
            print(f"Error processing relevance feedback: {e}")

        indices = scores.argsort()[::-1]

        similar_titles = pd.DataFrame({
            'Index': df.index[indices].tolist(),
            'Judul': df['judul'].iloc[indices].tolist(),
            'Penulis': df['penulis'].iloc[indices].tolist(),
            'Tahun': df['tahun'].iloc[indices].tolist(),
            'Deskripsi': df['deskripsi'].iloc[indices].tolist(),
            'Tautan': df['tautan'].iloc[indices].tolist(),
            'Kata_kunci': df['kata_kunci'].iloc[indices].tolist()
        })

        similar_titles = similar_titles[scores[indices] > 0]  # Remove the 0.1 threshold
        return similar_titles

    similar_titles = calculate_similarity(df, preprocessed_title)

    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = 8
    total_pages = math.ceil(len(similar_titles) / per_page)
    start = (page - 1) * per_page
    end = start + per_page

    data = []
    for index, row in similar_titles[start:end].iterrows():
        data.append({
            'index': row['Index'],
            'judul': row['Judul'],
            'penulis': row['Penulis'],
            'tahun': row['Tahun'],
            'deskripsi': row['Deskripsi'],
            'tautan': row['Tautan'],
            'kata_kunci': row['Kata_kunci']
        })

    jumlah_baris = len(similar_titles)

    return render_template('output.html', 
                           result_searching=data, 
                           jumlah_baris=jumlah_baris, 
                           page=page, 
                           total_pages=total_pages,
                           title=title)

@app.route('/detail/<int:index>', methods=['GET'])
def detail_ta(index):
    df = pd.read_csv('dataset_ta.csv')
    if index >= len(df) or index < 0:
        return "Index out of range", 404

    detail_data = {
        'judul': df.loc[index, 'judul'],
        'penulis': df.loc[index, 'penulis'],
        'tahun': df.loc[index, 'tahun'],
        'deskripsi': df.loc[index, 'deskripsi'],
        'tautan': df.loc[index, 'tautan'],
        'kata_kunci': df.loc[index, 'kata_kunci']
    }

    return render_template('detail.html', detail_data=detail_data)

@app.route('/relevance_feedback', methods=['POST'])
def relevance_feedback():
    relevant_docs = request.form.getlist('relevant_docs')
    irrelevant_docs = request.form.getlist('irrelevant_docs')
    query = request.form['query']

    if not relevant_docs and not irrelevant_docs:
        return redirect(url_for('hasil_search_ta', judul=query))

    conn = ensure_connection()
    try:
        with conn.cursor() as cursor:
            for doc_id in relevant_docs:
                cursor.execute("INSERT INTO relevance_feedback (query, document_id, relevance) VALUES (%s, %s, %s)", (query, doc_id, 1))
            for doc_id in irrelevant_docs:
                cursor.execute("INSERT INTO relevance_feedback (query, document_id, relevance) VALUES (%s, %s, %s)", (query, doc_id, 0))
            conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error saving relevance feedback: {e}")

    return redirect(url_for('hasil_search_ta', judul=query))


if __name__ == '__main__':
    app.run(debug=True)
