import flask
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import request

app = flask.Flask(__name__, template_folder='./')


@app.route('/')
def index():
    return(flask.render_template('index.html'))

@app.route('/search')
def search_ta():
    return(flask.render_template('search.html'))

@app.route('/hasil_search', methods=['POST'])
def hasil_search_ta():
    # Load your dataset
    df = pd.read_csv('dataset_ta.csv')

    # Preprocess the data
    nltk.download('punkt')
    nltk.download('stopwords')

    def preprocess_text(text):
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove stop words
        stop_words = set(stopwords.words('indonesian'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Stem the tokens
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
        
        return ' '.join(tokens)

    # Apply the preprocessing function to each title
    df['judul'] = df['judul'].apply(preprocess_text)

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer to the data and transform it
    tfidf = vectorizer.fit_transform(df['judul'])

    # Calculate the cosine similarity
    similarity = cosine_similarity(tfidf)

    # Function to search for similar titles
    def search_similar_titles(title, num_results=None):
        # Preprocess the title
        title = preprocess_text(title)
        
        # Get the TF-IDF vector for the title
        title_vector = vectorizer.transform([title])
        
        # Calculate the similarity scores
        scores = cosine_similarity(title_vector, tfidf)[0]
        
        # Get the indices of the similar titles
        indices = scores.argsort()
        
        # Create a DataFrame of the similar titles
        similar_titles = pd.DataFrame({
            'Judul': df['judul'].iloc[indices].tolist(),
            'Penulis': df['penulis'].iloc[indices].tolist(),
            'Tahun': df['tahun'].iloc[indices].tolist(),
            'Deskripsi': df['deskripsi'].iloc[indices].tolist(),
            'Tautan': df['tautan'].iloc[indices].tolist(),
            'Kata_kunci': df['kata_kunci'].iloc[indices].tolist()
        })

        # Filter the titles with similarity scores above 0.5
        similar_titles = similar_titles[similar_titles['Judul'].apply(lambda x: scores[indices[similar_titles['Judul'].tolist().index(x)]] > 0.1)]

        return similar_titles

    # Example usage
    title = request.form['judul']
    similar_titles = search_similar_titles(title)

    data = []

    for index, row in similar_titles.iterrows():
        data.append({'judul': row['Judul'], 'penulis': row['Penulis'], 'tahun': row['Tahun'], 'deskripsi': row['Deskripsi'], 'tautan': row['Tautan'], 'kata_kunci': row['Kata_kunci']})

    jumlah_baris = len(similar_titles)
    
    # Kirim hasil ke template HTML
    return flask.render_template('output.html', result_searching=data, jumlah_baris=jumlah_baris)

@app.route('/output.html', methods=['GET'])
def output_ta():
    return(flask.render_template('output.html'))


if __name__ == '__main__':
    app.run(debug=True)











