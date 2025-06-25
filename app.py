from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from flask import Flask, render_template, send_file, request
from flask import Flask, render_template, send_file
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import numpy as np
import base64

nltk.download('vader_lexicon')

app = Flask(__name__)
sia = SentimentIntensityAnalyzer()

def scrape_news():
    url = "https://news.google.com/rss"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "xml")
    headlines = [item.title.text for item in soup.findAll("item")[:15]]
    return headlines

def analyze_sentiment(headlines):
    results = []
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

    for headline in headlines:
        score = sia.polarity_scores(headline)['compound']
        if score > 0.05:
            sentiment = 'Positive'
        elif score < -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        counts[sentiment] += 1
        results.append((headline, sentiment))

    return results, counts

@app.template_filter('b64encode')
def b64encode_filter(data):
    return base64.b64encode(data).decode('utf-8')


@app.route('/')
def index():
    headlines = scrape_news()
    analyzed, counts = analyze_sentiment(headlines)

    # Plot sentiment distribution
    labels = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=['green', 'red', 'gray'])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")

    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_data = img.read()

    return render_template('index.html', headlines=analyzed, plot_image=plot_data)

@app.route('/wordcloud')
def wordcloud():
    headlines = scrape_news()
    text = ' '.join(headlines)
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)

    img = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(img, format='PNG')
    img.seek(0)

    return send_file(img, mimetype='image/png')
@app.route('/topics')
def topics():
    headlines = scrape_news()
    
    # Vectorize
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(headlines)

    # LDA Model
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(X)

    topic_words = []
    for topic in lda.components_:
        words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-6:]]
        topic_words.append(words[::-1])

    return render_template('topics.html', topics=topic_words)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


