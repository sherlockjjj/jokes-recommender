import pandas as pd
import csv
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
def parse(joke):
    clean_joke = re.sub("[(\\n)(\\r)(&#039)]", " ", joke)
    clean_joke = re.sub("<p[^>]*>", "", clean_joke)
    words = re.sub("[^a-zA-Z]", " ", clean_joke).lower().split()
    stop_words = ['br', 'quot']
    parsed = [word for word in words if word not in stop_words]
    return ( " ".join(parsed))

def open_dat(filename):
    with open('data/jokes.dat') as f:
        data = f.read()
    return data

def create_txt(clean_jokes, filename):
    text_file = open(filename, "w")
    for j in clean_jokes:
        line = j + '\n'
        text_file.write(line)
    text_file.close()

def clean_parse(data):
    clean_parsed = [stem_text(d) for d in data]
    return clean_parsed

def stem_text(text):
    ps = PorterStemmer()
    wnl = WordNetLemmatizer()
    words = [wnl.lemmatize(ps.stem(w)) for w in text.split()]
    return( " ".join(words))

def text_feature_extraction(vec, clean_parsed, num_fes=200):
    vectorizer = vec(analyzer = "word", stop_words = 'english', max_features = num_fes)
    data_fes = vectorizer.fit_transform(clean_parsed)
    f_list = list(vectorizer.get_feature_names())
    return f_list, data_fes, vectorizer

def scree_plot(pca, title=None):
    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 6), dpi=250)
    ax = plt.subplot(111)
    ax.bar(ind, vals, 0.35,
           color=[(0.949, 0.718, 0.004),
                  (0.898, 0.49, 0.016),
                  (0.863, 0, 0.188),
                  (0.694, 0, 0.345),
                  (0.486, 0.216, 0.541),
                  (0.204, 0.396, 0.667),
                  (0.035, 0.635, 0.459),
                  (0.486, 0.722, 0.329),
                 ])

    for i in xrange(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)

    ax.set_xticklabels(ind,
                       fontsize=12)

    ax.set_ylim(0, max(vals)+0.05)
    ax.set_xlim(0-0.45, 8+0.45)

    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)

    if title is not None:
        plt.title(title, fontsize=16)

if __name__ == '__main__':
    data = open_dat('data/jokes.dat')
    jokes = data.split('/p')[:150]
    clean_jokes = [parse(j) for j in jokes]
    create_txt(clean_jokes, "data/jokes.txt")
