import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from preprocess import *
import graphlab
from graphlab.toolkits import cross_validation
from graphlab.toolkits.model_parameter_search import grid_search
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
def feature_creation(df):
    df['joker_id'] = xrange(1, 151)
    df['length'] = df['jokes'].apply(lambda x: len(x.split()))
    df['dif_bet'] = df['jokes'].apply(lambda x: 1 if 'difference between' in x else 0)
    df['man_and_woman'] = df['jokes'].apply(lambda x: 1 if 'man' in x.split() and 'woman' in x.split() else 0)
    df['woman'] = df['jokes'].apply(lambda x: 1 if 'woman' in x else 0)
    df['man'] = df['jokes'].apply(lambda x: 1 if 'man' in x.split() else 0)
    df['q_and_a'] = df['jokes'].apply(lambda x: 1 if 'q' in x.split() else 0)
    return df

def grid_search_model(rating):
    rating = graphlab.SFrame(rating)
    mf_model = graphlab.recommender.factorization_recommender.create(rating, target = 'rating', solver = 'als', user_id = 'user_id', item_id = 'joke_id')
    predicted_rating = mf_model.predict(rating)
    coeffs = mf_model.get('coefficients')
    print mf_model.training_rmse
    folds = cross_validation.KFold(rating, 5)
    params = dict([ ('target', 'rating'),
                    ('solver','als'), ('user_id','user_id'), ('item_id','joke_id'),
                       ('max_iterations', [100, 200, 300]),
                       ('linear_regularization', [1e-6, 1e-8, 1e-10, 1e-12]),
                       ('num_factors', [6, 8, 10, 12])])
    job = grid_search.create(folds,
                            graphlab.recommender.factorization_recommender.create,
                            params)

    job.get_results()

def get_neighbor(joke_id, num_neighbors=5):
    return np.argsort(cov_mat[joke_id-1])[:-num_neighbors-1:-1]

def get_average_rating_from_neighbors(rating, user_id, joke_id, num_neighbors=5):
    neighbors_rating = []
    user = rating[rating.user_id==user_id]
    # print index
    for i in get_neighbor(joke_id, num_neighbors):
        if user[rating.joke_id==i].empty == False:
            # print user[rating.joke_id==i]['rating']
            neighbors_rating.append(user[rating.joke_id==i]['rating'].values)
            # print neighbors_rating
    if len(neighbors_rating) == 0:
        return 0
    return np.mean(np.array(neighbors_rating))

def row_neighbors(row):
    return get_average_rating_from_neighbors(rating, row.user_id, row.joke_id)

if __name__ == '__main__':
    df = pd.read_csv('../data/jokes.txt', names=['jokes'])
    jokes = df.jokes
    clean_parsed = clean_parse(jokes)
    # f_list_count, data_fes_c, cv = text_feature_extraction(CountVectorizer, clean_parsed)
    f_list_tf, data_fes_t, tf = text_feature_extraction(TfidfVectorizer, clean_parsed)
    dense = data_fes_t.todense()
    word_mat = pd.DataFrame(dense, columns=f_list_tf)
    df = feature_creation(df)

    rating = pd.read_table('../data/ratings.dat')
    rating['joke_id'][rating['joke_id']==151] = 15

    #grid_search_model(rating)

    pca = PCA(n_components=60)
    reduced_mat = pca.fit_transform(dense)
    # scree_plot(pca)
    # print(pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_))
    # plt.show()
    cov_mat = cosine_similarity(dense, dense)
    test = pd.read_csv('../data/test_ratings.csv')
    test['joke_id'][test['joke_id']==151] = 15
    # prediction = []
    # for user,joke in zip(test.user_id[:10], test.joke_id[:10]):
    #     prediction.append(get_average_rating_from_neighbors(rating,user, joke, num_neighbors = 10))
    # print prediction
