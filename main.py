import pandas as pd
import numpy as np
import graphlab

def makeCSV(df):
    # input = numpy array of
    # rows of [userid, jokeid, rating]
    filename = 'data/predictions.csv'
    df.to_csv(filename)

def avg_joke_score(df):
    Output = dict()
    for i in xrange(151):
        Output[i] = 0.0
        Output[i] = df['rating'][df['joke_id']==i].mean()
    return Output

avg_score = avg_joke_score(ratings)
ratings = pd.read_table("data/ratings.dat")
ratings['joke_id'][ratings['joke_id']==151] = 15
test_data = pd.read_csv("data/test_ratings.csv")


sf = graphlab.SFrame(ratings)
m1 = graphlab.factorization_recommender.create(sf, max_iterations=50, num_factors=2, linear_regularization=1e-12, user_id='user_id', item_id='joke_id', target='rating', solver='als')


test_sf = graphlab.SFrame(test_data)
predicted_ratings = np.array(m1.predict(test_sf))
output_df = test_data[['user_id','joke_id']]
output_df['rating'] = predicted_ratings

for i in avg_score:
    output_df['rating'][output_df['joke_id']==i] += avg_score[i]*0.1

makeCSV(output_df)
# python src/slack_poster.py data/predictions.csv
