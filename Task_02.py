# # TASK 2 
# # Restaurant Recommendation
# # AIM =  Create a restaurant recommendation system based on user preferences

# # STEP1 = IMPORT REQUIERED LIBRARIES
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# # STEP2 = Data Preprocessing
data = pd.read_csv('Dataset.csv')

# Handle missing values
data.fillna('', inplace=True)
# preprocessing 
# Combine relevant features for representation
data['combined_features'] = data['Cuisines'] + ' ' + data['Price range'].astype(str)

# content based filtering
vector = TfidfVectorizer(stop_words='english')
matrix = vector.fit_transform(data['combined_features'])

# cosine similarity
cos = linear_kernel(matrix, matrix)

# # STEP3 = to recommend restaurants# get restaurant recommendations based on user preferences
def get_recommendations(user_preferences):
    # a new dataframe with the user's preferences
    user = pd.DataFrame([user_preferences], columns=['Cuisines', 'Price range'])
    user['combined_features'] = user['Cuisines'] + ' ' + user['Price range'].astype(str)

    user_matrix = vector.transform(user['combined_features'])

    # similarity between user preferences and all restaurants
    scores = list(enumerate(cos[user_matrix.nonzero()[0][0]]))

    # Sort restaurants based on similarity 
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # top 5 restaurant recommendations
    top_restaurants = scores[1:6]

    # Return the restaurant details
    return data.iloc[[restaurant[0] for restaurant in top_restaurants]][['Restaurant Name', 'Cuisines', 'Average Cost for two', 'Aggregate rating']]
    

# # STEP4 = Test the recommendation system
# test recommendations
# Sample user preferences
user_preferences = {'Cuisines': 'Italian', 'Price range': 2}

# Get recommendations
recommendations = get_recommendations(user_preferences)

print(recommendations)
