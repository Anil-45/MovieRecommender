import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#importing data
data=pd.read_csv("movie_dataset.csv")
#selecting the features to consider
features=['keywords','genres','cast','director']

#replacing NaN values with empty string
for feature in features:
    data[feature]=data[feature].fillna('')

#combine features of rows    
def combine(row):
    try:
        return row['keywords']+" "+row['genres']+" "+row['cast']+ " "+row['director']
    except:
        print("Exception",row)
    
data["combined_features"]=data.apply(combine,axis=1)

#convert to count matrix
cv=CountVectorizer()
count_matrix=cv.fit_transform(data["combined_features"])

#cosine similarity
cos_sim=cosine_similarity(count_matrix)
movie="Avatar"

#getting title and index methods
def get_title(index):
    return data[data.index==index]["title"].values[0]
def get_index(title):
    return data[data.title==title]["index"].values[0]

#getting index of movie
movie_index=get_index(movie)
similar_movies=list(enumerate(cos_sim[movie_index]))
#sort to get most similar movie'
sort_mov=sorted(similar_movies,key=lambda x:x[1],reverse=True)
#printing 15 similar movies
i=0
for mov in sort_mov:
    print(get_title(mov[0]))
    i=i+1
    if i>15:
        break
    
