#pylint:disable=E0401
'''
import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
movies = pd.read_csv('archive/tmdb_5000_movies.csv')
credits = pd.read_csv('archive/tmdb_5000_credits.csv')
movies = movies.merge(credits,on='title')
movies = movies[['id','title','overview','genres','keywords','cast','crew']]
movies.dropna(inplace=True)
#movies.duplicated().sum()
#print(movies.iloc[0].genres)
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
movies['genres'] = movies['genres'].apply(convert)
#print(movies.iloc[0].genres)
#print(movies.iloc[0].keywords)
movies['keywords'] = movies['keywords'].apply(convert)
#print(movies.iloc[0].keywords)
def convert3(obj):
    L = []
    c=0
    for i in ast.literal_eval(obj):
        if(c!=3):
            L.append(i['name'])
            c+=1
        else:
            break
    return L
movies['cast'] = movies['cast'].apply(convert3)
#print(movies['cast'])
def fetchDirector(obj):
    L = []
    for i in ast.literal_eval(obj):
        if(i['job']=='Director'):
            L.append(i['name'])
            break
    return L
movies['crew'] = movies['crew'].apply(fetchDirector)
#print(movies['crew'])
movies['overview'] = movies['overview'].apply(lambda x:x.split())
#print(movies['overview'][0])
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
#print(movies.head())
movies['tags'] = movies['overview'] + movies['genres'] + movies['cast'] + movies['crew']
#print(movies.iloc[0]['tags'])
new_df = movies[['id','title','tags']]
#print(new_df.head())
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] =  new_df['tags'].apply(lambda x:x.lower())
#print(new_df['tags'][0])
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
ps = PorterStemmer()
#print(ps.stem('called'))
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
new_df['tags'] = new_df['tags'].apply(stem)
#cosine_similarity(vectors)
similarity = cosine_similarity(vectors)
import pickle
f=open('file.pkl', 'wb+')
pickle.dump([new_df,similarity],f)
f.close()
#print(similarity[0])
'''
from pickle import load
f=open('file.pkl','rb')
new_df,similarity=load(f)
f.close()
def recommend(movie):
    #find the index of the movies
    try:
     movie_index = new_df[new_df['title']==movie].index[0]
     distances = similarity[movie_index]
     movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:10]
    #to fetch movies from indices
     for i in movies_list:
      print('  '+new_df.iloc[i[0]].title)
    except(IndexError):
    		print('     No movie found in the database')
    except:
    	print('Another error occured')
c=input('  enter movie name\n')
while (c!='exit'):
 recommend(c)
 c=input('\n  enter movie name \n')
