
import streamlit as st
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# loading the data from the csv file to a pandas dataframe
movies_data = pd.read_csv('movies.csv', on_bad_lines='skip', quoting=1)

# selecting the relevant features for recommendation
selected_features = ['genres','keywords','tagline','cast','director']

# replacing the null values with the string
for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')

# combining all the 5 selected features
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']

# converting the text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)

list_of_all_titles = movies_data['title'].tolist()

st.title('Movie Recommendation System')

movie_name = st.text_input('Enter your favourite movie name :')

if movie_name:
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if find_close_match:
        close_match = find_close_match[0]

        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

        similarity_score = list(enumerate(similarity[index_of_the_movie]))

        sorted_similar_movies = sorted(similarity_score,key = lambda x:x[1],reverse = True)

        st.write('Movies suggested for you:\n')

        i=1

        for movie in sorted_similar_movies:
          index = movie[0]
          title_from_index = movies_data[movies_data.index == index]['title'].values[0]
          if(i<10): # Display top 10 similar movies
            st.write(f"{i}. {title_from_index}")

          i+=1
    else:
        st.write("Movie not found in the database.")