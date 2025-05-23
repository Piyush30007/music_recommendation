# -*- coding: utf-8 -*-
"""
Created on Fri May 23 09:17:15 2025

@author: Piyush Singh
"""

import pickle
import streamlit as st
import pandas as pd
import difflib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data():
    music_data = pd.read_csv("spotify_sample.csv")
    music_data = music_data.reset_index(drop=True)
    return music_data

def recommend_music(music_name , music_data):
   music_name = music_name.lower()
   list_of_all_title = [title.lower() for title in music_data['song'].tolist()]

   title_column = music_data['song'].str.lower().tolist()
   find_close_match = difflib.get_close_matches(music_name,list_of_all_title,n=1,cutoff=0.6)
   if not find_close_match:
      return None, []

   close_match = find_close_match[0]

   vectorizer = TfidfVectorizer()
   matrix = vectorizer.fit_transform(music_data['song']) 
   similarity  = cosine_similarity(matrix, matrix)
   index_of_the_music = music_data[music_data['song'].str.lower() == close_match].index.values[0]
   similarity_score = list(enumerate(cosine_similarity(matrix[index_of_the_music], matrix).flatten()))
   sorted_similar_musics = sorted(similarity_score , key = lambda x:x[1],reverse=True)
   recommendations = []
   i= 1
   for music in sorted_similar_musics:
    index = music[0]

    if(i<30):
        title_of_the_music = music_data[music_data.index==index]['song'].values[0]
        recommendations.append(title_of_the_music)
        i+=1
    return close_match , recommendations
         
        
st.title("Music Recommendation")
music_data = load_data()
music_name = st.text_input("Enter Music Name")

if st.button("Get Recoomendations"):
    if music_name.strip() == "" :
        st.warning("Please First Enter The Music Name ")
        
    else :
        close_match,recommendations = recommend_music(music_name , music_data)
        if close_match:
            st.success(f"Top recommendations for: {close_match.title()}")
            for i, song in enumerate(recommendations, 1):
                st.write(f"{i}. {song}")
        else:
            st.error("No close match found. Try again with a different song name.")