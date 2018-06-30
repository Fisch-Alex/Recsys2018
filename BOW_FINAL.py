
# coding: utf-8

# In[3]:


from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
import string
import numpy as np 
import pandas as pd
import math
import os 
import csv
from random import sample


# In[ ]:


#helper functions

#function to clean text
def cleaner(text):
    #make look up table with punctuation in
    table = str.maketrans({key: None for key in """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""})
    #remove punctuation and capitals
    text=[i.lower().translate(table) for i in text]
    #remove stop words
    forbidden_words = set(["the"])
    return [' '.join([word for word in i.split() if word not in forbidden_words]) for i in text]

def text_collecter(Subsampled_data_frame,playlist_info,track_info_dict):
    # playlist names 
    playlist_names=cleaner([playlist_info["name"][ii] for ii in Subsampled_data_frame["PID"]])
    # prediction artist names
    prediction_artist_names=cleaner([track_info_dict[ii]["artist_name"] for ii in Subsampled_data_frame["TO_PREDICT"]])
    # prediction album names
    prediction_album_names=cleaner([track_info_dict[ii]["album_name"] for ii in Subsampled_data_frame["TO_PREDICT"]])
    # prediction track names
    prediction_track_names=cleaner([track_info_dict[ii]["track_name"] for ii in Subsampled_data_frame["TO_PREDICT"]])
    #train artist names squished together
    #as they are squished never do ngrams>1 
    train_artist_names=cleaner([" ".join([track_info_dict[jj]["artist_name"] for jj in ii]) for ii in Subsampled_data_frame["PLAYLISTS"]])
    train_album_names=cleaner([" ".join([track_info_dict[jj]["album_name"] for jj in ii]) for ii in Subsampled_data_frame["PLAYLISTS"]])
    train_track_names=cleaner([" ".join([track_info_dict[jj]["track_name"] for jj in ii]) for ii in Subsampled_data_frame["PLAYLISTS"]])
    return([playlist_names,prediction_artist_names,prediction_album_names,prediction_track_names,train_artist_names,train_album_names,train_track_names])
    
def Find(x,y):
    return(list(set(x.split()).intersection(set(y.split()))))

def Find_shared_Words(a,b):
    return([' '.join(x) for x in list(map(Find,a,b))])    


# In[ ]:


#make big bag of words
#returns large sparse array and the tv objects that are needed for predictions

def BOW(Subsampled_data_frame,playlist_info,playlists,track_info_dict,Training,vectorizers,challenge):

    #hard code parameters for now
    MAX_FEATURES_PLAYLIST_NAMES=1000
    MAX_FEATURES_PREDICTION_ARTIST_NAMES=1000
    MAX_FEATURES_PREDICTION_ALBUM_NAMES=1000
    MAX_FEATURES_PREDICTION_TRACK_NAMES=1000
    MAX_FEATURES_TRAIN_ARTIST_NAMES=1000
    MAX_FEATURES_TRAIN_ALBUM_NAMES=1000
    MAX_FEATURES_TRAIN_TRACK_NAMES=1000
    MIN_COUNTS=10
    
    MAX_FEATURES_SHARED_TRACK=1000
    MAX_FEATURES_SHARED_ALBUM=1000
    MAX_FEATURES_SHARED_ARTIST=1000
    MIN_COUNTS_SHARED=1
    
    #collect clean text not shared
    [playlist_names,prediction_artist_names,prediction_album_names,prediction_track_names,train_artist_names,train_album_names,train_track_names]=text_collecter(Subsampled_data_frame,playlist_info,track_info_dict)
    
    #collect clean text shared
    #only looking at shared between playlist name and predictions for now
    if challenge not in [2,5]:
        shared_track_names   = Find_shared_Words(playlist_names,prediction_artist_names)
        shared_album_names   = Find_shared_Words(playlist_names,prediction_album_names)
        shared_artist_names  = Find_shared_Words(playlist_names,prediction_track_names)

    tv1 = vectorizers[0]
    tv2 = vectorizers[1]
    tv3 = vectorizers[2]
    tv4 = vectorizers[3]
    tv5 = vectorizers[4]
    tv6 = vectorizers[5]
    tv7 = vectorizers[6]
    tv8 = vectorizers[7]
    tv9  = vectorizers[8]
    tv10 = vectorizers[9]
    

    if Training:
        #transform
        if challenge not in [2,5]:
            playlist_names_tf_idf=tv1.fit_transform(playlist_names)
            print("done1")
        prediction_artist_names_tf_idf=tv2.fit_transform(prediction_artist_names)
        print("done2")
        prediction_album_names_tf_idf=tv3.fit_transform(prediction_album_names)
        print("done3")
        prediction_track_names_tf_idf=tv4.fit_transform(prediction_track_names)
        print("done4")

        if challenge not in [1,7,9]:
            tv5.fit_transform(train_artist_names[:min(1000000,len(train_artist_names))])
            print("done5")
            train_artist_names_tf_idf=tv5.transform(train_artist_names)
            print("done5")
            tv6.fit_transform(train_album_names[:min(1000000,len(train_artist_names))])
            print("done6")
            train_album_names_tf_idf=tv6.transform(train_album_names)
            print("done6")
            tv7.fit_transform(train_track_names[:min(1000000,len(train_artist_names))])
            print("done7")
            train_track_names_tf_idf=tv7.transform(train_track_names)
            
        if challenge not in [2,5]:
            shared_track_names_tf_idf=tv8.fit_transform(shared_track_names)
            print("done8")
            shared_album_names_tf_idf=tv9.fit_transform(shared_album_names)
            print("done9")
            shared_artist_names_tf_idf=tv10.fit_transform(shared_artist_names)
            print("done10")

        vectorizers = [tv1,tv2,tv3,tv4,tv5,tv6,tv7,tv8,tv9,tv10]

    else:
        #transform
        if challenge not in [2,5]:
            playlist_names_tf_idf=tv1.transform(playlist_names)
            
        prediction_artist_names_tf_idf=tv2.transform(prediction_artist_names)
        prediction_album_names_tf_idf=tv3.transform(prediction_album_names)
        prediction_track_names_tf_idf=tv4.transform(prediction_track_names)

        if challenge not in [1,7,9]:
            train_artist_names_tf_idf=tv5.transform(train_artist_names)
            train_album_names_tf_idf=tv6.transform(train_album_names)
            train_track_names_tf_idf=tv7.transform(train_track_names)

        if challenge not in [2,5]:
            shared_track_names_tf_idf=tv8.transform(shared_track_names)
            shared_album_names_tf_idf=tv9.transform(shared_album_names)
            shared_artist_names_tf_idf=tv10.transform(shared_artist_names)
    
    #merge into a mega spase array

    if challenge in [3,4,6,8,10]:  
        sparse_merge = hstack((playlist_names_tf_idf,prediction_artist_names_tf_idf,prediction_album_names_tf_idf,prediction_track_names_tf_idf,train_artist_names_tf_idf,train_album_names_tf_idf,train_track_names_tf_idf,shared_track_names_tf_idf,shared_album_names_tf_idf,shared_artist_names_tf_idf)).tocsr()
    if challenge in [1,7,9]:  
        sparse_merge = hstack((playlist_names_tf_idf,prediction_artist_names_tf_idf,prediction_album_names_tf_idf,prediction_track_names_tf_idf,shared_track_names_tf_idf,shared_album_names_tf_idf,shared_artist_names_tf_idf)).tocsr()
    if challenge in [2,5]:  
        sparse_merge = hstack((prediction_artist_names_tf_idf,prediction_album_names_tf_idf,prediction_track_names_tf_idf,train_artist_names_tf_idf,train_album_names_tf_idf,train_track_names_tf_idf)).tocsr()

    if Training:
        return(sparse_merge,vectorizers)
    else:
        return(sparse_merge)

