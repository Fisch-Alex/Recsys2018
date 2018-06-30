from random import sample
import math
import numpy as np
import random
from sklearn.utils import shuffle
from itertools import islice
from random import randrange
import pandas as pd
import pickle
import csv

import os
os.chdir(os.path.expanduser('~/Spotify'))

#map album to songs in album
#map artist to songs by artist
with open('data/Album_to_songs', 'rb') as handle:
    Album_to_song_dict = pickle.load(handle)
with open('data/Artist_to_songs', 'rb') as handle:
    Artist_to_song_dict = pickle.load(handle)
#load list of most frequent
with open('data/most_frequent_tracks', 'rb') as handle:
    Most_frequent = pickle.load(handle)

    
#put into lists for fast access
Album_to_song_list=list(Album_to_song_dict.values())
Artist_to_song_list=list(Artist_to_song_dict.values())

#helper function to split list of lists into data/response
def splitter(list):
    return [list[:-1],list[-1]]
    
#helper function to draw a T tracks at random from a number of most frequent and all other tracks by same album/artist in that playlsit that are not in given playlist
def draw_random(Full_playlist,Chosen_training_tracks,T,num_most_frequent):
    #draw T tracks at random from most frequent and all other tracks by same artist/album as training tracks
    #does not sample from rest of that playlist
    if (T==0):
        return([])
    tracks=[]
    #tracks_of_same_albums=[Album_to_song_list[sorted_track_info.iloc[int(track)]["album_id"]] for track in Playlists[0]]
    #tracks_of_same_artists=[Artist_to_song_list[sorted_track_info.iloc[int(track)]["artist_id"]] for track in Playlists[0]]
    domain=Most_frequent[:num_most_frequent]#+[item for sublist in tracks_of_same_artists for item in sublist]+[item for sublist in tracks_of_same_albums for item in sublist]
    while len(tracks)<T:
        index=randrange(0,len(domain),1)
        ID=domain[index]
        if ID not in set(Full_playlist):
            tracks.append(int(ID))
    return(tracks)




#function to draw samples from our population dataset
def super_duper_sampler_ordered(data,seed,p,n,t,num_most_frequent,temporal=False):
    #INPUTS:
    #Playlists: data
    #seed: random seed
    #p: number of positive samples (iterating through playlists, so if 2,000,000 then 2 from each)
    #n: number of negatives to go with each positive sample
    #t: number of tunes in the playlists chosen to base predictions off
    #num_most_frequent: number of most frequent tunes to sample from
    #OUTPUTS:
    #positives: positive training examples the of form [[X_ID_1,Y_ID_1],[X_ID_2,Y_ID_2],...]
    # e.g [[["1","2","3"],"4"],[["5","6","7"],"8"],...]
    #negatives: negative training examples of the from [[X_ID_1,Y_ID_1],[X_ID_2,Y_ID_2],...]
    #set seeds
    np.random.seed(seed)
    random.seed(seed)

    #add in index to lists of playlists to keep for later
    reformat_data=[ [i]+data[i] for i in range(0,len(data))]
    
    #load list of most frequent
    
    #load some pre computed dictionaries
    #with open('./Spotify/data/Album_to_songs', 'rb') as handle:
        #Album_to_song_dict = pickle.load(handle)
    #with open('./Spotify/data/Artist_to_songs', 'rb') as handle:
        #Artist_to_song_dict = pickle.load(handle)
        
    #turn to lists and order for computational speedup   
        
    Album_to_song_list=[x for x in Album_to_song_dict.items()]
    Artist_to_song_list=[x for x in Artist_to_song_dict.items()]
    #sorted_track_info=track_info.sort_values("track_id")

    
    
    #sort track info for fast access
    
    #sorted_track_info=Track_info.sort_values("track_id")
    
    
    #make generator to select playlists that are long enough. Keep passing though playistis if need to collect more than one pass
    # t+2 because have PID as first element
    sampled_playlists=[]
    filtered= (x for x in reformat_data if len(x)>=(t+2))
    sampled_playlists=sampled_playlists + list(islice(filtered,p))
    while len(sampled_playlists)<p:
        filtered= (x for x in reformat_data if len(x)>=(t+2))
        #keep passing through data until got enough positives
        sampled_playlists = sampled_playlists +list(islice(filtered,p-len(sampled_playlists)))
    
    #if ordering important
    if temporal:
        #save positives. Randomly select a block of t+1 consecutive songs in the playlist (ignoring the first element which is PID)we use
        # still random because could select a different consecutive block if playlist big enough
        positives=[[x[0]]+splitter(x[random.randint(1,len(x)-1-t):][:t+1]) for x in sampled_playlists]
        
        #save negatives. shuffle to see which t playlists are in X again ignoring first element
        #choose training to be a consecutive block
        raw_negatives=[[positives[i][0]]+[positives[i][1],draw_random(sampled_playlists[i][1:],positives[i][1],n,num_most_frequent)] for i in range(0,p)]
        
        #reformat
        raw_negatives_2=[[[x,y,a] for a in z] for x,y,z in raw_negatives]
        negatives= [item for sublist in raw_negatives_2 for item in sublist]
    
    #if ordering not important
    else:
        #save positives. Randomly select t+1 songs in the playlist (ignoring the first element which is PID)we use
        positives=[[x[0]]+splitter(sample(x[1:],t+1)) for x in sampled_playlists]
        #save negatives. shuffle to see which t playlists are in X again ignoring first element
        #choose training to be a consecutive block
        raw_negatives=[[positives[i][0]]+[positives[i][1],draw_random(sampled_playlists[i][1:],positives[i][1],n,num_most_frequent)] for i in range(0,p)]
        
        #reformat
        raw_negatives_2=[[[x,y,a] for a in z] for x,y,z in raw_negatives]
        negatives= [item for sublist in raw_negatives_2 for item in sublist]        
    
    #make df as requested by alex
    data_pos=pd.DataFrame(positives,columns=["PID","PLAYLISTS","TO_PREDICT"])
    data_pos["RESPONSE"]=1
    data_neg=pd.DataFrame(negatives,columns=["PID","PLAYLISTS","TO_PREDICT"])
    data_neg["RESPONSE"]=0
    final=pd.concat([data_pos,data_neg])
    del data_pos
    del data_neg
    
    return(final)

