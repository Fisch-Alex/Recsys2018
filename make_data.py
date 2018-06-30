import numpy as np 
import pandas as pd
import pickle
import csv
from collections import Counter

import os
os.chdir(os.path.expanduser('~/Spotify'))


#load data
Playlist_info=pd.read_csv("data/playlists_info.csv", encoding = "ISO-8859-1")
Track_info=pd.read_csv("data/track_info.csv", encoding = "ISO-8859-1")

Playlists=[]
for i in range(0,1000):
	with open('data/Playlists/playlists'+str(i)+'.csv', 'r') as f:
		reader = csv.reader(f)
		Playlists=Playlists+list(reader)

Playlists = [x for x in Playlists if x != []]    






#map album to songs in album
Album_to_songs = {album:Track_info[Track_info["album_id"]==album]["track_id"].values.tolist() for album in Track_info["album_id"].unique()[:1000]}
#map artist to songs by artist
Artist_to_songs = {artist:Track_info[Track_info["artist_id"]==artist]["track_id"].values.tolist() for artist in Track_info["artist_id"].unique()[:1000]}


#look at track frequencies
tracks=[track for playlist in Playlists for track in playlist]
frequencies=Counter(tracks)
mostfrequent=frequencies.most_common()
mostfrequent=[x[0] for x in mostfrequent]


#save dataset
#save as pickle
with open('data/Album_to_songs', 'wb') as handle:
    pickle.dump(Album_to_song_dict,handle)
    
 
with open('data/Artist_to_songs', 'wb') as handle:
    pickle.dump(Artist_to_song_dict,handle)

with open('data/most_frequent_tracks', 'wb') as handle:
    pickle.dump(mostfrequent,handle)
