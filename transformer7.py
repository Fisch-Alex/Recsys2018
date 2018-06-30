#### Creates new playlists containing only numbers

import json
import numpy as np
import csv
import pandas as pd

hi  = range(1,999002,1000)
hi2 = range(1000,1000001,1000)

names = ["mpd.slice."+str(hi[ii]-1)+"-"+str(hi2[ii]-1)+".json" for ii in range(1000)]

albums  = pd.read_csv("Album-Number.csv")
artists = pd.read_csv("Artist-Number.csv")
tracks  = pd.read_csv("Track-Number.csv")

album_dict  = dict(zip(albums["album_id"],albums["our_id"]))
artist_dict = dict(zip(artists["artist_id"],artists["our_id"]))
track_dict  = dict(zip(tracks["track_id"],tracks["our_id"]))


def transform_track(track):
	track_new = {'duration_ms': track['duration_ms'], 'artist_name': track['artist_name'], 
   'track_id': track_dict[track['track_uri']] , 'album_name': track['album_name'], 
   'track_name': track['track_name'], 'artist_id': artist_dict[track['artist_uri']] , 
   'album_id': album_dict[track['album_uri']]}
	return(track_new)


def transform_playlists(playlist):
	playlist_new = playlist
	playlist_new['tracks'] = list(map(transform_track,playlist['tracks']))
	return(playlist_new) 

def transform_playlists_final(playlist):
	playlist_new = [track_dict[track['track_uri']] for track in playlist['tracks'] ] 
	return(playlist_new)
	

for ii in range(1000):
	with open('data/' + names[ii]) as mydata:
		fulldata  = json.load(mydata)
		playlists = fulldata['playlists']

		#playlists_new = {"hi" : list(map(transform_playlists,playlists))}

		# print(track_dict[playlists[0]['tracks'][0]['track_uri']])


	tmp = [ [int(track_dict[track['track_uri']]) for track in playlist['tracks'] ] for  playlist in playlists ]

	with open("playlists"+str(ii)+".csv", "w") as f:
		writer = csv.writer(f)
		writer.writerows(tmp)

		