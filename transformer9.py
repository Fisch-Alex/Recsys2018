#### Creates a data frame which contains meta information about each track dor all 1000 playists

import json
import numpy as np
import csv
import pandas as pd
import pickle

hi  = range(1,999002,1000)
hi2 = range(1000,1000001,1000)

names = ["mpd.slice."+str(hi[ii]-1)+"-"+str(hi2[ii]-1)+".json" for ii in range(1000)]

albums  = pd.read_csv("Album-Number.csv")
artists = pd.read_csv("Artist-Number.csv")
tracks  = pd.read_csv("Track-Number.csv")

album_dict  = dict(zip(albums["album_id"],albums["our_id"]))
artist_dict = dict(zip(artists["artist_id"],artists["our_id"]))
track_dict  = dict(zip(tracks["track_id"],tracks["our_id"]))


for ii in range(1000):
	print(ii)
	with open('data/' + names[ii]) as mydata:
		fulldata  = json.load(mydata)
		playlists = fulldata['playlists']

		duration_ms = [str(track['duration_ms']) for playlist in playlists for track in playlist["tracks"]]
		track_name  = [str(track['track_name']) for playlist in playlists for track in playlist["tracks"]]
		album_name  = [str(track['album_name']) for playlist in playlists for track in playlist["tracks"]]
		artist_name = [str(track['artist_name']) for playlist in playlists for track in playlist["tracks"]]
		artist_id   = [str(artist_dict[track['artist_uri']]) for playlist in playlists for track in playlist["tracks"]]
		track_id    = [str(track_dict[track['track_uri']])  for playlist in playlists for track in playlist["tracks"]]
		album_id    = [str(album_dict[track['album_uri']])   for playlist in playlists for track in playlist["tracks"]]

		track_info = {str(track_id[jj]) : {'duration_ms' : duration_ms[jj] , 'track_name' : track_name[jj],  'album_name' : album_name[jj],  'artist_name' : artist_name[jj], 'artist_id' : artist_id[jj], 'album_id' : album_id[jj] } 
		for jj in range(len(track_id))}

		track_id = list(track_info.keys())

		duration_ms = [x['duration_ms'] for x in track_info.values()]
		track_name  = [x['track_name'] for x in track_info.values()]
		album_name  = [x['album_name'] for x in track_info.values()]
		artist_name = [x['artist_name'] for x in track_info.values()]
		artist_id   = [x['artist_id'] for x in track_info.values()]
		album_id    = [x['album_id'] for x in track_info.values()]

		d = {

		"track_id"  :  track_id,
		"duration_ms" : duration_ms,
		"track_name"   : track_name,
		"album_name" : album_name,    
		"artist_name" : artist_name,
		"artist_id"   : artist_id,
		"album_id"   : album_id

		}


		final = pd.DataFrame(d) 
		final.to_csv("track_info_list"+str(ii)+".csv", index=False)


#track_id = list(track_info.keys())

#duration_ms = [x['duration_ms'] for x in track_info.values()]
#track_name  = [x['track_name'] for x in track_info.values()]
#album_name  = [x['album_name'] for x in track_info.values()]
#artist_name = [x['artist_name'] for x in track_info.values()]
#artist_id   = [x['artist_id'] for x in track_info.values()]
#album_id    = [x['album_id'] for x in track_info.values()]

#d = {
#
#		"track_id"  :  track_id,
#		"duration_ms" : duration_ms,
#		"track_name"   : track_name,
#		"album_name" : album_name,    
#		"artist_name" : artist_name,
#		"artist_id"   : artist_id,
#		"album_id"   : album_id,
#
#		}





		