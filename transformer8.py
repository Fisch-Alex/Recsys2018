#### Creates a data frame which contains meta information about each playlist

import json
import numpy as np
import csv
import pandas as pd

hi  = range(1,999002,1000)
hi2 = range(1000,1000001,1000)

names = ["mpd.slice."+str(hi[ii]-1)+"-"+str(hi2[ii]-1)+".json" for ii in range(1000)]

albums  = pd.read_csv("Album-Number.csv",nrows=100)
artists = pd.read_csv("Artist-Number.csv",nrows=100)
tracks  = pd.read_csv("Track-Number.csv",nrows=100)

album_dict  = dict(zip(albums["album_id"],albums["our_id"]))
artist_dict = dict(zip(artists["artist_id"],artists["our_id"]))
track_dict  = dict(zip(tracks["track_id"],tracks["our_id"]))
	

def combinedata(ii):
	with open('data/' + names[ii]) as mydata:
		fulldata  = json.load(mydata)
		playlists = fulldata['playlists']

		num_edits     = [x['num_edits']   for x in playlists]
		duration_ms   = [x['duration_ms'] for x in playlists]
		num_tracks    = [x['num_tracks']  for x in playlists]
		name          = [x['name']  for x in playlists]
		modified_at   = [x['modified_at']  for x in playlists]
		num_albums    = [x['num_albums']  for x in playlists]
		num_artists   = [x['num_artists']  for x in playlists]
		num_followers = [x['num_followers']  for x in playlists]
		collaborative = [x['collaborative']  for x in playlists]

		tmp = {

		"num_edits"  :  num_edits,
		"duration_ms" : duration_ms,
		"num_tracks"   : num_tracks,
		"name" : name,    
		"modified_at" : modified_at,
		"num_albums"   : num_albums,
		"num_artists"   : num_artists,
		"num_followers" : num_followers,
		"collaborative" : collaborative,

		}

		return(tmp)

hello = list(map(combinedata,range(1000)))

num_edits     = [x for z in hello for x in z["num_edits"]]
duration_ms   = [x for z in hello for x in z["duration_ms"]]
num_tracks    = [x for z in hello for x in z["num_tracks"]]
name          = [x for z in hello for x in z["name"]]
num_albums    = [x for z in hello for x in z["num_albums"]]
modified_at   = [x for z in hello for x in z["modified_at"]]
num_artists   = [x for z in hello for x in z["num_artists"]]
num_followers = [x for z in hello for x in z["num_followers"]]
collaborative = [x for z in hello for x in z["collaborative"]]

d = {

		"num_edits"  :  num_edits,
		"duration_ms" : duration_ms,
		"num_tracks"   : num_tracks,
		"name" : name,    
		"modified_at" : modified_at,
		"num_albums"   : num_albums,
		"num_artists"   : num_artists,
		"num_followers" : num_followers,
		"collaborative" : collaborative,

		}

df = pd.DataFrame(data=d)

print(df)

df.to_csv("playlists_info.csv",index=False)




		