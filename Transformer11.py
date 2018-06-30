###  This file updates Track information and transforms the test data

import json
import numpy as np
import csv
import pandas as pd

import os

os.chdir(os.path.expanduser('~/Spotify'))

fulldata = pd.read_csv("data/track_info.csv", encoding = "ISO-8859-1")

print(fulldata)

conversions = pd.read_csv("data/Track-Number.csv")
id_dict     = {conversions["our_id"][ii]:conversions["track_id"][ii] for ii in range(conversions.shape[0])}
id_dict_inv = {conversions["track_id"][ii]:conversions["our_id"][ii] for ii in range(conversions.shape[0])}

print("done")

conversions     = pd.read_csv("data/Artist-Number.csv")
artist_dict     = {conversions["our_id"][ii]:conversions["artist_id"][ii] for ii in range(conversions.shape[0])}
artist_dict_inv = {conversions["artist_id"][ii]:conversions["our_id"][ii] for ii in range(conversions.shape[0])}

print("done")

conversions    = pd.read_csv("data/Album-Number.csv")
album_dict     = {conversions["our_id"][ii]:conversions["album_id"][ii] for ii in range(conversions.shape[0])}
album_dict_inv = {conversions["album_id"][ii]:conversions["our_id"][ii] for ii in range(conversions.shape[0])}

print("done")

existing_albums  = [album_dict[x]  for x in fulldata["album_id"]]
existing_tracks  = [id_dict[x]     for x in fulldata["track_id"]]
existing_artists = [artist_dict[x] for x in fulldata["artist_id"]]

print("done")

num_existing_albums  = len(existing_albums)
num_existing_tracks  = len(existing_tracks)
num_existing_artists = len(existing_artists)

with open('data/challenge_set.json') as mydata:

	testdata  = json.load(mydata)


	#### Need to update dictionaries 

	artist_name_list = [song['artist_name'] for playlist in testdata['playlists'][1000:10000] for song in playlist['tracks']]
	track_name_list  = [song['track_name']  for playlist in testdata['playlists'][1000:10000] for song in playlist['tracks']]
	album_name_list  = [song['artist_name'] for playlist in testdata['playlists'][1000:10000] for song in playlist['tracks']]
	duration_list    = [song['duration_ms'] for playlist in testdata['playlists'][1000:10000] for song in playlist['tracks']]
	track_uri_list   = [song['track_uri']   for playlist in testdata['playlists'][1000:10000] for song in playlist['tracks']]
	artist_uri_list  = [song['artist_uri']  for playlist in testdata['playlists'][1000:10000] for song in playlist['tracks']]
	album_uri_list   = [song['album_uri']   for playlist in testdata['playlists'][1000:10000] for song in playlist['tracks']]

	artist_set = set(artist_uri_list)
	album_set  = set(album_uri_list)
	track_set  = set(track_uri_list)

	track_duration_dict  = {track_uri_list[ii]:duration_list[ii]   for ii in range(len(track_uri_list))}
	track_album_set_dict = {track_uri_list[ii]:album_uri_list[ii]  for ii in range(len(track_uri_list))}
	track_artist_dict    = {track_uri_list[ii]:artist_uri_list[ii] for ii in range(len(track_uri_list))}

	new_artists = list(artist_set - set(existing_artists))
	new_tracks  = list(track_set  - set(existing_tracks))
	print(new_tracks)
	new_albums  = list(album_set  - set(existing_albums))

	new_artists_dict = {new_artists[ii]:(ii+num_existing_artists) for ii in range(len(new_artists))}
	new_tracks_dict  = {new_tracks[ii]:(ii+num_existing_tracks)   for ii in range(len(new_tracks))}
	new_albums_dict  = {new_albums[ii]:(ii+num_existing_albums)   for ii in range(len(new_albums))}

	album_dict_inv.update(new_albums_dict)
	print(len(id_dict_inv))
	id_dict_inv.update(new_tracks_dict)
	print(len(id_dict_inv))
	artist_dict_inv.update(new_artists_dict)


	####################################################################### 

	new_tracks_ids       = [new_tracks_dict[x] for x in new_tracks]
	new_tracks_duration  = [track_duration_dict[x] for x in new_tracks]
	new_tracks_album     = [track_album_set_dict[x] for x in new_tracks]
	new_tracks_album_id  = [full_albums_dict[track_album_set_dict[x]] for x in new_tracks]
	new_tracks_artist    = [track_artist_dict[x] for x in new_tracks]
	new_tracks_artist_id = [full_artists_dict[track_artist_dict[x]] for x in new_tracks]

	newdaf = pd.DataFrame({"track_name":new_tracks, "track_id":new_tracks_ids, "duration_ms":new_tracks_duration, "artist_name":new_tracks_artist, 
		"artist_id":new_tracks_artist_id, "album_name":new_tracks_album, "album_id":new_tracks_album_id})

	completedaf = pd.concat([newdaf,fulldata],ignore_index = True)

	completedaf.to_csv("data/track_full_info.csv", encoding = "utf8", index = False)

	# trackname_id_dictionary = {completedaf["track_name"][ii]:completedaf["track_id"][ii] for ii in range(completedaf.shape[0])}

	#### first 1k playlists are name only 

	pid        = [x['pid'] for x in testdata['playlists'][:1000]]
	num_tracks = [x['num_tracks'] for x in testdata['playlists'][:1000]]
	names      = [x['name'] for x in testdata['playlists'][:1000]]

	meta_name_only = pd.DataFrame({'pid' : pid, "num_tracks" : num_tracks, "name" : names})
	meta_name_only.to_csv("data/Test/meta_name_only.csv", encoding = "utf8", index = False)

	#### second 1k playlists are name and **first** 5 tracks

	pid        = [x['pid'] for x in testdata['playlists'][1000:2000]]
	num_tracks = [x['num_tracks'] for x in testdata['playlists'][1000:2000]]
	names      = [x['name'] for x in testdata['playlists'][1000:2000]]

	x = testdata['playlists'][1000]
	print(type(x))
	y = x["tracks"]
	print(type(y))
	print(y[0])
	print(y[0]["track_uri"])

	list_of_playlists = [ [ id_dict_inv[y["track_uri"]] for y in x["tracks"]] for x in testdata['playlists'][1000:2000]]


	with open("data/Test/playlist_name_first_five.csv", "w") as f:
		writer = csv.writer(f)
		writer.writerows(list_of_playlists)

	meta_name_first_five = pd.DataFrame({'pid' : pid, "num_tracks" : num_tracks, "name" : names})
	meta_name_first_five.to_csv("data/Test/meta_name_first_five.csv", encoding = "utf8", index = False)


	#### third 1k playlists are **first** 5 tracks and NO NAME

	pid        = [x['pid'] for x in testdata['playlists'][2000:3000]]
	num_tracks = [x['num_tracks'] for x in testdata['playlists'][2000:3000]]

	meta_random_five = pd.DataFrame({'pid' : pid, "num_tracks" : num_tracks})
	meta_random_five.to_csv("data/Test/meta_first_five.csv", encoding = "utf8", index = False)

	list_of_playlists = [ [ id_dict_inv[y["track_uri"]] for y in x["tracks"]] for x in testdata['playlists'][2000:3000]]

	with open("data/Test/playlist_first_five.csv", "w") as f:
		writer = csv.writer(f)
		writer.writerows(list_of_playlists)

	#### fourth 1k playlists are name and **first** 10 tracks  

	pid        = [x['pid'] for x in testdata['playlists'][3000:4000]]
	num_tracks = [x['num_tracks'] for x in testdata['playlists'][3000:4000]]
	names      = [x['name'] for x in testdata['playlists'][3000:4000]]

	meta_name_first_ten = pd.DataFrame({'pid' : pid, "num_tracks" : num_tracks, "name" : names})
	meta_name_first_ten.to_csv("data/Test/meta_name_first_ten.csv", encoding = "utf8", index = False)

	list_of_playlists = [ [ id_dict_inv[y["track_uri"]] for y in x["tracks"]] for x in testdata['playlists'][3000:4000]]


	with open("data/Test/playlist_name_first_ten.csv", "w") as f:
		writer = csv.writer(f)
		writer.writerows(list_of_playlists)

	#### fifth 1k playlists are first 10 tracks  

	pid        = [x['pid'] for x in testdata['playlists'][4000:5000]]
	num_tracks = [x['num_tracks'] for x in testdata['playlists'][4000:5000]]

	meta_first_ten = pd.DataFrame({'pid' : pid, "num_tracks" : num_tracks})
	meta_first_ten.to_csv("data/Test/meta_first_ten.csv", encoding = "utf8", index = False)

	list_of_playlists = [ [ id_dict_inv[y["track_uri"]] for y in x["tracks"]] for x in testdata['playlists'][4000:5000]]



	with open("data/Test/playlist_first_ten.csv", "w") as f:
		writer = csv.writer(f)
		writer.writerows(list_of_playlists)

	#### sixth 1k playlists are first 25 tracks and name 	

	pid        = [x['pid'] for x in testdata['playlists'][5000:6000]]
	num_tracks = [x['num_tracks'] for x in testdata['playlists'][5000:6000]]
	names      = [x['name'] for x in testdata['playlists'][5000:6000]]

	meta_name_first_twentyfive = pd.DataFrame({'pid' : pid, "num_tracks" : num_tracks, "name" : names})
	meta_name_first_twentyfive.to_csv("data/Test/meta_name_first_twentyfive.csv", encoding = "utf8", index = False)

	list_of_playlists = [ [ id_dict_inv[y["track_uri"]] for y in x["tracks"]] for x in testdata['playlists'][5000:6000]]


	with open("data/Test/playlist_name_first_twentyfive.csv", "w") as f:
		writer = csv.writer(f)
		writer.writerows(list_of_playlists)

	#### seventh 1k playlists are 25 tracks and name 	

	pid        = [x['pid'] for x in testdata['playlists'][6000:7000]]
	num_tracks = [x['num_tracks'] for x in testdata['playlists'][6000:7000]]
	names      = [x['name'] for x in testdata['playlists'][6000:7000]]

	meta_name_twentyfive = pd.DataFrame({'pid' : pid, "num_tracks" : num_tracks, "name" : names})
	meta_name_twentyfive.to_csv("data/Test/meta_name_twentyfive.csv", encoding = "utf8", index = False)

	list_of_playlists = [ [ id_dict_inv[y["track_uri"]] for y in x["tracks"]] for x in testdata['playlists'][6000:7000]]


	with open("data/Test/playlist_name_twentyfive.csv", "w") as f:
		writer = csv.writer(f)
		writer.writerows(list_of_playlists)

	#### eighth 1k playlists are first 100 tracks and name 	

	pid        = [x['pid'] for x in testdata['playlists'][7000:8000]]
	num_tracks = [x['num_tracks'] for x in testdata['playlists'][7000:8000]]
	names      = [x['name'] for x in testdata['playlists'][7000:8000]]

	meta_name_first_hundred = pd.DataFrame({'pid' : pid, "num_tracks" : num_tracks, "name" : names})
	meta_name_first_hundred.to_csv("data/Test/meta_name_first_hundred.csv", encoding = "utf8", index = False)

	list_of_playlists = [ [ id_dict_inv[y["track_uri"]] for y in x["tracks"]] for x in testdata['playlists'][7000:8000]]


	with open("data/Test/playlist_name_first_hundred.csv", "w") as f:
		writer = csv.writer(f)
		writer.writerows(list_of_playlists)

	#### ninth 1k playlists are 100 tracks and name 	

	pid        = [x['pid'] for x in testdata['playlists'][8000:9000]]
	num_tracks = [x['num_tracks'] for x in testdata['playlists'][8000:9000]]
	names      = [x['name'] for x in testdata['playlists'][8000:9000]]

	meta_name_hundred = pd.DataFrame({'pid' : pid, "num_tracks" : num_tracks, "name" : names})
	meta_name_hundred.to_csv("data/Test/meta_name_hundred.csv", encoding = "utf8", index = False)

	list_of_playlists = [ [ id_dict_inv[y["track_uri"]] for y in x["tracks"]] for x in testdata['playlists'][8000:9000]]

	with open("data/Test/playlist_name_hundred.csv", "w") as f:
		writer = csv.writer(f)
		writer.writerows(list_of_playlists)

	#### tenth 1k playlists are first tracks and name 	

	pid        = [x['pid'] for x in testdata['playlists'][9000:10000]]
	num_tracks = [x['num_tracks'] for x in testdata['playlists'][9000:10000]]
	names      = [x['name'] for x in testdata['playlists'][9000:10000]]

	meta_name_first = pd.DataFrame({'pid' : pid, "num_tracks" : num_tracks, "name" : names})
	meta_name_first.to_csv("data/Test/meta_name_first.csv", encoding = "utf8", index = False)

	list_of_playlists = [ [ id_dict_inv[y["track_uri"]] for y in x["tracks"]] for x in testdata['playlists'][9000:10000]]

	with open("data/Test/playlist_name_first.csv", "w") as f:
		writer = csv.writer(f)
		writer.writerows(list_of_playlists)