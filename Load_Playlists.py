import csv
import os

os.chdir(os.path.expanduser('~/Spotify'))

def Get_Playlist(line):
	return([int(song) for song in line])

def Load_Playlist_file(ii):
	with open("data/Playlists/playlists"+str(ii)+".csv") as tsv:
		Playlist = list(map(Get_Playlist,csv.reader(tsv, delimiter=',')))
	return(Playlist)

def Load_Playlists(n):
	list_of_playlist_lists = list(map(Load_Playlist_file,range(n)))
	return ([playlist for playlist_list in list_of_playlist_lists for playlist in playlist_list if len(playlist) > 0 ])

