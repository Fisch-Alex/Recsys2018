import csv
import os
import pandas as pd

os.chdir(os.path.expanduser('~/Spotify'))

def Get_Playlist(line):
	return([int(song) for song in line])

def Load_Playlist_file(name):
	with open("data/Test/playlist_"+name+".csv") as tsv:
		Playlist = list(map(Get_Playlist,csv.reader(tsv, delimiter=',')))
		[playlist for playlist in Playlist if len(playlist) > 0 ]
	return(Playlist)

def Load_Meta(name):
	return(pd.read_csv("data/Test/meta_"+name+".csv",encoding = "utf8"))



