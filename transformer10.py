#### Assembles the metainformation for all 1000 playists

import json
import numpy as np
import csv
import pandas as pd
import pickle



final = dict()
ii = 149
hello = pd.read_csv("track_info_list"+str(ii)+".csv", encoding = "ISO-8859-1")


for ii in range(1000):

	metadata = pd.read_csv("track_info_list"+str(ii)+".csv", encoding = "ISO-8859-1")

	duration_ms = list(metadata['duration_ms'])
	track_name  = list(metadata['track_name'])
	album_name  = list(metadata['album_name']) 
	artist_name = list(metadata['artist_name'])
	artist_id   = list(metadata['artist_id'])
	track_id    = list(metadata['track_id'])  
	album_id    = list(metadata['album_id']) 

	track_info = {str(track_id[jj]) : {'duration_ms' : duration_ms[jj] , 'track_name' : track_name[jj],  'album_name' : album_name[jj],  'artist_name' : artist_name[jj], 'artist_id' : artist_id[jj], 'album_id' : album_id[jj] } 
	for jj in range(len(track_id))}

	final.update(track_info)
	print(ii)
	print(len(final))


track_id    = list(final.keys())
duration_ms = [x['duration_ms'] for x in final.values()]
track_name  = [x['track_name'] for x in final.values()]
album_name  = [x['album_name'] for x in final.values()]
artist_name = [x['artist_name'] for x in final.values()]
artist_id   = [x['artist_id'] for x in final.values()]
album_id    = [x['album_id'] for x in final.values()]

d = {

		"track_id"  :  track_id,
		"duration_ms" : duration_ms,
		"track_name"   : track_name,
		"album_name" : album_name,    
		"artist_name" : artist_name,
		"artist_id"   : artist_id,
		"album_id"   : album_id

}

df = pd.DataFrame(data=d)

print(df.shape)

df.to_csv("track_info.csv",index=False)



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





		