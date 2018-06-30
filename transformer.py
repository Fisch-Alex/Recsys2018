#### extracts unique tracks for each of the 1k playlists

import json
import numpy as np
import pandas as pd

hi = range(1,999002,1000)

hi2 = range(1000,1000001,1000)

names = ["mpd.slice."+str(hi[ii]-1)+"-"+str(hi2[ii]-1)+".json" for ii in range(1000)]


def extracttracks(ii):
	with open('data/' + names[ii]) as mydata:
		fulldata  = json.load(mydata)
		playlists = fulldata['playlists']

		tracks = [x['track_uri'] for y in playlists for x in y["tracks"] ]

		tracks = list(set(tracks))
	return(tracks)


for ii in range(1000):
	tmp = extracttracks(ii)
	d   = {"track_id": tmp }
	final = pd.DataFrame(d) 
	final.to_csv("Track-Number"+str(ii)+".csv", index=False)