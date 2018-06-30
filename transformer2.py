#### Combines the unique tracks for each of the 1k playlists into one big dataframe Track-Number.csv

import json
import numpy as np
import pandas as pd

mylist = []

for ii in range(1000):
	tmp = pd.read_csv("Track-Number"+str(ii)+".csv")
	tmp2 = list(tmp["track_id"])
	mylist = list(set(mylist + tmp2))
	print(ii)
	print(len(mylist))

d     = {"track_id": mylist , "our_id": range(len(mylist))}
final = pd.DataFrame(d) 
final.to_csv("Track-Number.csv", index=False)