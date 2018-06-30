#### Combines the unique albums for each of the 1k playlists into one big dataframe Album-Number.csv

import json
import numpy as np
import pandas as pd

mylist = []

for ii in range(1000):
	tmp = pd.read_csv("Album-Number"+str(ii)+".csv")
	tmp2 = list(tmp["album_id"])
	mylist = list(set(mylist + tmp2))
	print(ii)
	print(len(mylist))

d     = {"album_id": mylist , "our_id": range(len(mylist))}
final = pd.DataFrame(d) 
final.to_csv("Album-Number.csv", index=False)