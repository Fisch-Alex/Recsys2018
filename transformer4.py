#### Combines the unique artists for each of the 1k playlists into one big dataframe Artist-Number.csv

import json
import numpy as np
import pandas as pd

mylist = []

for ii in range(1000):
	tmp = pd.read_csv("Artist-Number"+str(ii)+".csv")
	tmp2 = list(tmp["artist_id"])
	mylist = list(set(mylist + tmp2))
	print(ii)
	print(len(mylist))

d     = {"artist_id": mylist , "our_id": range(len(mylist))}
final = pd.DataFrame(d) 
final.to_csv("Artist-Number.csv", index=False)