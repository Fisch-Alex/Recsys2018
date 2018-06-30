import csv
import pandas as pd
import numpy as np 
import time

import os
os.chdir(os.path.expanduser('~/Spotify'))

NAME = "name_first_hundred"

from lgb_model10_exp import LGB_Model10 as model
from Generate_features_scenario10 import Generate_features_scenario10 as Generate_features
from NEW_SAMPLER_album_artist_fixed import get_predictors, track_info
from Load_Playlists import Load_Playlists 
from Load_Testdata import Load_Playlist_file, Load_Meta
from multiprocessing import Pool

DEBUG = False 

if (DEBUG):

	playlists = Load_Playlists(100)
	playlist_info = pd.read_csv("data/playlists_info.csv",nrows = 100000,encoding = "latin-1")
	N = 2000
	rounds = 60

else:

	playlists = Load_Playlists(1000)
	playlist_info = pd.read_csv("data/playlists_info.csv",encoding = "latin-1")
	N = 5000000
	rounds = 12000
	
playlist_info["name"] = playlist_info["name"].astype(str) 

track_info["track_id"]    = track_info["track_id"].astype(int)
track_info["album_id"]    = track_info["album_id"].astype(int) 
track_info["artist_id"]   = track_info["artist_id"].astype(int) 
track_info["album_name"]  = track_info["album_name"].astype(str) 
track_info["track_name"]  = track_info["track_name"].astype(str)
track_info["artist_name"] = track_info["artist_name"].astype(str)

track_info_dict = track_info.set_index('track_id').to_dict('index')

params = {
		'boosting_type' : 'gbdt',
        'learning_rate': 0.1,
        'application': 'binary',
        'max_depth': 17,
        'num_leaves': 150,
        'metric': 'binary_logloss',
        "lambda_l1": 1,
        'nthread': 25,
        'verbose' : -1
    }

print("hi1")
start_time = time.time()
ourmodel = model(params,10000,playlists)
print('[{}] Model initiated.'.format(time.time() - start_time))
print("hi2")
ourmodel.fit(playlist_info,playlists,track_info_dict,num_boost_round=rounds,early_stopping_rounds=200,number_of_samples=N,FPs=3)
print('[{}] Model fitted.'.format(time.time() - start_time))
print("hi3")
test_playlists     = Load_Playlist_file(NAME)
test_playlist_info = Load_Meta(NAME)
test_playlist_info.reset_index()
print("hi4")

def Get_Features(ii): 

	playlist   = test_playlists[ii]

	suggestions = list(set(get_predictors(playlist) + ourmodel.list_to_try))

	tobetested = [x for x in suggestions if x not in playlist]
	info       = pd.concat([test_playlist_info.loc[ii,:].to_frame().T]*len(tobetested), ignore_index=True )

	testdata   = pd.DataFrame({"TO_PREDICT":tobetested})
	testdata["PLAYLISTS"] = [playlist for ii in range(len(tobetested))]
	testdata["PID"]       = 0 

	info["num_tracks"] = info["num_tracks"].astype(int)
	info["pid"] = info["pid"].astype(int)
	info["name"] = info["name"].astype(str)

	X_test  = Generate_features(testdata,info,playlists,track_info_dict,ourmodel.list_to_try,ourmodel.CV_list_of_common_tracks ,ourmodel.CV_list_of_common_albums,ourmodel.CV_list_of_common_artists,ourmodel.vectorizers,False)

	return([X_test,testdata["TO_PREDICT"]])

def predict_one(pair):

	outputs = ourmodel.model.predict(pair[0])*(-1)
	indices = np.argsort(outputs)
	return([pair[1][ii] for ii in indices[:500]])

p = Pool(20)
test_sets       = list(p.map(Get_Features,range(0,250)))
ourpredictions1 = list(map(predict_one,test_sets))
test_sets       = list(p.map(Get_Features,range(250,500)))
ourpredictions2 = list(map(predict_one,test_sets))
test_sets       = list(p.map(Get_Features,range(500,750)))
ourpredictions3 = list(map(predict_one,test_sets))
test_sets       = list(p.map(Get_Features,range(750,1000)))
ourpredictions4 = list(map(predict_one,test_sets))
p.close()

ourpredictions = ourpredictions1 + ourpredictions2 + ourpredictions3 + ourpredictions4

print("hi5")

with open("Submissions_exp/challenge10_RAW_NEW_LARGE.csv", "w") as f:
	writer = csv.writer(f)
	writer.writerows(ourpredictions)



