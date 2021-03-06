import os
os.chdir(os.path.expanduser('~/Spotify'))

##### 

import lightgbm as lgb
import numpy as np
import pandas as pd
from Most_Frequent_Model import Most_Frequent_Model 
from sklearn.model_selection import train_test_split

##### 

from Load_Playlists import Load_Playlists
from Generate_features_scenario1 import Generate_features_scenario1 as Generate_features
from NEW_SAMPLER import super_duper_sampler_ordered as Henry_sampler

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class LGB_Model1:

	def __init__(self,parameters,num_tracks_considered,playlists):
		
		self.parameters = parameters

		tmp = Most_Frequent_Model([num_tracks_considered]) 
		tmp.fit(playlists)

		self.list_to_try = tmp.mostfrequent

		MAX_FEATURES_PLAYLIST_NAMES=15000
		MAX_FEATURES_PREDICTION_ARTIST_NAMES=1000
		MAX_FEATURES_PREDICTION_ALBUM_NAMES=1000
		MAX_FEATURES_PREDICTION_TRACK_NAMES=15000
		MAX_FEATURES_TRAIN_ARTIST_NAMES=1000
		MAX_FEATURES_TRAIN_ALBUM_NAMES=1000
		MAX_FEATURES_TRAIN_TRACK_NAMES=1000
		MAX_FEATURES_CHARS=1000
		MIN_COUNTS=10
    
		MAX_FEATURES_SHARED_TRACK=1000
		MAX_FEATURES_SHARED_ALBUM=1000
		MAX_FEATURES_SHARED_ARTIST=1000
		MIN_COUNTS_SHARED=1

		tv1  = TfidfVectorizer(max_features=MAX_FEATURES_PLAYLIST_NAMES,ngram_range=(1, 2),min_df=MIN_COUNTS)
		tv2  = TfidfVectorizer(max_features=MAX_FEATURES_PREDICTION_ARTIST_NAMES,ngram_range=(1,2),min_df=MIN_COUNTS)
		tv3  = TfidfVectorizer(max_features=MAX_FEATURES_PREDICTION_ALBUM_NAMES,ngram_range=(1,2),min_df=MIN_COUNTS)
		tv4  = TfidfVectorizer(max_features=MAX_FEATURES_PREDICTION_TRACK_NAMES,ngram_range=(1,2),min_df=MIN_COUNTS)
		tv5  = TfidfVectorizer(max_features=MAX_FEATURES_TRAIN_ARTIST_NAMES,ngram_range=(1,2),min_df=MIN_COUNTS)  
		tv6  = TfidfVectorizer(max_features=MAX_FEATURES_TRAIN_ALBUM_NAMES,ngram_range=(1,2),min_df=MIN_COUNTS)
		tv7  = TfidfVectorizer(max_features=MAX_FEATURES_TRAIN_TRACK_NAMES,ngram_range=(1,2),min_df=MIN_COUNTS)
    
		tv8  = TfidfVectorizer(max_features=MAX_FEATURES_SHARED_TRACK,ngram_range=(1,1),min_df=MIN_COUNTS_SHARED)
		tv9  = TfidfVectorizer(max_features=MAX_FEATURES_SHARED_ALBUM,ngram_range=(1,1),min_df=MIN_COUNTS_SHARED)
		tv10 = TfidfVectorizer(max_features=MAX_FEATURES_SHARED_ARTIST,ngram_range=(1,1),min_df=MIN_COUNTS_SHARED)
		tv11 = TfidfVectorizer(max_features=MAX_FEATURES_CHARS,ngram_range=(1,3),min_df=MIN_COUNTS,analyzer="char")

		self.vectorizers=[tv1,tv2,tv3,tv4,tv5,tv6,tv7,tv8,tv9,tv10,tv11]

	def fit(self,playlist_info,playlists,track_info_dict,num_boost_round,early_stopping_rounds,number_of_samples,FPs,num_seed_track = 0,verbose_eval=1,tuning=True,test_size=0.2,seed=134):

		Subsampled_data_frame = Henry_sampler(playlists,seed,number_of_samples,FPs,num_seed_track,len(self.list_to_try),False)

		y = Subsampled_data_frame["RESPONSE"]
		X,self.vectorizers = Generate_features(Subsampled_data_frame,playlist_info,playlists,track_info_dict,self.vectorizers,True)

		if (tuning):

			train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = test_size, random_state = 101) 

			d_train   = lgb.Dataset(train_X, label=train_y)
			d_valid   = lgb.Dataset(valid_X, label=valid_y)

			watchlist = [d_train,d_valid]

		else:

			d_train   = lgb.Dataset(X, label=y)
			watchlist = d_train

		self.model = lgb.train(self.parameters, train_set = d_train, num_boost_round = num_boost_round, valid_sets = watchlist, early_stopping_rounds = early_stopping_rounds, verbose_eval=verbose_eval)


	def predict(self,playlist_info,test_playlists,playlists,track_info_dict):

		def predict_one(ii):

			if (ii % 10 == 0):
				print(ii)

			info       = playlist_info.loc[ii,:]
			playlist   = test_playlists[ii]
			tobetested = [x for x in self.list_to_try if x not in playlist]

			testdata   = pd.DataFrame({"TO_PREDICT":tobetested})
			testdata["PLAYLISTS"] = [playlist for ii in range(len(tobetested))]
			testdata["PID"]       = 0 

			X_test  = Generate_features(testdata,info,playlists,track_info_dict,self.vectorizers,False)
			outputs = self.model.predict(X_test)*(-1)
			indices = np.argsort(outputs)
			return([testdata["TO_PREDICT"][ii] for ii in indices[:500]])


		predictions = list(map(predict_one,range(len(test_playlists))))

		return(predictions)
