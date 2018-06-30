import numpy as np
from collections import Counter

def obtain_homogeneity(x):
	return(sum([1/float(y[1]) for y in list(x.most_common())]))

def obtain(x,y):
	return(int(x[y]))

def number(x):
	return(len(dict(x)))

def max_number(x):
	return(max(dict(x).values()))

def Extract_Shared_Artist_Album(featuredaf,Subsampled_data_frame,track_info_dict):

	Albums_in_seed     = [ [track_info_dict[track]['album_id'] for track in Seed_playlist] for Seed_playlist in Subsampled_data_frame["PLAYLISTS"]]
	Artists_in_seed    = [ [track_info_dict[track]['artist_id'] for track in Seed_playlist] for Seed_playlist in Subsampled_data_frame["PLAYLISTS"]]
	Track_time         = [ [track_info_dict[track]['duration_ms'] for track in Seed_playlist] for Seed_playlist in Subsampled_data_frame["PLAYLISTS"]]
	Track_time_squared = [ [track_info_dict[track]['duration_ms']**2 for track in Seed_playlist] for Seed_playlist in Subsampled_data_frame["PLAYLISTS"]]

	Album_frequencies    = list(map(Counter,Albums_in_seed))
	Artist_frequencies   = list(map(Counter,Artists_in_seed))

	featuredaf["ARTIST_NUMBER"] = np.array(list(map(number,Artist_frequencies)))
	featuredaf["ALBUM_NUMBER"]  = np.array(list(map(number,Album_frequencies)))	

	featuredaf["ARTIST_MAX_NUMBER"] = np.array(list(map(max_number,Artist_frequencies)))
	featuredaf["ALBUM_MAX_NUMBER"]  = np.array(list(map(max_number,Album_frequencies)))	

	featuredaf["Artist_Difference"] = featuredaf["ARTIST_MAX_NUMBER"] - featuredaf["ARTIST_NUMBER"]
	featuredaf["Album_Difference"]  = featuredaf["ALBUM_MAX_NUMBER"]  - featuredaf["ALBUM_NUMBER"]

	featuredaf["ARTIST_NUMBER_HOMOGENEITY"] = np.array(list(map(obtain_homogeneity,Artist_frequencies)))
	featuredaf["ALBUM_NUMBER_HOMOGENEITY"]  = np.array(list(map(obtain_homogeneity,Album_frequencies)))

	featuredaf["ARTIST_NUMBER_IN_PLAYLIST"] = np.array(list(map(obtain,Artist_frequencies,[track_info_dict[ids]['artist_id'] for ids in Subsampled_data_frame["TO_PREDICT"]])))
	featuredaf["ALBUM_NUMBER_IN_PLAYLIST"]  = np.array(list(map(obtain,Album_frequencies, [track_info_dict[ids]['album_id']  for ids in Subsampled_data_frame["TO_PREDICT"]])))

	featuredaf["track_mean"]     = np.array(list(map(sum,Track_time)))/len(Track_time[0])
	featuredaf["track_variance"] = np.array(list(map(sum,Track_time_squared)))/len(Track_time[0]) - featuredaf["track_mean"]*featuredaf["track_mean"]