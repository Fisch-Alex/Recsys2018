import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer

def Artist_Album_Track_SPARSE(Subsampled_data_frame,track_info_dict,list_to_try,CV_list_of_common_tracks,CV_list_of_common_albums,CV_list_of_common_artists):

	Albums_in_seed  = [ [track_info_dict[track]['album_id'] for track in Seed_playlist] for Seed_playlist in Subsampled_data_frame["PLAYLISTS"]]
	Artists_in_seed = [ [track_info_dict[track]['artist_id'] for track in Seed_playlist] for Seed_playlist in Subsampled_data_frame["PLAYLISTS"]]

	Albums_in_seed_string  = [' '.join([str(x*10) for x in album_ids ]) for album_ids  in Albums_in_seed ]
	Artists_in_seed_string = [' '.join([str(x*10) for x in artist_ids]) for artist_ids in Artists_in_seed]
	Track_in_seed_string   = [' '.join([str(x*10) for x in playlist]) for playlist in Subsampled_data_frame["PLAYLISTS"]]

	Albums_Sparse  = CV_list_of_common_albums.transform(Albums_in_seed_string)
	Artists_Sparse = CV_list_of_common_artists.transform(Artists_in_seed_string)
	Tracks_Sparse  = CV_list_of_common_tracks.transform(Track_in_seed_string)

	Predict_Album_sparse  = CV_list_of_common_albums.transform([str(track_info_dict[ids]['album_id']*10) for ids in Subsampled_data_frame["TO_PREDICT"]])
	Predict_Artist_sparse = CV_list_of_common_artists.transform([str(track_info_dict[ids]['artist_id']*10) for ids in Subsampled_data_frame["TO_PREDICT"]])
	Predict_Track_sparse  = CV_list_of_common_tracks.transform([str(ids*10) for ids in Subsampled_data_frame["TO_PREDICT"]])

	Sparse_array = hstack( ( Albums_Sparse,Artists_Sparse,Predict_Album_sparse,Predict_Artist_sparse,Tracks_Sparse,Predict_Track_sparse) ).tocsr()

	return(Sparse_array)