import pandas as pd 
import numpy as np 

from Get_Proportion import Get_Proportion 
from Extract_Special_Chars_From_Playlist_NAME import Extract_Special_Chars_From_Playlist_NAME
from Extract_Special_Chars_From_Track_NAME import Extract_Special_Chars_From_Track_NAME
from Extract_Shared_Final_Artist_Album import Extract_Shared_Final_Artist_Album
from Find_Common_Words import Find_Common_Words
from Artist_Album_Track_SPARSE_LARGE import Artist_Album_Track_SPARSE
from BOW_FINAL import BOW , cleaner

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import csr_matrix, hstack

def Generate_features_scenario10(Subsampled_data_frame,playlist_info,playlists,track_info_dict,list_to_try,CV_list_of_common_tracks,CV_list_of_common_albums,CV_list_of_common_artists,list_of_vectorisers,Training):

	proportions = Get_Proportion(Subsampled_data_frame["TO_PREDICT"],playlists)
	featuredaf  = pd.DataFrame({"PROPORTIONS":proportions})
	Extract_Special_Chars_From_Playlist_NAME(featuredaf,playlist_info["name"],Subsampled_data_frame["PID"])
	Extract_Special_Chars_From_Track_NAME(featuredaf,[str(track_info_dict[ids]['track_name']) for ids in Subsampled_data_frame["TO_PREDICT"]],Subsampled_data_frame["PID"])
	Extract_Shared_Final_Artist_Album(featuredaf,Subsampled_data_frame,track_info_dict)

	shared_track_name   = Find_Common_Words([track_info_dict[ids]['track_name'] for ids in Subsampled_data_frame["TO_PREDICT"]] , [playlist_info["name"][ii] for ii in Subsampled_data_frame["PID"]])
	shared_album_name   = Find_Common_Words([track_info_dict[ids]['album_name'] for ids in Subsampled_data_frame["TO_PREDICT"]] , [playlist_info["name"][ii] for ii in Subsampled_data_frame["PID"]])
	shared_artist_name  = Find_Common_Words([track_info_dict[ids]['artist_name'] for ids in Subsampled_data_frame["TO_PREDICT"]], [playlist_info["name"][ii] for ii in Subsampled_data_frame["PID"]])

	featuredaf["shared_track_num"]  = np.array(list(map(len,shared_track_name)))
	featuredaf["shared_album_num"]  = np.array(list(map(len,shared_album_name)))
	featuredaf["shared_artist_num"] = np.array(list(map(len,shared_artist_name)))
	featuredaf["num_tracks"]        = np.array([playlist_info["num_tracks"][ii] for ii in Subsampled_data_frame["PID"]]) 
	featuredaf["Song_duration"]          = np.array([track_info_dict[ids]['duration_ms'] for ids in Subsampled_data_frame["TO_PREDICT"]])
	featuredaf["Song_duration_relative"] = (featuredaf["Song_duration"] - featuredaf["track_mean"])/np.sqrt(featuredaf["track_variance"])
	
	#shared_track_name_words  = [' '.join(words) for words in shared_track_name]
	#shared_album_name_words  = [' '.join(words) for words in shared_album_name]
	#shared_artist_name_words = [' '.join(words) for words in shared_artist_name]

	playlist_names = cleaner([playlist_info["name"][ii] for ii in Subsampled_data_frame["PID"]])
	tv11           = list_of_vectorisers[0]

	SPARSE_ARTIST_ALBUM_TRACK_ARRAY = Artist_Album_Track_SPARSE(Subsampled_data_frame,track_info_dict,list_to_try,CV_list_of_common_tracks,CV_list_of_common_albums,CV_list_of_common_artists)

	if Training:
		CHAR_GRAM = tv11.fit_transform(playlist_names)
		list_of_vectorisers = [tv11]
	else:
		CHAR_GRAM = tv11.transform(playlist_names)

	# output = hstack((featuredaf,SPARSE_ARTIST_ALBUM_TRACK_ARRAY,Huge_BOW)).tocsr()
	output = hstack((featuredaf,SPARSE_ARTIST_ALBUM_TRACK_ARRAY,CHAR_GRAM)).tocsr()

	if Training:
		return(output,list_of_vectorisers)
	else:
		return(output)