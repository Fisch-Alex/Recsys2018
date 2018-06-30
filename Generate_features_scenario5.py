import pandas as pd 
import numpy as np 

from Get_Proportion import Get_Proportion 
from Extract_Special_Chars_From_Playlist_NAME import Extract_Special_Chars_From_Playlist_NAME
from Extract_Special_Chars_From_Track_NAME import Extract_Special_Chars_From_Track_NAME
from Extract_Shared_Final_Artist_Album import Extract_Shared_Final_Artist_Album
from Find_Common_Words import Find_Common_Words
from Artist_Album_Track_SPARSE_LARGE import Artist_Album_Track_SPARSE
from BOW_FINAL import BOW , cleaner

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack

def Generate_features_scenario5(Subsampled_data_frame,playlist_info,playlists,track_info_dict,list_to_try,CV_list_of_common_tracks,CV_list_of_common_albums,CV_list_of_common_artists,list_of_vectorisers,Training):

	proportions = Get_Proportion(Subsampled_data_frame["TO_PREDICT"],playlists)
	featuredaf  = pd.DataFrame({"PROPORTIONS":proportions})
	Extract_Special_Chars_From_Track_NAME(featuredaf,[str(track_info_dict[ids]['track_name']) for ids in Subsampled_data_frame["TO_PREDICT"]],Subsampled_data_frame["PID"])
	Extract_Shared_Final_Artist_Album(featuredaf,Subsampled_data_frame,track_info_dict)

	featuredaf["num_tracks"]             = np.array([playlist_info["num_tracks"][ii] for ii in Subsampled_data_frame["PID"]]) 
	featuredaf["Song_duration"]          = np.array([track_info_dict[ids]['duration_ms'] for ids in Subsampled_data_frame["TO_PREDICT"]])
	featuredaf["Song_duration_relative"] = (featuredaf["Song_duration"] - featuredaf["track_mean"])/np.sqrt(featuredaf["track_variance"])

	#shared_track_name_words  = [' '.join(words) for words in shared_track_name]
	#shared_album_name_words  = [' '.join(words) for words in shared_album_name]
	#shared_artist_name_words = [' '.join(words) for words in shared_artist_name]

	SPARSE_ARTIST_ALBUM_TRACK_ARRAY = Artist_Album_Track_SPARSE(Subsampled_data_frame,track_info_dict,list_to_try,CV_list_of_common_tracks,CV_list_of_common_albums,CV_list_of_common_artists)

	if Training:
		Huge_BOW, vectorizers = BOW(Subsampled_data_frame,playlist_info,playlists,track_info_dict,Training,list_of_vectorisers,5)
	else:
		Huge_BOW = BOW(Subsampled_data_frame,playlist_info,playlists,track_info_dict,Training,list_of_vectorisers,5)

	output = hstack((featuredaf,Huge_BOW,SPARSE_ARTIST_ALBUM_TRACK_ARRAY)).tocsr()

	if Training:
		return(output,vectorizers)
	else:
		return(output)