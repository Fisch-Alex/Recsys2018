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
from Extract_Special_Chars_From_Past_NAME import Extract_Special_Chars_From_Past_NAME

def Generate_features_scenario2(Subsampled_data_frame,playlist_info,playlists,track_info_dict,list_to_try,CV_list_of_common_tracks,CV_list_of_common_albums,CV_list_of_common_artists,list_of_vectorisers,Training):

	proportions = Get_Proportion(Subsampled_data_frame["TO_PREDICT"],playlists)
	featuredaf  = pd.DataFrame({"PROPORTIONS":proportions})
	
	Extract_Special_Chars_From_Track_NAME(featuredaf,[str(track_info_dict[ids]['track_name']) for ids in Subsampled_data_frame["TO_PREDICT"]],Subsampled_data_frame["PID"])
	Extract_Shared_Final_Artist_Album(featuredaf,Subsampled_data_frame,track_info_dict)
	Extract_Special_Chars_From_Past_NAME(featuredaf,Subsampled_data_frame["PLAYLISTS"],track_info_dict)

	featuredaf["Song_duration"]          = np.array([track_info_dict[ids]['duration_ms'] for ids in Subsampled_data_frame["TO_PREDICT"]])
	featuredaf["Song_duration_relative"] = (featuredaf["Song_duration"] - featuredaf["track_mean"])/np.sqrt(featuredaf["track_variance"])

	featuredaf["num_tracks"]        = np.array([playlist_info["num_tracks"][ii] for ii in Subsampled_data_frame["PID"]]) 

	SPARSE_ARTIST_ALBUM_TRACK_ARRAY = Artist_Album_Track_SPARSE(Subsampled_data_frame,track_info_dict,list_to_try,CV_list_of_common_tracks,CV_list_of_common_albums,CV_list_of_common_artists)

	track_names    = cleaner([str(track_info_dict[ids]['track_name']) for ids in Subsampled_data_frame["TO_PREDICT"]])
	tv11           = list_of_vectorisers[10]

	if Training:
		Huge_BOW, vectorizers = BOW(Subsampled_data_frame,playlist_info,playlists,track_info_dict,Training,list_of_vectorisers,2)
		CHAR_GRAM  = tv11.fit_transform(track_names)
		vectorizers.append(tv11)
	else:
		Huge_BOW = BOW(Subsampled_data_frame,playlist_info,playlists,track_info_dict,Training,list_of_vectorisers,2)
		CHAR_GRAM  = tv11.transform(track_names)

	output = hstack((featuredaf,Huge_BOW,CHAR_GRAM,SPARSE_ARTIST_ALBUM_TRACK_ARRAY)).tocsr()

	if Training:
		return(output,vectorizers)
	else:
		return(output)