import pandas as pd 
import numpy as np 

from Get_Proportion import Get_Proportion 
from Extract_Special_Chars_From_Playlist_NAME import Extract_Special_Chars_From_Playlist_NAME
from Extract_Special_Chars_From_Track_NAME import Extract_Special_Chars_From_Track_NAME
from Find_Common_Words import Find_Common_Words
from BOW_FINAL import BOW , cleaner

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import csr_matrix, hstack

def Generate_features_scenario1(Subsampled_data_frame,playlist_info,playlists,track_info_dict,list_of_vectorisers,Training):

	proportions = Get_Proportion(Subsampled_data_frame["TO_PREDICT"],playlists)
	featuredaf  = pd.DataFrame({"PROPORTIONS":proportions})
	Extract_Special_Chars_From_Playlist_NAME(featuredaf,playlist_info["name"],Subsampled_data_frame["PID"])
	Extract_Special_Chars_From_Track_NAME(featuredaf,[str(track_info_dict[ids]['track_name']) for ids in Subsampled_data_frame["TO_PREDICT"]],Subsampled_data_frame["PID"])

	shared_track_name   = Find_Common_Words([track_info_dict[ids]['track_name'] for ids in Subsampled_data_frame["TO_PREDICT"]] , [playlist_info["name"][ii] for ii in Subsampled_data_frame["PID"]])
	shared_album_name   = Find_Common_Words([track_info_dict[ids]['album_name'] for ids in Subsampled_data_frame["TO_PREDICT"]] , [playlist_info["name"][ii] for ii in Subsampled_data_frame["PID"]])
	shared_artist_name  = Find_Common_Words([track_info_dict[ids]['artist_name'] for ids in Subsampled_data_frame["TO_PREDICT"]], [playlist_info["name"][ii] for ii in Subsampled_data_frame["PID"]])

	featuredaf["shared_track_num"]  = np.array(list(map(len,shared_track_name)))
	featuredaf["shared_album_num"]  = np.array(list(map(len,shared_album_name)))
	featuredaf["shared_artist_num"] = np.array(list(map(len,shared_artist_name)))
	featuredaf["num_tracks"]        = np.array([playlist_info["num_tracks"][ii] for ii in Subsampled_data_frame["PID"]]) 
	featuredaf["Song_duration"]     = np.array([track_info_dict[ids]['duration_ms'] for ids in Subsampled_data_frame["TO_PREDICT"]])

	playlist_names = cleaner([playlist_info["name"][ii] for ii in Subsampled_data_frame["PID"]])
	tv11           = list_of_vectorisers[10]
	
	if Training:
		CHAR_GRAM = tv11.fit_transform(playlist_names)
		Huge_BOW, vectorizers = BOW(Subsampled_data_frame,playlist_info,playlists,track_info_dict,Training,list_of_vectorisers,1)
		vectorizers.append(tv11)
	else:
		Huge_BOW = BOW(Subsampled_data_frame,playlist_info,playlists,track_info_dict,Training,list_of_vectorisers,1)
		CHAR_GRAM = tv11.transform(playlist_names)

	output = hstack((featuredaf,Huge_BOW,CHAR_GRAM)).tocsr()

	if Training:
		return(output,vectorizers)
	else:
		return(output)