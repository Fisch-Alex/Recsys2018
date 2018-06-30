import pandas as pd 
import numpy as np

import re

upper   = re.compile('[A-Z]')
numbers = re.compile('[0-9]')

def count_numbers(key):
    return len(numbers.findall(key))
    
def count_upper(key):
    return len(upper.findall(key))

def Extract_Special_Chars_From_Past_NAME(featuredaf,seed_playlists,track_info_dict):

	track_num_upper = {x:count_upper(track_info_dict[x]["track_name"]) for x in track_info_dict.keys()}
	seed_num_upper  = [ [track_num_upper[track] for track in seedplaylist ] for seedplaylist in seed_playlists]

	track_num_numbe = {x:count_numbers(track_info_dict[x]["track_name"]) for x in track_info_dict.keys()}
	seed_num_numbe  = [ [track_num_numbe[track] for track in seedplaylist ] for seedplaylist in seed_playlists]

	track_length    = {x:len(track_info_dict[x]["track_name"]) for x in track_info_dict.keys()}
	seed_length     = [ [track_length[track] for track in seedplaylist ] for seedplaylist in seed_playlists]

	track_num_words = {x:track_info_dict[x]["track_name"].count(' ') for x in track_info_dict.keys()}
	seed_num_words  = [ [track_num_words[track] for track in seedplaylist ] for seedplaylist in seed_playlists]

	featuredaf["mean_num_upper"] = np.array(list(map(sum,seed_num_upper)))
	featuredaf["mean_num_numbe"] = np.array(list(map(sum,seed_num_numbe)))
	featuredaf["mean_length"]    = np.array(list(map(sum,seed_length)))
	featuredaf["mean_numwords"]  = np.array(list(map(sum,seed_num_words)))