import pandas as pd 
import numpy as np

import re

upper   = re.compile('[A-Z]')
numbers = re.compile('[0-9]')

def count_numbers(key):
    return len(numbers.findall(key))
    
def count_upper(key):
    return len(upper.findall(key))

def Extract_Special_Chars_From_Track_NAME(featuredaf,names,ids):

	featuredaf['track_name_length'] = np.array(list(map(len, names)))
	featuredaf['track_num_exclams'] = np.array(list(map(lambda d: d.count('!') , names)))
	featuredaf['track_num_questmk'] = np.array(list(map(lambda d: d.count('?') , names)))
	featuredaf['track_num_fullstp'] = np.array(list(map(lambda d: d.count('.') , names)))
	featuredaf['track_num_commas']  = np.array(list(map(lambda d: d.count(',') , names)))
	featuredaf['track_num_hastags'] = np.array(list(map(lambda d: d.count('#') , names)))
	featuredaf['track_num_at']      = np.array(list(map(lambda d: d.count('@') , names)))
	featuredaf['track_num_brackri'] = np.array(list(map(lambda d: d.count(')') , names)))
	featuredaf['track_num_brackle'] = np.array(list(map(lambda d: d.count('(') , names)))
	featuredaf['track_num_minus']   = np.array(list(map(lambda d: d.count('-') , names)))
	featuredaf['track_length']      = np.array(list(map(lambda d: d.count(' ') , names)))
	featuredaf['track_num_perc']    = np.array(list(map(lambda d: d.count('%') , names)))
	featuredaf['track_num_doll']    = np.array(list(map(lambda d: d.count('$') , names)))
	featuredaf['track_num_star']    = np.array(list(map(lambda d: d.count('*') , names)))
	featuredaf['track_num_ands']    = np.array(list(map(lambda d: d.count('&') , names)))
	featuredaf['track_num_equal']   = np.array(list(map(lambda d: d.count('=') , names)))
	featuredaf['track_num_upper']   = np.array(list(map(count_upper, names)))
	featuredaf['track_num_numbers'] = np.array(list(map(count_numbers, names)))