import pandas as pd 
import numpy as np

from collections import Counter

import re

upper   = re.compile('[A-Z]')
numbers = re.compile('[0-9]')

def count_numbers(key):
    return len(numbers.findall(key))
    
def count_upper(key):
    return len(upper.findall(key))

def Extract_Special_Chars_From_Playlist_NAME(featuredaf,names,ids):

	considered_names = [str(names[ii]) for ii in ids]

	featuredaf['pl_name_length'] = np.array(list(map(len, considered_names)))
	featuredaf['pl_num_exclams'] = np.array(list(map(lambda d: d.count('!') , considered_names)))
	featuredaf['pl_num_questmk'] = np.array(list(map(lambda d: d.count('?') , considered_names)))
	featuredaf['pl_num_fullstp'] = np.array(list(map(lambda d: d.count('.') , considered_names)))
	featuredaf['pl_num_commas']  = np.array(list(map(lambda d: d.count(',') , considered_names)))
	featuredaf['pl_num_hastags'] = np.array(list(map(lambda d: d.count('#') , considered_names)))
	featuredaf['pl_num_at']      = np.array(list(map(lambda d: d.count('@') , considered_names)))
	featuredaf['pl_num_brackri'] = np.array(list(map(lambda d: d.count(')') , considered_names)))
	featuredaf['pl_num_brackle'] = np.array(list(map(lambda d: d.count('(') , considered_names)))
	featuredaf['pl_num_minus']   = np.array(list(map(lambda d: d.count('-') , considered_names)))
	featuredaf['pl_length']      = np.array(list(map(lambda d: d.count(' ') , considered_names)))
	featuredaf['pl_num_perc']    = np.array(list(map(lambda d: d.count('%') , considered_names)))
	featuredaf['pl_num_doll']    = np.array(list(map(lambda d: d.count('$') , considered_names)))
	featuredaf['pl_num_star']    = np.array(list(map(lambda d: d.count('*') , considered_names)))
	featuredaf['pl_num_ands']    = np.array(list(map(lambda d: d.count('&') , considered_names)))
	featuredaf['pl_num_equal']   = np.array(list(map(lambda d: d.count('=') , considered_names)))
	featuredaf['pl_num_upper']   = np.array(list(map(count_upper, considered_names)))
	featuredaf['pl_num_numbers'] = np.array(list(map(count_numbers, considered_names)))
