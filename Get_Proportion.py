from Load_Playlists import Load_Playlists
from collections import Counter

playlists   = Load_Playlists(1000)

tracks      = [int(track) for playlist in playlists for track in playlist]

frequencies = Counter(tracks)

def Get_Proportion(tracklist,playlists):

	return([frequencies[int(track)] for track in tracklist])