from collections import Counter
import os
import pickle

os.chdir(os.path.expanduser('~/Spotify'))

with open('data/most_frequent_tracks', 'rb') as handle:
    Most_frequent = pickle.load(handle)

class Most_Frequent_Model:

	def __init__(self,parameters):
		
		self.mostfrequent = []
		self.name = "basic_frequency"

		self.parameters = parameters


	def fit(self,playlists):

		#tracks      = [track for playlist in playlists for track in playlist]
		#frequencies = Counter(tracks)

		#mostfrequent = frequencies.most_common(self.parameters[0])

		#self.mostfrequent = [x[0] for x in mostfrequent]

		self.mostfrequent = [int(x) for x in Most_frequent[:(self.parameters[0])]]


	def predict(self,playlists):
		return([[x for x in self.mostfrequent if x not in playlist][:500] for playlist in playlists])


