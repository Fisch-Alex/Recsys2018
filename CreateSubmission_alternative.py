import pandas as pd 
import os
import csv

os.chdir(os.path.expanduser('~/Spotify'))

conversions = pd.read_csv("data/Track-Number.csv")

id_dict = {conversions["our_id"][ii]:conversions["track_id"][ii] for ii in range(conversions.shape[0])}

Challengenames = [
"name_only",
"random_five",
"name_first",
"name_first_five",
"first_ten",
"name_first_ten",
"name_first_twentyfive",
"name_hundred",
"name_twentyfive",
"name_first_hundred"
]

def Get_Playlist(line):
	return([id_dict[int(song)] for song in line])

def Load_Prediction_file(ii):
	with open("Submissions_exp/challenge"+str(ii+1)+"_RAW_LARGE.csv") as tsv:
		Playlist = list(map(Get_Playlist,csv.reader(tsv, delimiter=',')))
		[playlist for playlist in Playlist if len(playlist) > 0 ]
	return(Playlist)

List_of_Prediction_lists = list(map(Load_Prediction_file,range(10)))
List_of_Predictions      = [x for y in List_of_Prediction_lists for x in y]

def Get_pids(name):
	return(list(pd.read_csv("data/Test/meta_"+name+".csv")["pid"]))

List_of_ids_list = list(map(Get_pids,Challengenames))


List_of_ids = [x for y in List_of_ids_list for x in y]

finallist = [ ([List_of_ids[ii]] + List_of_Predictions[ii]) for ii in range(len(List_of_ids))]

header = [["team_info" , "STORMtroopers" , "main", "a.t.fisch@lancaster.ac.uk"]]

with open("Submissions_exp/FINAL_SUBMISSION_exp_main.csv", "w") as f:
	writer = csv.writer(f)
	writer.writerows(header)

with open("Submissions_exp/FINAL_SUBMISSION_exp_main.csv", "a") as f:
	writer = csv.writer(f)
	writer.writerows(finallist)