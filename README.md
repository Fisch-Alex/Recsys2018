# Recsys2018
Final model for the 2018 Recys challenge

Our solutions can be reproduced as follows:

1) Move all of these files to the directory "~/Spotify" 
2) Store all the csv files in the data directory
3) Run transformer.py, transformer2.py, ... , transformer10.py (in that order!) 
4) Move all csv files to the data directory 
5) Run Transformer11.py 
6) Run make_data.py

These steps significantly reduce the disk space required creating a variety of dictionaries. We are now ready to run the models 

1) Run Challenge1.py
2) Run Challenge2_exp.py
...
10) Run Challenge10_exp.py

(N.B. We ran these models on large nodes. Some of these models request 25 cores and need 400GB of RAM - You can however reduce the amout of data used to reduce the computational ressources required.)

We can now match all these csv files into one final submission: 

1) Run CreateSubmission_alternative.py 

This is it! Please do not hesitate to get in touch should you have any questions. 
