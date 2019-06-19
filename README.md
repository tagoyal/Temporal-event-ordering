# Temporal Event Ordering
This repository contains data and code for the following paper: Embedding time expressions for deep temporal ordering models, Tanya Goyal and Greg Durrett, ACL 2019. (short paper)

# Dataset
The paper contains experiments on two dataset: MATRES and distantly collected data.

The folder distant_data contains 5000 temporally ordered event pairs collected from Gigaword. The format of the data is: <sentence 1> <sentence 2> <event 1> <event 2> < temporal relation>.
In some cases, sentence 1 and sentence 2 may correspond to the same sentence. 
The code uses processed distant data and can be downloaded <a href="https://drive.google.com/open?id=1J69G_xa3KbilMpz9in7PxlfNtagX6zx7">here</a> 

# Code
The timex_model folder contains code for training timex models from randomly sampled timex pairs.

The event_model folder contains code for training event ordering temporal models (with and without timex embeddings).

The download_data folder contains code for collecting additional event pairs from Gigaword.  
