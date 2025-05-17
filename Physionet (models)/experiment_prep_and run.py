
#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Karel Roots"

import os
import sys

import numpy as np
from EEGModels import get_models
from data_loader import load_data
from experiment_data import Experiment
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from experiment_running import run_experiment
import random
import tensorflow as tf

full_res=[]
#full_list=[93,69,60,15,101]
full_list=[93,69,65,9,78]
#full_list=[i for i in range(1,110)]
subs_considered = [f"S{str(e).zfill(3)}" for e in full_list if len(f"S{str(e).zfill(3)}") == 4]

nr_of_epochs = 75
nb_classes = 2
trial_type = 2 # 2: imagined
seed=172
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
val_size=.125
test_size=0.2
use_cpu = True 
test_model=True
base_folder=fr'C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\PhD-S7\Dissertation\Data\data\\'

# Loading data from files
X, y = load_data(FNAMES=subs_considered, trial_type=trial_type, chunk_data=True, 
                chunks=2, base_folder=base_folder, sample_rate=160,
                samples=640,cpu_format=use_cpu,
                preprocessing=True, bp_low=6, bp_high=40,notch_f=50,
                notch=False, bp_filter=True, artifact_removal=True,normalize=True,num_trials_per_run=15)

# Data formatting
if use_cpu:
    print("Using CPU")
    K.set_image_data_format('channels_last')
    samples = X.shape[1]
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
else:
    print("Using GPU")
    K.set_image_data_format('channels_first')
    samples = X.shape[2]
    X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
y = to_categorical(y, nb_classes)

my_experiment = Experiment(trial_type, 
                        'Num1', 
                        get_models(trial_type, nb_classes, samples, use_cpu), 
                        nr_of_epochs,
                        val_size, 
                        test_size)

#X=X[:,160:480,:,:]

temp_res=run_experiment(X, y, my_experiment, use_cpu, test_model,seed)

stop=1