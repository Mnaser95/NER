####################################### Libraries
import numpy as np
import mne
import sys
import scipy.io
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from scipy.signal import stft
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from collections import Counter
import pandas as pd
import statsmodels.api as sm
from experiment_prep import experiment_prep
import random
import os
import tensorflow as tf
##################################### Inputs
#sess=[i for i in range(1,19)] # list of sessions (for all 9 subjects). Two sessions per subject so the total is 18.
sess=[6,18,5,16,15] # list of sessions (for all 9 subjects). Two sessions per subject so the total is 18.
#sess=[6,16,8,17,3]

f_low_MI=.5   # low frequency
f_high_MI=60 # high frequency
tmin_MI = 0
tmax_MI = 4
fs=250
notch_f=50

def load_data(ses, data_type):
    my_file = fr"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\PhD-S7\Dissertation\Data\2a2b data\full_2a_data\Data\{ses-1}.mat"
    mat_data = scipy.io.loadmat(my_file)
    if data_type == 'rest':
        my_data_eeg = np.squeeze(mat_data['data'][0][1][0][0][0][:, 0:22]) # the first 22 channels are EEG
        my_data_eog = np.squeeze(mat_data['data'][0][1][0][0][0][:, 22:25]) # the rest are EOG
    elif data_type == 'mi':
        my_data_eeg = np.squeeze(mat_data['data'][0][run+3][0][0][0][:, 0:22])
        my_data_eog = np.squeeze(mat_data['data'][0][run+3][0][0][0][:, 22:25])
    return np.hstack([my_data_eeg, my_data_eog]),mat_data
def create_mne_raw(data):
    numbers = list(range(1, 26))
    ch_names = [str(num) for num in numbers]
    ch_types = ['eeg'] * 22 + ['eog'] * 3
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
    raw = mne.io.RawArray(data.T, info)
    return raw
def process_mi_data(raw, mat_data):
    raw.filter(f_low_MI, f_high_MI, fir_design='firwin') # FIR filtration to keep a range of frequencies
    raw.notch_filter(notch_f, fir_design='firwin') # FIR filtration to keep a range of frequencies

    events = np.squeeze(mat_data['data'][0][run+3][0][0][2]) # only the first run of each session is taken (total number of trials is 48, only left and right hand considered so 24)
    event_indices = np.squeeze(mat_data['data'][0][run+3][0][0][1])
    mne_events = np.column_stack((event_indices, np.zeros_like(event_indices), events))

    event_id_MI = dict({'769': 1, '770': 2})
    epochs_MI = mne.Epochs(raw, mne_events, event_id_MI, tmin_MI, tmax_MI, proj=True,  baseline=None, preload=True)
    labels_MI = epochs_MI.events[:, -1]
    data_MI_original = epochs_MI.get_data()

    return (labels_MI,data_MI_original)

all_ses_data_list=[]
all_ses_labels_list=[]

for ses in sess:
    all_run_data=[]
    all_run_labels=[]
    for run in range(0,3):
        mi_data, mat_data = load_data(ses, 'mi')
        raw = create_mne_raw(mi_data)

        labels_MI,data_MI_original = process_mi_data(raw, mat_data)
        all_run_data.append(data_MI_original)     
        all_run_labels.append(labels_MI)  
    all_ses_data=np.concatenate(all_run_data,axis=0)
    all_ses_labels=np.concatenate(all_run_labels,axis=0)

    all_ses_data_list.append(all_ses_data)
    all_ses_labels_list.append(all_ses_labels)

all_ses_data_arr=np.array(all_ses_data_list)    
all_ses_labels_arr=np.array(all_ses_labels_list)   

all_ses_data_arr_reshaped=all_ses_data_arr.reshape(-1,all_ses_data_arr.shape[2],all_ses_data_arr.shape[3])
data_ready=all_ses_data_arr_reshaped.swapaxes( 1, 2)
labels_ready=all_ses_labels_arr.reshape(-1)


# split_data = []
# split_labels = []

# for i in range(len(data_ready)):
#     split_data.append(data_ready[i,:500,:])
#     split_data.append(data_ready[i,500:-1,:])
#     split_labels.append(labels_ready[i])
#     split_labels.append(labels_ready[i])

split_data=data_ready
split_labels=labels_ready

split_data_ready=np.stack(split_data)
split_labels_ready=np.stack(split_labels)

acc_res=experiment_prep(split_data_ready,split_labels_ready)

stop=1