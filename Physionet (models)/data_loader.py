#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Karel Roots"

import os
from glob import glob
import sys
import numpy as np
import pyedflib
sys.path.append('gumpy')
from gumpy import signal

def preprocess_data(data, sample_rate, 
                    bp_low, bp_high, notch_f,
                    notch, bp_filter, artifact_removal, normalize):
    if notch:
        data = notch_filter(data, notch_f, sample_rate)
    if bp_filter:
        data = bandpass_filter(data, bp_low, bp_high, sample_rate)
    if normalize:
        data = normalize_data(data, 'mean_std')
    if artifact_removal:
        data = remove_artifacts(data)

    return data
def notch_filter(data, ac_freq, sample_rate):
    w0 = ac_freq / (sample_rate / 2)
    return signal.notch(data, w0)
def bandpass_filter(data, bp_low, bp_high, sample_rate):
    return signal.butter_bandpass(data, bp_low, bp_high, order=5, fs=sample_rate)
def normalize_data(data, strategy):
    return signal.normalize(data, strategy)
def remove_artifacts(data):
    cleaned = signal.artifact_removal(data.reshape((-1, 1)))[0]
    return np.squeeze(cleaned)

def load_data(FNAMES, trial_type, chunk_data, chunks, base_folder, sample_rate,
              samples, cpu_format, preprocessing, bp_low, bp_high, notch_f,
              notch,bp_filter, artifact_removal,normalize,num_trials_per_run):
    
    # Get file paths
    PATH = base_folder

    # Remove the subjects with incorrectly annotated data that will be omitted from the final dataset
    subjects = ['S038', 'S088', 'S089', 'S092', 'S100', 'S104']
    try:
        for sub in subjects:
            FNAMES.remove(sub)
    except:
        pass

    def convert_label_to_int(str):
        if str == 'T1':
            return 0
        if str == 'T2':
            return 1
        raise Exception("Invalid label %s" % str)

    def divide_chunks(data, chunks):
        for i in range(0, len(data), chunks):
            yield data[i:i + chunks]

    imagined_trials = '04,08,12'.split(',')
    samples_per_chunk = int(samples / chunks)
    
    file_numbers = imagined_trials



    X = []
    y = []

    # Iterate over different subjects
    for subj in FNAMES:

        # Load the file names for given subject
        fnames = glob(os.path.join(PATH, subj, subj + 'R*.edf'))
        fnames = [name for name in fnames if name[-6:-4] in file_numbers]

        # Iterate over the trials for each subject
        for file_name in fnames:

            # Load the file
            # print("File name " + file_name)
            loaded_file = pyedflib.EdfReader(file_name)
            annotations = loaded_file.readAnnotations()
            times = annotations[0]
            durations = annotations[1]
            tasks = annotations[2]

            # Load the data signals into a buffer
            signals = loaded_file.signals_in_file
            # signal_labels = loaded_file.getSignalLabels()
            sigbufs = np.zeros((signals, loaded_file.getNSamples()[0]))
            for i in np.arange(signals):
                sigbufs[i, :] = loaded_file.readSignal(i)

            # initialize the result arrays with preferred shapes
            if chunk_data:
                trial_data = np.zeros((num_trials_per_run, 64, chunks, samples_per_chunk))
            else:
                trial_data = np.zeros((num_trials_per_run, 64, samples))
            labels = []

            signal_start = 0
            k = 0

            # Iterate over tasks in the trial run
            for i in range(len(times)):
                # Collects only the num_trials_per_run non-rest tasks in each run
                if k == num_trials_per_run:
                    break

                current_duration = durations[i]
                signal_end = signal_start + samples

                # Skipping tasks where the user was resting
                if tasks[i] == 'T0':
                    signal_start += int(sample_rate * current_duration)
                    continue

                # Iterate over each channel
                for j in range(len(sigbufs)):
                    channel_data = sigbufs[j][signal_start:signal_end]
                    if preprocessing:
                        channel_data = preprocess_data(channel_data, sample_rate, 
                                                       bp_low,bp_high, notch_f,
                                                       notch, bp_filter, artifact_removal,normalize)
                    if chunk_data:
                        channel_data = list(divide_chunks(channel_data, samples_per_chunk))

                    # Add data for the current channel and task to the result
                    trial_data[k][j] = channel_data

                # add label(s) for the current task to the result
                if chunk_data:
                    # multiply the labels by the chunk size for chunked mode
                    labels.extend([convert_label_to_int(tasks[i])] * chunks)
                else:
                    labels.append(convert_label_to_int(tasks[i]))

                signal_start += int(sample_rate * current_duration)
                k += 1

            # Add labels and data for the current run into the final output numpy arrays
            y.extend(labels)
            if cpu_format:
                if chunk_data:
                    # (num_trials_per_run, 64, 8, 80) => (num_trials_per_run, 64, 80, 8) => (num_trials_per_run, 8, 80, 64) => (120, 80, 64)
                    X.extend(trial_data.swapaxes(2, 3).swapaxes(1, 3).reshape((-1, samples_per_chunk, 64)))
                else:
                    # (num_trials_per_run, 64, 640) => (num_trials_per_run, 640, 64)
                    X.extend(trial_data.swapaxes(1, 2))
            else:
                if chunk_data:
                    # (num_trials_per_run, 64, 8, 80) => (num_trials_per_run, 8, 64, 80) => (120, 64, 80)
                    X.extend(trial_data.swapaxes(1, 2).reshape((-1, 64, samples_per_chunk)))
                else:
                    # (num_trials_per_run, 64, 640)
                    X.extend(trial_data)

    # Shape the final output arrays to the correct format
    X = np.stack(X)
    y = np.array(y).reshape((-1, 1))

    print("Loaded data shapes:")
    print(X.shape, y.shape)

    return X, y


