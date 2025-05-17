#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Karel Roots"

import time
from glob import glob
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
import copy
class SaveBestModel(Callback):
    def __init__(self):
        super().__init__()
        self.best_weights = None
        self.best_val_loss = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_weights = copy.deepcopy(self.model.get_weights())

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
def predict_accuracy(model, X_test, y_test, model_name, multi_branch):
    if multi_branch:
        probs = model.predict([X_test, X_test, X_test])
    else:
        probs = model.predict(X_test)

    preds = probs.argmax(axis=-1)
    equals = preds == y_test.argmax(axis=-1)
    acc = np.mean(equals)

    print("Classification accuracy for %s : %f " % (model_name, acc))

    return acc, equals
def train_test_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test, multi_branch, nr_of_epochs,test_model):

    print("######################### Model: " + model_name+" ###########################")

    callbacks_list = [SaveBestModel(),callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)]
    model.compile(loss=binary_crossentropy, optimizer=Adam(lr=0.001), metrics=['accuracy'])

    if multi_branch:
        model.fit([X_train, X_train, X_train], y_train, batch_size=64, shuffle=True, epochs=nr_of_epochs,
                            validation_data=([X_val, X_val, X_val], y_val), verbose=False, callbacks=callbacks_list)
    else:
        model.fit(X_train, y_train, batch_size=64, shuffle=True, epochs=nr_of_epochs, validation_data=(X_val, y_val), verbose=False, callbacks=callbacks_list)

    if test_model:
        acc, equals= predict_accuracy(model, X_test, y_test, model_name, multi_branch=multi_branch)
    return model, acc, equals
def run_experiment(X, y, experiment, use_cpu, test_model,seed):
    if use_cpu:
        K.set_image_data_format('channels_last')
    else:
        K.set_image_data_format('channels_first')

    test_split = experiment.get_test_split()
    val_split = experiment.get_val_split()

    ################################
    # X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_split, random_state=seed)
    # X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,test_size=val_split, random_state=seed)

    ################################
    # X_tail = X[-720:]
    # y_tail = y[-720:]

    # # Test indices (fixed)
    # test_indices = list(range(420, 480)) + list(range(540, 600)) + list(range(660, 720))
    # X_test = X_tail[test_indices]
    # y_test = y_tail[test_indices]

    # # Validation candidates (remaining of last 720) + everything before
    # val_indices = list(range(360, 420)) + list(range(480, 540)) + list(range(600, 660)) + list(range(0, 360))
    # X_val_candidates = X_tail[val_indices]
    # y_val_candidates = y_tail[val_indices]

    # # Combine with early training data
    # X_combined = np.concatenate([X[:-720], X_val_candidates])
    # y_combined = np.concatenate([y[:-720], y_val_candidates])

    # # Now randomly split combined data into new training and validation sets
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_combined, y_combined, test_size=0.2, random_state=42, shuffle=True
    # )
    ################################
    n_test = 72
    n_val = 72
    step = 72
    total_len = len(X)
    i=4
    n_iters=10
    start_test = i * step
    end_test = start_test + n_test

    # Normal case
    if i < n_iters - 1:
        start_val = end_test
        end_val = start_val + n_val
    else:
        # Last iteration: validation = first 72
        start_val = 0
        end_val = n_val

    # Assign test and validation sets
    X_test = X[start_test:end_test]
    y_test = y[start_test:end_test]

    X_val = X[start_val:end_val]
    y_val = y[start_val:end_val]

    # Mask test and val ranges to exclude from training
    mask = np.ones(total_len, dtype=bool)
    mask[start_test:end_test] = False
    mask[start_val:end_val] = False

    X_train = X[mask]
    y_train = y[mask]

    for model in experiment.get_models().values():
        
        _model = model.get_model()
        model_name = model.get_name() + '_' + experiment.get_exp_type()
        multi_branch = model.get_mb()

        _model, acc, equals = train_test_model(_model, model_name, X_train, y_train,
                                               X_val, y_val, X_test, y_test,
                                               multi_branch, experiment.get_epochs(),
                                               test_model)

        model.set_accuracy(acc)
        model.set_equals(equals)

    return experiment,acc
