# Import necessary modules:
from astropy.io import fits

import h5py
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import copy
import pickle
import os
import gc
    
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K

# ------------------------------------------------------------------------------------------------------------

# Flag to remove the young and old clusters:
flag_remove_extremes = False # Currently removing younger than 10^7 years and older than 10^9.5

# Flag to augment or not:
flag_use_augmented = False

# ------------------------------------------------------------------------------------------------------------
# Directories:

# Raw data:
dir_data_raw = "/pool001/vianajr/cluster_ages_1/data/data_raw/raw_phangs_dataset.h5"

# Results parent directory:
dir_results_parent = "/pool001/vianajr/cluster_ages_1/results/baseline/"

# Aux text:
txt_remextremes = "yes" if flag_remove_extremes else "no"
# Base prefix for the results directory:
results_prefix = f"baseline_remextremes_{txt_remextremes}"

# Create the folder for these results:
dir_top_results = dir_results_parent + results_prefix 

# Create if not exists:
if not os.path.exists(dir_top_results): os.makedirs(dir_top_results)

# ------------------------------------------------------------------------------------------------------------
# Display:

print()
print("--------------------------------------")
print("flag_remove_extremes: ", flag_remove_extremes)
print("--------------------------------------")
print()

# ------------------------------------------------------------------------------------------------------------
# Set the seed for reproducibility
random_seed = 15
random.seed(random_seed)
np.random.seed(random_seed)  # If you also want to ensure reproducibility with numpy functions

# ------------------------------------------------------------------------------------------------------------
# Class to read the dataset in the format it is:
class ReadPhangsH5:
    
    def __init__(self, hdf5_filename):
        # loading from hdf5 file
        with h5py.File(hdf5_filename, "r") as hf:
            
            # Get the cluster ID:
            self.cluster_ids = np.array(hf["cluster_ids"], dtype=np.int32)
            # Use astype(str) to correctly convert HDF5 string datasets to Python strings, for the galaxy_ids:
            self.galaxy_ids = np.array(hf["galaxy_ids"]).astype(str)
            # Get the image cutouts:
            self.image_cutouts = np.array(hf["image_cutouts"], dtype=np.float32)
            # Get the log of the ages:
            self.cluster_log_ages = np.array(hf["cluster_log_ages"], dtype=np.float32)

    def __getitem__(self, index):
        # Get the image cutouts for the instance (5 images)
        x = self.image_cutouts[index]

        # Take the mean of the 5 images along the first dimension (channel dimension)
        x_mean = np.mean(x, axis=0)  # Shape will now be (112, 112)

        # Get the log of the ages:
        y = self.cluster_log_ages[index]

        return x_mean, y

    def __len__(self):
        return len(self.image_cutouts)

    
# ------------------------------------------------------------------------------------------------------------
# Function to split the input (X) and output (Y) from the dataset
def separate_X_Y(dataset):
    X = [x for x, _ in dataset]
    Y = [y for _, y in dataset]
    return np.array(X), np.array(Y)


# ------------------------------------------------------------------------------------------------------------
# Define a function to split the data into tr, vl, and ts sets
def split_dataset(N, tr_ratio=0.7, vl_ratio=0.15, seed=42):
    
    # Initialize:
    random.seed(seed)
    indices = list(range(N))
    random.shuffle(indices)
    
    tr_split = int(tr_ratio * N)
    vl_split = int((tr_ratio + vl_ratio) * N)
    
    tr_indices = indices[:tr_split]
    vl_indices = indices[tr_split:vl_split]
    ts_indices = indices[vl_split:]
    
    return tr_indices, vl_indices, ts_indices


# ------------------------------------------------------------------------------------------------------------
# Custom median absolute error metric using numpy
def custom_median_absolute_error(dnrm_y_pred, dnrm_y_true):
    
    # Convert tensors to numpy arrays
    abs_diff = tf.abs(dnrm_y_true - dnrm_y_pred)
    # Use numpy to calculate the median and multiply by 1.49
    median_abs_diff = np.median(abs_diff.numpy())  # Convert tensor to numpy and compute the median
    return 1.49 * median_abs_diff


# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

# Start running the code

# Create an instance of the dataset class
raw_dataset = ReadPhangsH5(dir_data_raw)

# Get X and Y:
raw_X, raw_Y = separate_X_Y(raw_dataset)
# Display:
print("X.shape: ", raw_X.shape)
print()

# Get the list of cluster ids and galaxy ids:
raw_clust_ids = raw_dataset.cluster_ids
raw_galax_ids = raw_dataset.galaxy_ids
    
# Remove instances where Y is less than 7 or bigger than 9.5 if flag_remove_extremes is True
if flag_remove_extremes:
    
    # Mask:
    mask = (raw_Y >= 7) & (raw_Y <= 9.5)
    
    # Apply mask to get the curated sets:
    cur_X = raw_X[mask]
    cur_Y = raw_Y[mask]
    # Update the ids:
    cur_clust_ids = raw_clust_ids[mask]
    cur_galax_ids = raw_galax_ids[mask]
    
    # Display:
    print("New X.shape after removing young clusters: ", cur_X.shape)
    print()
    
# Otherwise simple assignation:
else:
    cur_X = raw_X
    cur_Y = raw_Y
    
# Get tr, vl, and ts indices
tr_indices, vl_indices, ts_indices = split_dataset(len(cur_Y))


# ------------------------------------------------------------------------------------------------------------
# Get the X_tr, X_vl, X_ts using the indices:
X_tr = cur_X[tr_indices]
X_vl = cur_X[vl_indices]
X_ts = cur_X[ts_indices]

# Get the Y_tr, Y_vl, Y_ts using the indices:
Y_tr = cur_Y[tr_indices]
Y_vl = cur_Y[vl_indices]
Y_ts = cur_Y[ts_indices]

# Display shapes:
print("Tr. set shapes: ", X_tr.shape, Y_tr.shape)
print("Vl. set shapes: ", X_vl.shape, Y_vl.shape)
print("Ts. set shapes: ", X_ts.shape, Y_ts.shape)
print()

# ------------------------------------------------------------------------------------------------------------
# Now get the mean output of the training set:
mean_train_output = np.mean(Y_tr)
print("Mean of training set (log age): ", mean_train_output)
print()

# ------------------------------------------------------------------------------------------------------------
# Use this mean as a prediction for all, the training, the validation and the test sets:
Y_pred_tr = np.full_like(Y_tr, mean_train_output)
Y_pred_vl = np.full_like(Y_vl, mean_train_output)
Y_pred_ts = np.full_like(Y_ts, mean_train_output)

# ------------------------------------------------------------------------------------------------------------
# Get the custom_median_absolute_error of each set:
dex_tr = custom_median_absolute_error(tf.constant(Y_pred_tr), tf.constant(Y_tr))
dex_vl = custom_median_absolute_error(tf.constant(Y_pred_vl), tf.constant(Y_vl))
dex_ts = custom_median_absolute_error(tf.constant(Y_pred_ts), tf.constant(Y_ts))

# Display errors:
print("Custom Median Absolute Error - Tr. set:", dex_tr)
print("Custom Median Absolute Error - Vl. set:", dex_vl)
print("Custom Median Absolute Error - Ts. set:", dex_ts)
print()

# ------------------------------------------------------------------------------------------------------------
# Save the custom_median_absolute_error of each set:
results_dex = {
    "dex_tr": dex_tr,
    "dex_vl": dex_vl,
    "dex_ts": dex_ts,
    "mean_train_output": mean_train_output
}

# Save results as pickle file
results_path = os.path.join(dir_top_results, "baseline_dex_results.pkl")
with open(results_path, "wb") as f:
    pickle.dump(results_dex, f)

print(f"Results saved at: {results_path}")
print("--------------------------------------")


