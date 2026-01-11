# THE MAIN DIFFERENCE OF THIS FILE WITH THE 5 IMAGE CASE IS THAT HERE WE TAKE 2 AUTOENCODERS:
# FIRST AUTOENCODER COMPRESSES THE INFORMATION OF THE ENVIRONMENT
# SECOND AUTOENCODER COMPRESSES THE INFORMATION OF THE CLUSTER
# THEN WITH THE LATENT FEATURES OF BOTH WE MAKE THE INFERENCE

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

# Choose the normalization method:
norm_by = "five-images"  # "dataset", "filter, "five-images", single-image"
# normalizing by dataset or by filter makes the weird results because the points values are too different. 
# very small or very large.

# ------------------------------------------------------------------------------------------------------------
# Directories:

# Raw data:
dir_data_raw = "/pool001/vianajr/cluster_ages_1/data/data_raw/raw_phangs_dataset.h5"

# Results parent directory:
dir_results_parent = "/pool001/vianajr/cluster_ages_1/results/single_case/"

# Aux text:
txt_extremes = "yes" if flag_remove_extremes else "no"
txt_augment = "yes" if flag_use_augmented else "no"
# Base prefix for the results directory:
results_prefix = f"single_case_5imB_3models_remextremes_{txt_extremes}_augment_{txt_augment}_"


# Define the radius for the blacking:
R = 6

# Parameters for training:
patience = 30
batch_size = 16

arr_epochs_1_and_2 = [100, 10] # 100
arr_learn_rates_1_and_2 = [0.000001, 0.0000001]

arr_epochs_3 = [200] # 100
arr_learn_rates_3 = [0.0000001] # Good result with 0.000001
    
# Create the folder for these results:
dir_results = dir_results_parent + results_prefix + f"blackout_both_normby_{norm_by}/R_{R}/"
 
    
# Create if not exists:
if not os.path.exists(dir_results): os.makedirs(dir_results)

# ------------------------------------------------------------------------------------------------------------

# Number of models per case to get an average of errors:
num_models_per_case = 5

# Flag to plot preliminary data visualization:
flag_plot_data_viz = True

# ------------------------------------------------------------------------------------------------------------
# Display:

print()
print()
print("--------------------------------------")
print("SINGLE -------------------------------")
print("5-im case ----------------------------")
print()
print("Params:")
print()
print("flag_remove_extremes: ", flag_remove_extremes)
print("flag_use_augmented: ", flag_use_augmented)
print("R: ", R)
print()
print("Normalization by")
print(norm_by)
print()
print("batch_size: ", batch_size)
print("patience: ", patience)
print()
print("--------------------------------------")
print()
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

        # Get the log of the ages:
        y = self.cluster_log_ages[index]

        return x, y

    def __len__(self):
        return len(self.image_cutouts)

    
# ------------------------------------------------------------------------------------------------------------
# Function to split the input (X) and output (Y) from the dataset
def separate_X_Y(dataset):
    X = [x for x, _ in dataset]
    Y = [y for _, y in dataset]
    return np.array(X), np.array(Y)


# ------------------------------------------------------------------------------------------------------------
# Function to blackout a circle from the center:
def blackout(images, R, flag_black_inner):
    """
    Apply a circular mask blackout to the center of the images with radius R.
    The blackout will be an approximation since the images are 2D matrices.
    
    :param images: A numpy array of shape (n_samples, 5, 112, 112)
    :param R: Radius of the blackout circle (better if it is an odd number)
    :param flag_black_inner: If True we are removing the Inner center of the image, else the outer.
    :return: Modified images with the center blacked out
    """

    # Copy the input to avoid in-place modification
    images = images.copy()
    
    # Dimensions of the images
    n_samples, filters, height, width = images.shape # Here shape is: samples, filters, height, widht
    
    # Center of the images
    center_x, center_y = width // 2, height // 2  # For 111x111, this will be 55, 55
    
    # Create a mask with the same dimensions as the image
    y, x = np.ogrid[:height, :width]
    
    # Correct the radius to be applied in a "circle" like pattern
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= (R ** 2)
    
    # If we are not blacking out the inner, then we are the outer:
    if not flag_black_inner:
        mask = np.logical_not(mask)
        
    # Show the mask:
    # plt.imshow(mask)

    # Apply the mask to each image in the dataset
    for i in range(n_samples):
        for j in range(filters):
            images[i, j][mask] = 0  # Zero out the masked area
        
    return images


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
# Function to plot 5 heatmaps in a row with different colormaps
def plot_multiple_heatmaps(image_cutouts, num_rows):
    
    # Params:
    num_cols = 5  # Assuming we have 5 columns in the instances
      
    # Initialize the figure:
    plt.figure(figsize=(20, 4 * num_rows))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows))
    
    # List of colormaps
    colormaps = ["twilight_shifted", "terrain", "BrBG", "PRGn_r", "coolwarm"]
    
    for row in range(num_rows):
        # Loop through the slices of the instance:
        for i in range(num_cols):
            # Plot in the axes[row, i]
            sns.heatmap(image_cutouts[row, i, :, :], cmap=colormaps[i], cbar=False, ax=axes[row, i])
            axes[row, i].set_title(f"Image {i + 1}")
            axes[row, i].axis('off')  # Turn off axis

    plt.tight_layout()
    plt.savefig(dir_results + "1_heatmaps_indiv_instances.png")
    plt.show()
    
      
# ------------------------------------------------------------------------------------------------------------
def plot_multiple_histograms(image_cutouts, num_rows, suffix):
    
    num_cols = 5  # Assuming we are showing 5 images per instance
    plt.figure(figsize=(20, 4 * num_rows))
    
    # Initialize the figure:
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows))
    
    for row in range(num_rows):
        for i in range(num_cols):
            # Flatten the image to create a histogram of pixel values
            axes[row, i].hist(image_cutouts[row, i].flatten(), bins=50, color='purple', edgecolor='black', alpha=0.7)
            axes[row, i].set_title(f"Image {i + 1} Histogram")
            axes[row, i].set_xlabel("Pixel Value")
            axes[row, i].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(dir_results + f"2_histograms_indiv_instances_{suffix}.png")
    plt.show()

    
# ------------------------------------------------------------------------------------------------------------
# Function to plot age values across different datasets with fixed titles and colors
def plot_age_histograms(Y_data, suffix, bins=30):
    """
    Plots histograms of age values across different datasets with fixed titles and colors.
    
    Parameters:
    - Y_data: List of Y values (e.g., [Y_tr, Y_vl, Y_ts]).
    - bins: Number of bins for the histograms (default=30).
    """
    # Define fixed titles and colors
    titles = ['Training Set', 'Validation Set', 'Test Set']
    colors = ['blue', 'green', 'red']
    
    # Create subplots
    fig, axes = plt.subplots(1, len(Y_data), figsize=(15, 5))
    
    # Plot histograms
    for ax, Y, title, color in zip(axes, Y_data, titles, colors):
        ax.hist(Y, bins=bins, color=color, edgecolor='black', alpha=0.7)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Cluster Log Ages', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(dir_results + f"3_histograms_outputs_{suffix}.png")
    plt.show()
    
    
# ------------------------------------------------------------------------------------------------------------
# Function to augment a dataset by creating 8 versions 4 rotations and reversed 4 totations:
def augment_dataset_full(data):
    
    # The actual augmented data:
    aug_data = []
    # Reference for the past indexes that they corresponded to:
    aug_past_idxs = [] 
    
    for i, (x, y) in enumerate(data):

        # Original + 90° rotations
        for i in range(4):
            
            # Rotate 0°, 90°, 180°, 270°
            rot_x = np.array([ np.rot90(image, k=i) for image in x ])
            
            # Increase data:
            aug_data.append( (rot_x, y) )  
            aug_past_idxs.append(i)
        
        # Flip the image (up-down flip)
        flip_x = np.array([ np.flipud(image) for image in x ])
        
        # Flipped + 90° rotations
        for i in range(4):
            # Rotate flipped image
            rot_flip_x = np.array([ np.rot90(flip_image, k=i) for flip_image in flip_x ])
            # Increase data:
            aug_data.append( (rot_flip_x, y) )  
            aug_past_idxs.append(i)


    return aug_data, aug_past_idxs


# ------------------------------------------------------------------------------------------------------------
# Plotting function for visualizing the  data
def plot_first_instances(input_data, suffix, num_rows=4, num_cols=5):

    # Initialize the figure:
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4*num_rows))

    # Loop through each row (instance):
    for i in range(num_rows):
        # Get the data (x contains the 5 images, y_label is the label for that instance):
        x_ims, y_label = input_data[i]

        # Plot the 5 images for the current instance (i.e., for one instance)
        for j in range(num_cols):
            # Display the image (assuming x_ims contains 5 images, we display the j-th image)
            axes[i, j].imshow(x_ims[j], cmap="gray")  # Display j-th image in the row
            axes[i, j].axis('off')  # Hide axis

            # Add a title to each subplot showing the value of y
            axes[i, j].set_title(f"y = {y_label}", fontsize=16)

        # Add a row label for the instance
        axes[i, 0].set_ylabel(f'Instance {i+1}', fontsize=12, rotation=0, labelpad=80)

    # Ensure layout is tidy and titles/labels are visible
    plt.tight_layout()
    plt.savefig(dir_results + f"4_indiv_instances_post_augm_{suffix}.png")
    plt.show()


# ------------------------------------------------------------------------------------------------------------
# Function to create model with a single branch for all 5-channel images
def create_model_1_or_2(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='sigmoid'))
    model.add(layers.Dense(1))  # Single output for predicting log age
    return model

# Function to create model with a single branch for all 5-channel images
def create_model_3(input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))  # Input shape is now 256
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1))  # Single output for predicting log age
    return model
    
# ------------------------------------------------------------------------------------------------------------
# Results plots:

# Plot loss over epochs:
def plot_loss_over_epochs(loss, val_loss, which):
    plt.figure(figsize=(10, 6))
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title(f'Loss over Epochs {which}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(dir_results_model_k + f"5_losses_evolution_{which}.png")
    plt.show()
    
# Function to plot predicted vs true values
def plot_pred_vs_true(y_true, y_pred, title, which_set):

    # Initialize:
    plt.figure(figsize=(6, 6))
    
    # Scatter plot with alpha blending and small markers to match the look
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='none', color='steelblue', label='Cluster stamps', s=18)
    
    # Red dashed line for perfect prediction
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect prediction')
    
    # Labels and title
    plt.xlabel('True log(age) [yrs]', fontsize=12)
    plt.ylabel('Predicted log(age) [yrs]', fontsize=12)
    plt.title(title, fontsize=14)
    
    # Add a legend
    plt.legend(loc='upper left', fontsize=12)
    
    # Set axis limits to be square like the example
    plt.xlim([y_true.min(), y_true.max()])
    plt.ylim([y_true.min(), y_true.max()])
    
    # Add grid lines
    plt.grid(True)
    
    # Show the plot
    plt.tight_layout()
    plt.savefig(dir_results_model_k + f"6_pred_vs_true_{which_set}.png")
    plt.show()


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
    # Update the data:
    cur_X = raw_X
    cur_Y = raw_Y
    # Update the ids:
    cur_clust_ids = raw_clust_ids
    cur_galax_ids = raw_galax_ids

# Get tr, vl, and ts indices
tr_indices, vl_indices, ts_indices = split_dataset(len(cur_Y))

# ------------------------------------------------------------------------------------------------------------

# Here we divide the dataset in two groups:
# We differentiate the data between envir (surrounding environment) and centr (center of the cluster):
X_envir = blackout(cur_X.copy(), R, True)
X_centr = blackout(cur_X.copy(), R, False)

# ------------------------------------------------------------------------------------------------------------

# Access the data envir:
X_tr_envir = np.array([X_envir[i] for i in tr_indices])
X_vl_envir = np.array([X_envir[i] for i in vl_indices])
X_ts_envir = np.array([X_envir[i] for i in ts_indices])

# Access the data envir:
X_tr_centr = np.array([X_centr[i] for i in tr_indices])
X_vl_centr = np.array([X_centr[i] for i in vl_indices])
X_ts_centr = np.array([X_centr[i] for i in ts_indices])
     
# Access the data envir:
Y_tr = np.array([cur_Y[i] for i in tr_indices])
Y_vl = np.array([cur_Y[i] for i in vl_indices])
Y_ts = np.array([cur_Y[i] for i in ts_indices])


# Print the shapes of the data
print(f"Shape of X_tr_envir: {X_tr_envir.shape}")
print(f"Shape of X_vl_envir: {X_vl_envir.shape}")
print(f"Shape of X_ts_envir: {X_ts_envir.shape}")
print()
print(f"Shape of X_tr_centr: {X_tr_centr.shape}")
print(f"Shape of X_vl_centr: {X_vl_centr.shape}")
print(f"Shape of X_ts_centr: {X_ts_centr.shape}")
print()
print(f"Shape of Y_tr: {Y_tr.shape}")
print(f"Shape of Y_vl: {Y_vl.shape}")
print(f"Shape of Y_ts: {Y_ts.shape}")
print()

# Get the ids for the sets:
# Cluster ids:
tr_clust_ids = [cur_clust_ids[i] for i in tr_indices]
vl_clust_ids = [cur_clust_ids[i] for i in vl_indices]
ts_clust_ids = [cur_clust_ids[i] for i in ts_indices]
# Galaxy ids:
tr_galax_ids = [cur_galax_ids[i] for i in tr_indices]
vl_galax_ids = [cur_galax_ids[i] for i in vl_indices]
ts_galax_ids = [cur_galax_ids[i] for i in ts_indices]


# ------------------------------------------------------------------------------------------------------------
# Regardless of which set we use for training, for evaluation purposes we want to have two subsets of the data:
#
#     Inner: Data inside the limits, without the extremes (young and old stars)
#     Outer: Data in the extremes (excluding inner)

# Get the masks:
msk_subset_inner = (raw_Y >= 7) & (raw_Y <= 9.5)
msk_subset_outer = ~ msk_subset_inner

# First, you actually need to get the blacked X of all the instances:
treated_all_X_envir = blackout(raw_X.copy(), R, True)
treated_all_X_centr = blackout(raw_X.copy(), R, False)

# Apply the masks to treated_all_X:
subset_outer_X_envir = treated_all_X_envir[msk_subset_outer]
subset_inner_X_envir = treated_all_X_envir[msk_subset_inner]

subset_outer_X_centr = treated_all_X_centr[msk_subset_outer]
subset_inner_X_centr = treated_all_X_centr[msk_subset_inner]

# Apply the masks to raw_Y:
subset_outer_Y = raw_Y[msk_subset_outer]
subset_inner_Y = raw_Y[msk_subset_inner]

# Apply the masks to clust_ids:
subset_outer_clust_ids = raw_clust_ids[msk_subset_outer]
subset_inner_clust_ids = raw_clust_ids[msk_subset_inner]

# Apply the masks to galax_ids:
subset_outer_galax_ids = raw_galax_ids[msk_subset_outer]
subset_inner_galax_ids = raw_galax_ids[msk_subset_inner]

# CAREFUL: You must get the test points of each subset, you cannot pick any from the training set.
# We will proceed by creating a unique identifier of each instance, then seeing which are both in the ts and the subset.

# Function to create a unique identifier for each instance based on the cluster and the galaxy id:
def create_unique_ids(clust_ids, galax_ids):
    return [str(a) + b for a, b in zip(clust_ids, galax_ids)]

# Obtain the unique identifiers
ts_unique_ids = create_unique_ids(ts_clust_ids, ts_galax_ids)
subset_outer_unique_ids = create_unique_ids(subset_outer_clust_ids, subset_outer_galax_ids)
subset_inner_unique_ids = create_unique_ids(subset_inner_clust_ids, subset_inner_galax_ids)

# Find matching unique IDs
matching_ts_and_outer_ids = set(ts_unique_ids) & set(subset_outer_unique_ids)
matching_ts_and_inner_ids = set(ts_unique_ids) & set(subset_inner_unique_ids)

# If we are removing the extremes, then matching_ts_and_outer_ids is empty:
if flag_remove_extremes: matching_ts_and_outer_ids = set(subset_outer_unique_ids)
    
# Select 600 points from matching IDs - We specify 600 for the test set, to make sure all comparisons are fair among all cases:
matching_ts_and_outer_indices = np.random.choice(list(matching_ts_and_outer_ids), 600, replace=False)
matching_ts_and_inner_indices = np.random.choice(list(matching_ts_and_inner_ids), 600, replace=False)

# Create boolean masks
ts_msk_subset_outer = np.isin(subset_outer_unique_ids, matching_ts_and_outer_indices)
ts_msk_subset_inner = np.isin(subset_inner_unique_ids, matching_ts_and_inner_indices)

# Finally get the evaluation subsets of both X and Y:

# First the X:
ts_subset_outer_X_envir = subset_outer_X_envir[ts_msk_subset_outer]
ts_subset_inner_X_envir = subset_inner_X_envir[ts_msk_subset_inner]

ts_subset_outer_X_centr = subset_outer_X_centr[ts_msk_subset_outer]
ts_subset_inner_X_centr = subset_inner_X_centr[ts_msk_subset_inner]

# Then the Y, and don't forget to ravel the Ys:
ts_subset_outer_Y = np.ravel(subset_outer_Y[ts_msk_subset_outer])
ts_subset_inner_Y = np.ravel(subset_inner_Y[ts_msk_subset_inner])


# ------------------------------------------------------------------------------------------------------------
  
# If we are not augmenting, the variables we are using are directly these:

# X and Y:
use_X_tr_envir = X_tr_envir
use_X_tr_centr = X_tr_centr
use_Y_tr = np.ravel(Y_tr) 
# The ids:
use_tr_clust_ids = tr_clust_ids 
use_tr_galax_ids = tr_galax_ids

# The vl and ts Xs:
use_X_vl_envir = X_vl_envir
use_X_vl_centr = X_vl_centr
use_X_ts_envir = X_ts_envir
use_X_ts_centr = X_ts_centr

# The vl and ts Ys:
use_Y_vl = np.ravel(Y_vl)
use_Y_ts = np.ravel(Y_ts)
# The ids:
use_vl_clust_ids = vl_clust_ids
use_ts_clust_ids = ts_clust_ids
use_vl_galax_ids = vl_galax_ids
use_ts_galax_ids = ts_galax_ids


# Rearrange the shape of the inputs, so the channels/filters is the last dimension, now (n, 112, 112, 5)
use_X_tr_envir = np.stack([use_X_tr_envir[:, i, :, :] for i in range(5)], axis=-1)
use_X_tr_centr = np.stack([use_X_tr_centr[:, i, :, :] for i in range(5)], axis=-1)

use_X_vl_envir = np.stack([use_X_vl_envir[:, i, :, :] for i in range(5)], axis=-1)
use_X_vl_centr = np.stack([use_X_vl_centr[:, i, :, :] for i in range(5)], axis=-1)

use_X_ts_envir = np.stack([use_X_ts_envir[:, i, :, :] for i in range(5)], axis=-1)
use_X_ts_centr = np.stack([use_X_ts_centr[:, i, :, :] for i in range(5)], axis=-1)

ts_subset_outer_X_envir = np.stack([ts_subset_outer_X_envir[:, i, :, :] for i in range(5)], axis=-1)
ts_subset_outer_X_centr = np.stack([ts_subset_outer_X_centr[:, i, :, :] for i in range(5)], axis=-1)

ts_subset_inner_X_envir = np.stack([ts_subset_inner_X_envir[:, i, :, :] for i in range(5)], axis=-1)
ts_subset_inner_X_centr = np.stack([ts_subset_inner_X_centr[:, i, :, :] for i in range(5)], axis=-1)

# Print the shapes of the data
print(f"Shape of use_X_tr_envir: {use_X_tr_envir.shape}")
print(f"Shape of use_X_tr_centr: {use_X_tr_centr.shape}")
print()
print(f"Shape of use_X_vl_envir: {use_X_vl_envir.shape}")
print(f"Shape of use_X_vl_centr: {use_X_vl_centr.shape}")
print()
print(f"Shape of use_X_ts_envir: {use_X_ts_envir.shape}")
print(f"Shape of use_X_ts_centr: {use_X_ts_centr.shape}")
print()
print(f"Shape of use_Y_tr: {use_Y_tr.shape}")
print(f"Shape of use_Y_vl: {use_Y_vl.shape}")
print(f"Shape of use_Y_ts: {use_Y_ts.shape}")
print()

# ------------------------------------------------------------------------------------------------------------
# Data Normalization:



# For both the input normalization or standardization we could group the data differently before taking the
#
#     min & max:   for the normalization
#    mean & stdv:  for the standardization
#
# Below are all the possible groupings we can perform:

# "dataset":
#  ▢ ▢ ▢ ▢ ▢
#  ▢ ▢ ▢ ▢ ▢
#     ...
#  ▢ ▢ ▢ ▢ ▢

# "filter":
#     ▢
#     ▢
#    ...
#     ▢

# "five-images":
#  ▢ ▢ ▢ ▢ ▢

# "single-image":
#     ▢

# NOTE: In the cases of "dataset", "filter" and "five-images", we will use the medians of the min & max or mean & stdvs.
# NOTE: For the 1im case, we only have "dataset" and "single-image".

# ------------------------------------------------------------------------------------------
# 3. Normalize using the "five-images" min and max:
# THE ONLY AVAILABLE FOR THIS CODE:
if norm_by == "five-images":
        
    # Normalize using the "five-images" min and max:
    def get_mins_and_maxs_of_five_images(X):
        # Compute mins and maxs across all 5 filters for each instance (grouped across 5 filters)
        mins = np.min(X, axis=(1, 2, 3), keepdims=True)  # Shape (n, 1, 1, 1)
        maxs = np.max(X, axis=(1, 2, 3), keepdims=True)  # Shape (n, 1, 1, 1)
        return mins, maxs
    
    def norm_ins_by_five_images(X, mins, maxs):
        # Normalize each group of 5 filters together
        return (X - mins) / (maxs - mins)
    
    # Use for envir:
    mins_tr_envir, maxs_tr_envir = get_mins_and_maxs_of_five_images(use_X_tr_envir)
    mins_vl_envir, maxs_vl_envir = get_mins_and_maxs_of_five_images(use_X_vl_envir)
    mins_ts_envir, maxs_ts_envir = get_mins_and_maxs_of_five_images(use_X_ts_envir)
    mins_ts_subset_outer_envir, maxs_ts_subset_outer_envir = get_mins_and_maxs_of_five_images(ts_subset_outer_X_envir)
    mins_ts_subset_inner_envir, maxs_ts_subset_inner_envir = get_mins_and_maxs_of_five_images(ts_subset_inner_X_envir)
    
    nrm_X_tr_envir = norm_ins_by_five_images(use_X_tr_envir, mins_tr_envir, maxs_tr_envir)
    nrm_X_vl_envir = norm_ins_by_five_images(use_X_vl_envir, mins_vl_envir, maxs_vl_envir)
    nrm_X_ts_envir = norm_ins_by_five_images(use_X_ts_envir, mins_ts_envir, maxs_ts_envir)
    nrm_ts_subset_outer_X_envir = norm_ins_by_five_images(ts_subset_outer_X_envir, mins_ts_subset_outer_envir, maxs_ts_subset_outer_envir)
    nrm_ts_subset_inner_X_envir = norm_ins_by_five_images(ts_subset_inner_X_envir, mins_ts_subset_inner_envir, maxs_ts_subset_inner_envir)

    # Use for clust:
    mins_tr_centr, maxs_tr_centr = get_mins_and_maxs_of_five_images(use_X_tr_centr)
    mins_vl_centr, maxs_vl_centr = get_mins_and_maxs_of_five_images(use_X_vl_centr)
    mins_ts_centr, maxs_ts_centr = get_mins_and_maxs_of_five_images(use_X_ts_centr)
    mins_ts_subset_outer_centr, maxs_ts_subset_outer_centr = get_mins_and_maxs_of_five_images(ts_subset_outer_X_centr)
    mins_ts_subset_inner_centr, maxs_ts_subset_inner_centr = get_mins_and_maxs_of_five_images(ts_subset_inner_X_centr)
    
    nrm_X_tr_centr = norm_ins_by_five_images(use_X_tr_centr, mins_tr_centr, maxs_tr_centr)
    nrm_X_vl_centr = norm_ins_by_five_images(use_X_vl_centr, mins_vl_centr, maxs_vl_centr)
    nrm_X_ts_centr = norm_ins_by_five_images(use_X_ts_centr, mins_ts_centr, maxs_ts_centr)
    nrm_ts_subset_outer_X_centr = norm_ins_by_five_images(ts_subset_outer_X_centr, mins_ts_subset_outer_centr, maxs_ts_subset_outer_centr)
    nrm_ts_subset_inner_X_centr = norm_ins_by_five_images(ts_subset_inner_X_centr, mins_ts_subset_inner_centr, maxs_ts_subset_inner_centr)

# ------------------------------------------------------------------------------------------
# Error if we are using the filter or image option in 
else: 
    raise("Error, choose a valid normalizing method.")

# ------------------------------------------------------------------------------------------

print("nrm_X_tr_envir.shape")
print(nrm_X_tr_envir.shape)
print()
print("nrm_X_tr_centr.shape")
print(nrm_X_tr_centr.shape)
print()

# ------------------------------------------------------------------------------------------

# For the normalization or standardizaiton of the outputs there is only one way:

# Define normalization function for outputs
def norm_outs(Y, min_Y, max_Y):
    return (Y - min_Y) / (max_Y - min_Y)

# Function to denormalize outputs:
def denorm_outs(norm_Y, min_Y, max_Y):
    return norm_Y * (max_Y - min_Y) + min_Y
    

# Get the min and max of the Y values in the training set
min_Y_tr = np.min(use_Y_tr)
max_Y_tr = np.max(use_Y_tr)

# Normalize the outputs for training, validation, and test sets
nrm_Y_tr = norm_outs(use_Y_tr, min_Y_tr, max_Y_tr)
nrm_Y_vl = norm_outs(use_Y_vl, min_Y_tr, max_Y_tr)
nrm_Y_ts = norm_outs(use_Y_ts, min_Y_tr, max_Y_tr)

# ------------------------------------------------------------------------------------------------------------
# Prepare the inputs:
input_shape = (112, 112, 5)

# ------------------------------------------------------------------------------------------------------------
# Initialize model arrays of metrics:
all_tr_final_MSE_nrm = []
all_vl_final_MSE_nrm = []
all_ts_final_MSE_nrm = []
all_tr_final_MdAEs_dnrm = []
all_vl_final_MdAEs_dnrm = []
all_ts_final_MdAEs_dnrm = []

all_ts_subset_inner_MdAEs_dnrm = []
all_ts_subset_outer_MdAEs_dnrm = []

# ------------------------------------------------------------------------------------------------------------
# Model definition:

# Do these steps num_models_per_case times:
for k in range(num_models_per_case):

    # Print:
    print(), print(f"MODEL {k}"), print()
    
    # Cleaning space;
    if k != 0:
        # Delete the model and any large variables
        del history, nrm_Y_tr_pred, nrm_Y_vl_pred, nrm_Y_ts_pred
        del dnrm_Y_tr_pred, dnrm_Y_vl_pred, dnrm_Y_ts_pred
        # Force garbage collection to free memory
        gc.collect()

    # The directory of the model:
    dir_results_model_k = dir_results + f"model_{k}/"
    if not os.path.exists(dir_results_model_k): os.makedirs(dir_results_model_k)

    # Create the first two models
    model_1 = create_model_1_or_2(input_shape)
    model_2 = create_model_1_or_2(input_shape)
    
    # Create an optimizer with a learning rate of 0.00001
    optimizer_1 = tf.keras.optimizers.Adam(learning_rate=arr_learn_rates_1_and_2[0])
    optimizer_2 = tf.keras.optimizers.Adam(learning_rate=arr_learn_rates_1_and_2[0])

    # Compile the model with the custom metric
    model_1.compile(optimizer=optimizer_1, loss='mean_squared_error')
    model_2.compile(optimizer=optimizer_2, loss='mean_squared_error')

    print()
    print("Model 1")
    model_1.summary()
    
    print()
    print("Model 2")
    model_2.summary()
    
    # ------------------------------------------------------------------------------------------------------------
    # Model 1 and 2 training:

    # Define ReduceLROnPlateau callback
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience, verbose=1)
    # Define EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    # ------------------------------------------------------------------------------------------------------------

    # Initialize the losses:
    loss_model_1 = np.array([])
    val_loss_model_1 = np.array([])
    
    # Loop through the epochs:
    for epochs, lr in zip(arr_epochs_1_and_2, arr_learn_rates_1_and_2):
        
        # Set optimizer with the new learning rate
        tf.keras.backend.set_value(model_1.optimizer.lr, lr)
        # Train the model
        history = model_1.fit( nrm_X_tr_envir, nrm_Y_tr, 
                               validation_data=(nrm_X_vl_envir, nrm_Y_vl),
                               epochs=epochs, 
                               batch_size=batch_size,  # Modify if needed based on memory and dataset size
                               callbacks=[early_stopping],   # reduce_lr, early_stopping
                               verbose=0)
        # Append losses to single arrays
        loss_model_1 = np.append(loss_model_1, history.history['loss'])
        val_loss_model_1 = np.append(val_loss_model_1, history.history['val_loss'])
        
    print("Model 1 Training Finished"), print()

    
    # ------------------------------------------------------------------------------------------------------------

    # Initialize the losses:
    loss_model_2 = np.array([])
    val_loss_model_2 = np.array([])
    
    # Loop through the epochs:
    for epochs, lr in zip(arr_epochs_1_and_2, arr_learn_rates_1_and_2):
        
        # Set optimizer with the new learning rate
        tf.keras.backend.set_value(model_2.optimizer.lr, lr)
        # Train the model
        history = model_2.fit( nrm_X_tr_centr, nrm_Y_tr, 
                               validation_data=(nrm_X_vl_centr, nrm_Y_vl),
                               epochs=epochs, 
                               batch_size=batch_size,  # Modify if needed based on memory and dataset size
                               callbacks=[early_stopping],   # reduce_lr, early_stopping
                               verbose=0)
        # Append losses to single arrays
        loss_model_2 = np.append(loss_model_2, history.history['loss'])
        val_loss_model_2 = np.append(val_loss_model_2, history.history['val_loss'])
        
    print("Model 2 Training Finished"), print()

    # ------------------------------------------------------------------------------------------------------------

    # Plot loss over epochs:
    plot_loss_over_epochs(loss_model_1, val_loss_model_1, "Model_1")
    plot_loss_over_epochs(loss_model_2, val_loss_model_2, "Model_2")

    # ------------------------------------------------------------------------------------------------------------

    # Perform encoding of variables using previous two models:
    
    # For model_1 (Environment)
    encoder_model_1 = tf.keras.Model(inputs=model_1.input, outputs=model_1.layers[-2].output)
    encod_nrm_X_tr_envir = encoder_model_1.predict(nrm_X_tr_envir, verbose=0)
    encod_nrm_X_vl_envir = encoder_model_1.predict(nrm_X_vl_envir, verbose=0)
    encod_nrm_X_ts_envir = encoder_model_1.predict(nrm_X_ts_envir, verbose=0)
    
    # For model_2 (Cluster Center)
    encoder_model_2 = tf.keras.Model(inputs=model_2.input, outputs=model_2.layers[-2].output)
    encod_nrm_X_tr_centr = encoder_model_2.predict(nrm_X_tr_centr, verbose=0)
    encod_nrm_X_vl_centr = encoder_model_2.predict(nrm_X_vl_centr, verbose=0)
    encod_nrm_X_ts_centr = encoder_model_2.predict(nrm_X_ts_centr, verbose=0)

    # ------------------------------------------------------------------------------------------------------------
    # Concatenate the encoded features
    def concatenate_encoded_features(encoder_output_1, encoder_output_2):
        return np.concatenate([encoder_output_1, encoder_output_2], axis=1)
    
    # Concatenate train, validation, and test sets
    mrg_encod_nrm_X_tr = concatenate_encoded_features(encod_nrm_X_tr_envir, encod_nrm_X_tr_centr)
    mrg_encod_nrm_X_vl = concatenate_encoded_features(encod_nrm_X_vl_envir, encod_nrm_X_vl_centr)
    mrg_encod_nrm_X_ts = concatenate_encoded_features(encod_nrm_X_ts_envir, encod_nrm_X_ts_centr)
    
    print("Concatenated Train Shape:", mrg_encod_nrm_X_tr.shape)
    print("Concatenated Validation Shape:", mrg_encod_nrm_X_vl.shape)
    print("Concatenated Test Shape:", mrg_encod_nrm_X_ts.shape)

    # ------------------------------------------------------------------------------------------------------------

        
    # Now train a final model that uses the features that encode the previous two models:

    # Create the third model
    model_3 = create_model_3(256) # 128 x 2
    
    # Create an optimizer with a learning rate of 0.00001
    optimizer_3 = tf.keras.optimizers.Adam(learning_rate=arr_learn_rates_3[0])

    # Compile the model with the custom metric
    model_3.compile(optimizer=optimizer_3, loss='mean_squared_error')

    print()
    print("Model 3")
    model_3.summary()

    # Initialize the losses:
    loss_model_3 = np.array([])
    val_loss_model_3 = np.array([])
    
    # Loop through the epochs:
    for epochs, lr in zip(arr_epochs_3, arr_learn_rates_3):
        
        # Set optimizer with the new learning rate
        tf.keras.backend.set_value(model_3.optimizer.lr, lr)
        # Train the model
        history = model_3.fit( mrg_encod_nrm_X_tr, nrm_Y_tr, 
                               validation_data=(mrg_encod_nrm_X_vl, nrm_Y_vl),
                               epochs=epochs, 
                               batch_size=batch_size,  # Modify if needed based on memory and dataset size
                               callbacks=[early_stopping],   # reduce_lr, early_stopping
                               verbose=0)
        # Append losses to single arrays
        loss_model_3 = np.append(loss_model_3, history.history['loss'])
        val_loss_model_3 = np.append(val_loss_model_3, history.history['val_loss'])
        
    print("Model 3 Training Finished"), print()
    
    # ------------------------------------------------------------------------------------------------------------

    # Save all metric variables in a pickle file
    with open(dir_results_model_k + 'loss_data.pkl', 'wb') as f:
        pickle.dump({'tr_MSE_nrm': loss_model_3, 'vl_MSE_nrm': val_loss_model_3}, f)

    # Plot loss over epochs:
    plot_loss_over_epochs(loss_model_3, val_loss_model_3, "Model_3")

    # ------------------------------------------------------------------------------------------------------------
    # True vs Pred plots:

    # Predict the values for each dataset, and ravel to make sure is one dimensional:
    nrm_Y_tr_pred_3 = np.ravel( model_3.predict(mrg_encod_nrm_X_tr, verbose=0) )
    nrm_Y_vl_pred_3 = np.ravel( model_3.predict(mrg_encod_nrm_X_vl, verbose=0) )
    nrm_Y_ts_pred_3 = np.ravel( model_3.predict(mrg_encod_nrm_X_ts, verbose=0) )

    # Denormalize:
    dnrm_Y_tr_pred_3 = denorm_outs(nrm_Y_tr_pred_3, min_Y_tr, max_Y_tr)
    dnrm_Y_vl_pred_3 = denorm_outs(nrm_Y_vl_pred_3, min_Y_tr, max_Y_tr)
    dnrm_Y_ts_pred_3 = denorm_outs(nrm_Y_ts_pred_3, min_Y_tr, max_Y_tr)

    # Example usage for the training set
    plot_pred_vs_true(use_Y_tr, dnrm_Y_tr_pred_3, "Predictions vs True Ages (Tr. Set)", "tr")

    # Example usage for the validation set
    plot_pred_vs_true(use_Y_vl, dnrm_Y_vl_pred_3, "Predictions vs True Ages (Vl. Set)", "vl")

    # Example usage for the test set
    plot_pred_vs_true(use_Y_ts, dnrm_Y_ts_pred_3, "Predictions vs True Ages (Ts. Set)", "ts")

    # Most importantly, save the trues and the predicted:
    with open(dir_results_model_k + 'preds_and_trues.pkl', 'wb') as f:
        pickle.dump({
            'dnrm_Y_tr_pred': dnrm_Y_tr_pred_3,
            'dnrm_Y_vl_pred': dnrm_Y_vl_pred_3,
            'dnrm_Y_ts_pred': dnrm_Y_ts_pred_3,
    
            'use_Y_tr': use_Y_tr,
            'use_Y_vl': use_Y_vl,
            'use_Y_ts': use_Y_ts,
        }, f)
    
    # ------------------------------------------------------------------------------------------------------------

    # Evaluate the model on the test data
    test_loss = model_3.evaluate(mrg_encod_nrm_X_ts, nrm_Y_ts, verbose=0)

    # Append the last losses to the array of losses as a function of the R:
    all_tr_final_MSE_nrm.append(loss_model_3[-1])
    all_vl_final_MSE_nrm.append(val_loss_model_3[-1])
    all_ts_final_MSE_nrm.append(test_loss)

    # Print the final MSE metrics for reference
    print(f"Tr. Final MSE (nrm. units): {all_tr_final_MSE_nrm[-1]}")
    print(f"Vl. Final MSE (nrm. units): {all_vl_final_MSE_nrm[-1]}")
    print(f"Ts. Final MSE (nrm. units): {all_ts_final_MSE_nrm[-1]}")

    # Calculate the metrics for the sets in a denormalized scale:
    tr_final_MdAEs_dnrm = custom_median_absolute_error(dnrm_Y_tr_pred_3, use_Y_tr)
    vl_final_MdAEs_dnrm = custom_median_absolute_error(dnrm_Y_vl_pred_3, use_Y_vl)
    ts_final_MdAEs_dnrm = custom_median_absolute_error(dnrm_Y_ts_pred_3, use_Y_ts)

    # Append the results:
    all_tr_final_MdAEs_dnrm.append(tr_final_MdAEs_dnrm)
    all_vl_final_MdAEs_dnrm.append(vl_final_MdAEs_dnrm)
    all_ts_final_MdAEs_dnrm.append(ts_final_MdAEs_dnrm)

    # Print the final metrics for reference
    print(f"Tr. Final MdAE (dnrm. units): {all_tr_final_MdAEs_dnrm[-1]}")
    print(f"Vl. Final MdAE (dnrm. units): {all_vl_final_MdAEs_dnrm[-1]}")
    print(f"Ts. Final MdAE (dnrm. units): {all_ts_final_MdAEs_dnrm[-1]}")

    # Concatenate subsets 1 and 2 each independently
    mrg_nrm_ts_subset_outer_X = concatenate_encoded_features(nrm_ts_subset_outer_X_envir, nrm_ts_subset_outer_X_centr)
    mrg_nrm_ts_subset_inner_X = concatenate_encoded_features(nrm_ts_subset_inner_X_envir, nrm_ts_subset_inner_X_centr)

    # For the subsets (Outer and Inner)
    mrg_encod_ts_subset_outer = concatenate_encoded_features( encoder_model_1.predict(nrm_ts_subset_outer_X_envir, verbose=0),
                                                              encoder_model_2.predict(nrm_ts_subset_outer_X_centr, verbose=0) )

    mrg_encod_ts_subset_inner = concatenate_encoded_features( encoder_model_1.predict(nrm_ts_subset_inner_X_envir, verbose=0),
                                                              encoder_model_2.predict(nrm_ts_subset_inner_X_centr, verbose=0))

    # Evaluate the model on the subsets 1 and 2:
    nrm_Y_ts_subset_outer_pred_3 = np.ravel( model_3.predict(mrg_encod_ts_subset_outer, verbose=0) )
    nrm_Y_ts_subset_inner_pred_3 = np.ravel( model_3.predict(mrg_encod_ts_subset_inner, verbose=0) )
    
    dnrm_Y_ts_subset_outer_pred_3 = denorm_outs(nrm_Y_ts_subset_outer_pred_3, min_Y_tr, max_Y_tr)
    dnrm_Y_ts_subset_inner_pred_3 = denorm_outs(nrm_Y_ts_subset_inner_pred_3, min_Y_tr, max_Y_tr)
    
    ts_subset_outer_MdAEs_dnrm = custom_median_absolute_error(dnrm_Y_ts_subset_outer_pred_3, ts_subset_outer_Y)
    ts_subset_inner_MdAEs_dnrm = custom_median_absolute_error(dnrm_Y_ts_subset_inner_pred_3, ts_subset_inner_Y)

    # Append results of the evaluation subsets:
    all_ts_subset_outer_MdAEs_dnrm.append( ts_subset_outer_MdAEs_dnrm )
    all_ts_subset_inner_MdAEs_dnrm.append( ts_subset_inner_MdAEs_dnrm )
    
    # Save the computed metrics in a pickle file
    with open(dir_results_model_k + 'final_metrics.pkl', 'wb') as f:
        pickle.dump({
            'tr_final_MSE_nrm': all_tr_final_MSE_nrm[-1],
            'vl_final_MSE_nrm': all_vl_final_MSE_nrm[-1],
            'ts_final_MSE_nrm': all_ts_final_MSE_nrm[-1],   

            'tr_final_MdAEs_dnrm': all_tr_final_MdAEs_dnrm[-1],
            'vl_final_MdAEs_dnrm': all_vl_final_MdAEs_dnrm[-1],
            'ts_final_MdAEs_dnrm': all_ts_final_MdAEs_dnrm[-1],

            'ts_subset_outer_MdAEs_dnrm': ts_subset_outer_MdAEs_dnrm,
            'ts_subset_inner_MdAEs_dnrm': ts_subset_inner_MdAEs_dnrm,
        }, f)


# ------------------------------------------------------------------------------------------------------------

# After all the models have been visited, save the arrays:
with open(dir_results + 'all_models_final_metrics.pkl', 'wb') as f:
    pickle.dump({
        'all_tr_final_MSE_nrm': all_tr_final_MSE_nrm,
        'all_vl_final_MSE_nrm': all_vl_final_MSE_nrm,
        'all_ts_final_MSE_nrm': all_ts_final_MSE_nrm,   

        'all_tr_final_MdAEs_dnrm': all_tr_final_MdAEs_dnrm,
        'all_vl_final_MdAEs_dnrm': all_vl_final_MdAEs_dnrm,
        'all_ts_final_MdAEs_dnrm': all_ts_final_MdAEs_dnrm,

        'all_ts_subset_outer_MdAEs_dnrm': all_ts_subset_outer_MdAEs_dnrm,
        'all_ts_subset_inner_MdAEs_dnrm': all_ts_subset_inner_MdAEs_dnrm,
    }, f)


# ------------------------------------------------------------------------------------------------------------

# Plot these results:

# Scatter plot for MSE (normalized) and MdAE (de-normalized)
plt.figure(figsize=(10, 6))

# Plot:
plt.scatter(all_tr_final_MSE_nrm, all_tr_final_MdAEs_dnrm, label='Train', color='blue', marker='o')
plt.scatter(all_vl_final_MSE_nrm, all_vl_final_MdAEs_dnrm, label='Validation', color='green', marker='o')
plt.scatter(all_ts_final_MSE_nrm, all_ts_final_MdAEs_dnrm, label='Test', color='red', marker='o')

# Adding labels and title
plt.xlabel('MSE (Normalized)')
plt.ylabel('MdAE (De-normalized)')
plt.title('MSE vs MdAE for Train, Validation, and Test Sets')
plt.legend()
plt.grid(True)

# Save:
plt.savefig(dir_results + 'all_models_final_metrics_scatterplot.png')

