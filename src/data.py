
import os.path
import numpy as np
import h5py
import scipy.io

from .training.asyncloader import HDF5Concat

def load_dataset(base_path, filenames, heatmapfilenames, keys):
    full_filenames = [os.path.join(base_path, filename) for filename in filenames]
    files = [h5py.File(i, "r") for i in full_filenames]
    heatmap_full_filenames = [os.path.join(base_path, filename) for filename in heatmapfilenames]
    heatmap_files = [h5py.File(i, "r") for i in heatmap_full_filenames]
    datasets = []
    for key in keys[:2]:
        datasets.append(HDF5Concat([f[key] for f in files]))
    for key in keys[2:]:
        datasets.append(HDF5Concat([f[key] for f in heatmap_files]))
    return datasets

def load_training(train_ims, train_hms, keys):
    base_path = ""
    filenames = [train_ims]
    heatmap_filenames = [train_hms]
    return load_dataset(base_path, filenames, heatmap_filenames, keys)

def load_testing(val_ims, val_hms, keys):
    base_path = ""
    filenames = [val_ims]
    heatmap_filenames = [val_hms]
    return load_dataset(base_path, filenames, heatmap_filenames, keys)
