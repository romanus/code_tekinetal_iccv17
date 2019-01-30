
import ctypes
from collections import defaultdict
from itertools import groupby
import multiprocessing as mp
import time

import os.path
import h5py

import numpy as np

def divmod_splits(indices, splits):
    
    indices = np.asarray(indices)
    splits = np.asarray(splits)
    
    split_indices = np.searchsorted(splits, indices, side='right')
    elem_indices = indices
    elem_indices[split_indices != 0] -= splits[split_indices[split_indices != 0] - 1]
    
    return split_indices, elem_indices

def list_to_slices(data):
    slices = []
    for key, group in groupby(enumerate(data), lambda i: i[0] - i[1]):
        # group = map(itemgetter(1), group)
        group = [g[1] for g in group]
        slices.append(slice(group[0], group[-1] + 1))
    
    return slices

class HDF5Concat(object):
    
    def __init__(self, datasets):
        
        self.datasets = datasets
        self.sizes = list(map(len, self.datasets))
        self.cumsizes = np.cumsum(self.sizes)
        # print self.cumsizes[-1]
        # print self.datasets[0].shape
        self.shape = (self.cumsizes[-1], self.datasets[0].shape[1])
        self.size = np.prod(self.shape)
    
    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, indices):

        if isinstance(indices, slice):
            indices = range(self.cumsizes[-1])[indices]

        indices = np.asarray(indices)
        
        argsort = np.argsort(indices)
        inv_argsort = np.argsort(argsort)
        sorted_indices = indices[argsort]
        
        dataset_indices, elem_indices = divmod_splits(sorted_indices, self.cumsizes)
        
        dataset_to_elem_indices = defaultdict(list)
        for i, j in zip(dataset_indices, elem_indices):
            dataset_to_elem_indices[i].append(j)
        
        data = []
        for dataset_index, elem_indices in dataset_to_elem_indices.items():
            dataset = self.datasets[dataset_index]
            slices = list_to_slices(elem_indices)
            data.extend([dataset[slc] for slc in slices])
            # data.append(dataset[elem_indices])
        data = np.vstack(data)
        
        # data = []
        # for i, j in zip(dataset_indices, elem_indices):
        #     dataset = self.datasets[i]
        #     data.append(dataset[j])
        # data = np.array(data)
        
        return data[inv_argsort]
    
def _load(outmparrays, datasets, indices, lock):
    
    outarrays = [np.frombuffer(i.get_obj(), dtype=np.float32) for i in outmparrays]
    
    lock.acquire()
    for outarray, dataset in zip(outarrays, datasets):
        # dataset[indices]
        outarray[:] = dataset[indices].flatten()
    lock.release()
    
    return "Nothing to see here."

class DataAsyncLoader(object):
    
    def __init__(self, datasets):
        self.datasets = datasets
        self.lock = mp.Lock()
    
    def load_async(self, indices):
        
        if len(indices) == 0:
            return
        
        # print("Creating shared arrays...")
        shapes = [(len(indices), d.shape[1]) for d in self.datasets]
        mparrays = [mp.Array(ctypes.c_float, shape[0] * shape[1]) for shape in shapes]
        
        # print("Creating process...")
        process = mp.Process(target=_load, args=(mparrays, self.datasets, indices, self.lock))
        
        # print("Reading buffer...")
        results = [np.reshape(np.frombuffer(mparray.get_obj(), dtype=np.float32), shape)
                    for (mparray, shape) in zip(mparrays, shapes)]
        
        # print("Starting process...")
        process.start()
        
        return process, results
    
