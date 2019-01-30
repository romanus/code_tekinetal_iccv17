
import os.path
from collections import defaultdict
from functools import reduce

import numpy as np

class Summary(object):
    
    def __init__(self):
        
        self.content = defaultdict(dict)
    
    def register(self, tag, index, value):
        
        self.content[tag][index] = value
    
    def get(self, tag):
        
        if not self.content.has_key(tag):
            raise KeyError(tag)
        
        data = self.content[tag]
        
        indices = []
        values = []
        for index in sorted(data):
            indices.append(index)
            values.append(data[index])
        
        return np.asarray(indices), np.asarray(values)
    
    def get_many(self, tags):
        
        dicts = [self.content[tag] for tag in tags]
        indices = [d.keys() for d in dicts]
        indices = reduce(np.intersect1d, indices)
        indices = sorted(indices)
        
        results = tuple([] for tag in tags)
        for index in indices:
            for d, res in zip(dicts, results):
                res.append(d[index])
        
        return indices, results
    
    def save(self, filename, backup=False):
        
        if backup and os.path.isfile(filename):
            os.rename(filename, filename + ".bak")
        
        np.save(filename, self.content)
    
    def load(self, filename):
        self.content = np.load(filename).item()
    
