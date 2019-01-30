import theano
import logging
import itertools
import sys
import cv2
import numpy as np
from .asyncloader import DataAsyncLoader
import theano.tensor as tt

logger = logging.getLogger(__name__)

_rng = np.random.RandomState()

class AsyncLoaderSampler(object):
    def __init__(self,
                 datasets,
                 minibatch_size,
                 loadblock_size,  # How many minibatches to preload asynchronously
                 minibatch_shapes=[],
                 training=1,
                 subsampling=4,
                 rng=np.random.RandomState()):

        self.rng = rng
        self.pad = 16
        self.minibatch_size = minibatch_size
        self.loadblock_size = loadblock_size
        self.minibatch_shapes = minibatch_shapes
        self.im_height = minibatch_shapes[0][2] 
        self.im_width = minibatch_shapes[0][3] 
        self.hm_height = minibatch_shapes[2][2] 
        self.hm_width  = minibatch_shapes[2][3] 
        self.height_da = self.im_height - self.pad # height after data augmentation
        self.width_da = self.im_width - self.pad # width after data augmentation

        self.loader = DataAsyncLoader(datasets)

        self.sampling = subsampling
        self.iters_per_epoch = len(datasets[0]) // self.minibatch_size
        self.num_samples = self.iters_per_epoch * self.minibatch_size
        self.num_batches = self.iters_per_epoch // self.sampling
        self.indices = np.arange(self.num_samples)
        self.training = training

        self.blocks = {}

    def get_minibatch(self, index, data_augmentation=1):

        minibatch_index = index % self.iters_per_epoch
        block_out_index, block_in_index = divmod(minibatch_index, self.loadblock_size)
        self.load_blocks([block_out_index + i for i in range(3)])
        process, block = self.blocks[block_out_index]
        process.join(timeout=1)
        if process.is_alive():
            process.join()

        block_slice = slice(block_in_index * self.minibatch_size, (block_in_index + 1) * self.minibatch_size)

        minibatch = tuple(i[block_slice] for i in block)

        for i, sh in zip(minibatch, self.minibatch_shapes):
            if sh is not None:
                i.shape = sh

        current_x = minibatch[0]
        current_y = minibatch[1]
        current_heatmap = minibatch[2]

        if data_augmentation == 1:
            current_x_da = np.empty([self.minibatch_size, 3, self.height_da, self.width_da], dtype='float32')
            current_heatmap_da = np.empty([self.minibatch_size, 16, self.height_da, self.width_da], dtype='float32')

            current_x = np.reshape(current_x, self.minibatch_shapes[0])
            current_heatmap = np.reshape(current_heatmap, self.minibatch_shapes[2])

            if self.training and np.random.rand(1)[0] > 0.3:
                # Random crop
                x, y = np.random.randint(self.height_da+2, self.im_height-2, 2) - self.height_da
                for i in range(self.minibatch_size):
                    temp_im = np.transpose(
                        np.asarray(cv2.resize(np.transpose(current_x[i, :, :, :], (1, 2, 0)), (self.im_height, self.im_width)),
                                   dtype=theano.config.floatX), (2, 0, 1))
                    current_x_da[i, :, :, :] = temp_im[:, x:x+self.height_da, y:y+self.width_da]

                    temp_hm = np.transpose(
                        np.asarray(cv2.resize(np.transpose(current_heatmap[i, :, :, :], (1, 2, 0)), (self.im_height, self.im_width)),
                                   dtype=theano.config.floatX), (2, 0, 1))
                    current_heatmap_da[i, :, :, :] = temp_hm[:, x:x+self.height_da, y:y+self.width_da]
            else:
                # Scaling
                for i in range(self.minibatch_size):
                    current_x_da[i, :, :, :] = np.transpose(
                        np.asarray(cv2.resize(np.transpose(current_x[i, :, :, :], (1, 2, 0)), (self.height_da, self.width_da)),
                                   dtype=theano.config.floatX), (2, 0, 1))
                    current_heatmap_da[i, :, :, :] = np.transpose(
                        np.asarray(cv2.resize(np.transpose(current_heatmap[i, :, :, :], (1, 2, 0)), (self.height_da, self.width_da)),
                                   dtype=theano.config.floatX), (2, 0, 1))
            return np.float32(current_x_da), np.float32(current_heatmap_da), np.float32(current_y)

        return np.float32(current_x), np.float32(current_heatmap), np.float23(current_y)

    def load_blocks(self, block_indices):

        for key in list(self.blocks.keys()):
            if key not in block_indices:
                # logger.debug("Unloading block {}...".format(key))
                del self.blocks[key]

        for block_index in block_indices:
            if block_index in self.blocks:
                continue

            block_elements = self.get_indices_of_block_elements(block_index)
            if len(block_elements) == 0:
                continue

            self.blocks[block_index] = self.loader.load_async(block_elements)

    def get_indices_of_minibatch_elements(self, minibatch_index):
        return self.indices[minibatch_index * self.minibatch_size: (minibatch_index + 1) * self.minibatch_size]

    def get_indices_of_block_elements(self, block_index):

        elements_in_block = self.minibatch_size * self.loadblock_size
        return self.indices[block_index * elements_in_block: (block_index + 1) * elements_in_block]

    def shuffle(self):

        indices = np.arange(self.num_samples)
        grouped_indices = np.reshape(indices, (self.iters_per_epoch, self.minibatch_size))

        # Shuffle minibatches
        aux = np.arange(len(grouped_indices))
        self.rng.shuffle(aux)
        grouped_indices = grouped_indices[aux]

        indices = grouped_indices.flatten()

        # Group in blocks
        n = self.minibatch_size * self.loadblock_size
        grouped_indices = [indices[i: i + n] for i in range(0, len(indices), n)]

        # Shuffle every block
        for i in grouped_indices:
            self.rng.shuffle(i)

        self.indices = np.hstack(grouped_indices)

    def save_state(self, filename):
        pass

    def load_state(self, filename):
        self.shuffle()

    def iterable(self):
        for i in range(0,self.iters_per_epoch,self.sampling):
            yield self.get_minibatch(i)

    def num_minibatches(self):
        return self.iters_per_epoch // self.sampling


