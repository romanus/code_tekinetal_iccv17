
import numpy as np

import theano
import theano.tensor as tt
from theano.tensor import nnet
from itertools import chain

from . import netlayers

def connect_many(layers, input, intermediate_outputs=False):
    cur_output = input
    outputs = []

    for layer in layers:
        cur_output = layer.connect(cur_output).output
        outputs.append(cur_output)

    if intermediate_outputs:
        return cur_output, outputs

    return cur_output

class FusionNet(object):
    
    def __init__(self, type=1, ibeta=None, reg_factor=1000):
        
        self.x1 = tt.tensor4('x1')
        self.x2 = tt.tensor4('x2')
        self.y = tt.matrix('y')
        
        self.modules1, self.testing_modules1 = self._create_layers(False, 3, type)
        self.modules2, self.testing_modules2 = self._create_layers(False, 16, type)
        self.modules_fusion, self.testing_modulesf = self._create_layers(True, 19, type)
        
        # Fusion control
        init_alpha = np.float32(0.1)
        # init_alpha = np.float32(1)
        self.alpha = theano.shared(init_alpha)
        if ibeta is not None:
            self.beta = theano.shared(np.float32(ibeta+0.6))
        else:
            self.beta = theano.shared(np.float32(len(self.modules1) / 2 + 0.6))
        # self.beta = theano.shared(np.float32(4.5))
        self.weights = [nnet.sigmoid(self.alpha*self.alpha*(i - self.beta)) for i in range(len(self.modules1))]
        
        self.params = []
        # List the params
        # Last modules of streams 1 and 2 are omitted since they do not contribute to the output
        for layers in [self.modules1[:-1], self.modules2[:-1], self.modules_fusion]:
            self.params.extend(chain(*(layer.params for sublayers in layers for layer in sublayers)))
        self.params.extend([self.alpha, self.beta])
        
        self.output = self._connect_streams(self.modules1, self.modules2, self.modules_fusion,
                                            self.x1, self.x2,
                                            self.weights)
        self.testing_output = self._connect_streams(self.testing_modules1, self.testing_modules2, self.testing_modulesf,
                                            self.x1, self.x2,
                                            self.weights)

        # Cost
        self.residuals3d = self.output - self.y
        sqrdiff3d = tt.sqr(self.residuals3d)
        self.cost = 0.5 * tt.mean(tt.sum(sqrdiff3d, axis=1))

        # Error
        self.residuals3d_test = self.testing_output - self.y
        sqrdiff3d_test = tt.sqr(self.residuals3d_test)
        num_samples = sqrdiff3d_test.shape[0]
        num_joints = sqrdiff3d_test.shape[1] / 3
        sqrdiff3d_test = tt.reshape(sqrdiff3d_test, (num_samples, num_joints, 3))
        self.error = tt.mean(tt.sqrt(tt.sum(sqrdiff3d_test, 2)))

        self.grads = theano.grad(self.cost + reg_factor * 1.0/(self.alpha*self.alpha), self.params)

        self.compute_output = theano.function([self.x1, self.x2], self.output, allow_input_downcast=True)
        self.compute_error = theano.function([self.x1, self.x2, self.y], self.error, allow_input_downcast=True)
        self.compute_cost = theano.function([self.x1, self.x2, self.y], self.cost, allow_input_downcast=True)
        self.get_weights = theano.function([], self.weights)
        self.get_alpha = theano.function([], self.alpha)
        self.get_beta = theano.function([], self.beta)
        self.compute_testing_output = theano.function([self.x1, self.x2], self.testing_output, allow_input_downcast=True)
    
    @staticmethod
    def _connect_streams(modules1, modules2, modules_fusion, input1, input2, weights):
        
        cur_output1 = input1
        cur_output2 = input2
        cur_outputf = 0
        
        for layers1, layers2, layersf, weight in zip(modules1, modules2, modules_fusion, weights):
            concat_output = tt.concatenate([cur_output1, cur_output2], axis=1)
            cur_inputf = weight * cur_outputf + (1 - weight) * concat_output
            cur_output1 = connect_many(layers1, cur_output1)
            cur_output2 = connect_many(layers2, cur_output2)
            cur_outputf = connect_many(layersf, cur_inputf)
        
        return cur_outputf


    @staticmethod
    def _create_layers(fusion_stream, input_channels, type=0):
        
        m = 1
        if fusion_stream:
            m = 2
        
        if type == 1:
            flag_on = 1
            training_modules = [
                [netlayers.Conv2dLayer2((36 * m, input_channels, 9, 9)),
                netlayers.MaxPool2dLayer2((2, 2)),
                netlayers.LambdaLayer(nnet.relu)],

                [netlayers.Conv2dLayer2((72 * m, 36 * m, 5, 5)),
                netlayers.MaxPool2dLayer2((2, 2)),
                netlayers.LambdaLayer(nnet.relu)],

                [netlayers.Conv2dLayer2((72 * m, 72 * m, 5, 5)),
                netlayers.MaxPool2dLayer2((2, 2)),
                netlayers.LambdaLayer(nnet.relu),
                netlayers.LambdaLayer(lambda x: tt.reshape(x, (x.shape[0], -1)))],

                [netlayers.InnerProductLayer2(72 * m * 10 * 10, 2048 * m), 
                netlayers.LambdaLayer(nnet.relu),
                netlayers.DropoutLayer(0.5, flag_on=flag_on)],

                [netlayers.InnerProductLayer2(2048 * m, 2048 * m), 
                netlayers.LambdaLayer(nnet.relu),
                netlayers.DropoutLayer(0.5, flag_on=flag_on)],

                [netlayers.InnerProductLayer(2048 * m, 512 * m),
                netlayers.LambdaLayer(nnet.relu)],

                [netlayers.InnerProductLayer(512 * m, 512 * m),
                 netlayers.LambdaLayer(nnet.relu)],

                [netlayers.InnerProductLayer(512 * m, 51)]
            ]

        elif type == 2:
            flag_on = 1
            training_modules = [
                [netlayers.Conv2dLayer2((36 * m, input_channels, 9, 9)),
                 netlayers.MaxPool2dLayer2((2, 2)),
                 netlayers.LambdaLayer(nnet.relu)],

                [netlayers.Conv2dLayer2((72 * m, 36 * m, 5, 5)),
                 netlayers.MaxPool2dLayer2((2, 2)),
                 netlayers.LambdaLayer(nnet.relu)],

                [netlayers.Conv2dLayer2((72 * m, 72 * m, 5, 5)),
                 netlayers.MaxPool2dLayer2((2, 2)),
                 netlayers.LambdaLayer(nnet.relu),
                 netlayers.LambdaLayer(lambda x: tt.reshape(x, (x.shape[0], -1)))],

                [netlayers.InnerProductLayer2(72 * m * 10 * 10, 512 * m),
                 netlayers.LambdaLayer(nnet.relu),
                 netlayers.DropoutLayer(0.5, flag_on=flag_on)],

                [netlayers.InnerProductLayer2(512 * m, 2048 * m),
                 netlayers.LambdaLayer(nnet.relu),
                 netlayers.DropoutLayer(0.5, flag_on=flag_on)],

                [netlayers.InnerProductLayer2(2048 * m, 4096 * m),
                 netlayers.LambdaLayer(nnet.relu),
                 netlayers.DropoutLayer(0.5, flag_on=flag_on)],

                [netlayers.InnerProductLayer(4096 * m, 512 * m),
                 netlayers.LambdaLayer(nnet.relu)],

                [netlayers.InnerProductLayer(512 * m, 512 * m),
                 netlayers.LambdaLayer(nnet.relu)],

                [netlayers.InnerProductLayer(512 * m, 51)]
            ]

        # Testing modules are the same as training modules removing Dropout layers.
        testing_modules = [[layer if not isinstance(layer, netlayers.DropoutLayer) else netlayers.DropoutLayer(0.5, flag_on=0)  for layer in layers]
                               for layers in training_modules]

        return training_modules, testing_modules

    def save_state(self, filename):
        state = self.__getstate__()  # Remove the config from the state
        np.savez(filename, **state)

    def load_state(self, filename):
        state = np.load(filename)

        if isinstance(state, np.ndarray):
            warnings.warn("Using a deprecated file format. Please, save the U-Net again with the new version.")
            state = {"config": state[0], "params": state[1:]}

        for self_p, p in zip(self.params, state["params"]):
            self_p.set_value(p)

    def __getstate__(self):
        state = {}
        state["params"] = [i.get_value() for i in self.params]
        return state