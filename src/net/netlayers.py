
import numpy as np

import theano
import theano.tensor as tt
from theano.tensor import nnet
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d
from theano.sandbox.cuda.dnn import dnn_conv, dnn_conv3d
from theano.tensor.shared_randomstreams import RandomStreams

_rng = np.random.RandomState()

class ConnectedLayer(object):
    
    def __init__(self, layer, input, output, params):
        
        self.layer = layer
        self.input = input
        self.output = output
        self.params = params
    

def connect_many(layers, input, intermediate_outputs=False):
    
    cur_output = input
    outputs = []
    
    for layer in layers:
        cur_output = layer.connect(cur_output).output
        outputs.append(cur_output)
    
    if intermediate_outputs:
        return cur_output, outputs
    
    return cur_output

class LambdaLayer(object):
    
    def __init__(self, op, params=None):
        
        self.op = op
        
        self.params = params
        if params is None:
            self.params = []
    
    def connect(self, input):
        
        output = self.op(input)
        return ConnectedLayer(self, input, output, self.params)
    
ElemwiseLayer = LambdaLayer

class Conv2dLayer(object):
    
    def __init__(self, filter_shape, border_mode='valid', rng=_rng):
        
        self.border_mode = border_mode
        
        fan_in = np.prod(filter_shape[1:])
        w_bound = np.sqrt(2. / fan_in)
        self.weights = theano.shared(
            np.array(
                rng.normal(scale=w_bound, size=filter_shape),
                # rng.uniform(low=-w_bound, high=w_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        
        self.bias = theano.shared(
            np.zeros(filter_shape[0], dtype=theano.config.floatX),
            borrow=True
        )
        
        self.params = [self.weights, self.bias]
    
    def connect(self, input):
        
        conv_output = dnn_conv(
            img=input,
            kerns=self.weights,
            border_mode=self.border_mode
        )
        
        output = conv_output + self.bias.dimshuffle('x', 0, 'x', 'x')
        return ConnectedLayer(self, input, output, self.params)


class Conv2dLayer2(object):
    def __init__(self, filter_shape, border_mode='valid', rng=_rng):
        self.border_mode = border_mode

        # W_bound = numpy.sqrt(2. / numpy.prod(filter_shape[1:]))
        # wInitVals = numpy.asarray(rng.normal(loc=0.0, scale=W_bound, size=filter_shape), dtype=floatX)

        fan_in = np.prod(filter_shape[1:])
        w_bound = np.sqrt(2. / fan_in)
        self.weights = theano.shared(
            np.array(
                rng.normal(loc=0.0, scale=w_bound, size=filter_shape),
                # rng.uniform(low=-w_bound, high=w_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # b_values = numpy.zeros((filter_shape[0],), dtype=floatX)  # TODO ones
        self.bias = theano.shared(
            np.zeros(filter_shape[0], dtype=theano.config.floatX),
            borrow=True
        )

        self.params = [self.weights, self.bias]

    def connect(self, input):
        conv_output = dnn_conv(
            img=input,
            kerns=self.weights,
            border_mode=self.border_mode
        )

        output = conv_output + self.bias.dimshuffle('x', 0, 'x', 'x')
        return ConnectedLayer(self, input, output, self.params)


class Conv3dLayer(object):
    
    def __init__(self, filter_shape, border_mode='valid', rng=_rng):
        
        self.border_mode = border_mode
        
        # Weights
        fan_in = np.prod(filter_shape[1:])
        w_bound = np.sqrt(2. / fan_in)
        self.weights = theano.shared(
            np.array(
                rng.normal(scale=w_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        
        # Bias
        self.bias = theano.shared(
            np.zeros(filter_shape[0], dtype=theano.config.floatX),
            borrow=True
        )
        
        self.params = [self.weights, self.bias]
    
    def connect(self, input):
        
        conv_output = dnn_conv3d(
            input,
            self.weights,
            border_mode=self.border_mode
        )
        
        output = self.conv_output + self.bias.dimshuffle('x', 0, 'x', 'x', 'x')
        return ConnectedLayer(self, input, output, self.params)
    

class MaxPool3dLayer(object):
    
    def __init__(self, pool_shape):
        
        self.pool_shape = pool.shape
        self.params = []
    
    def connect(self, input):
        output = max_pool_3d(input,
                             self.pool_shape,
                             ignore_border=False)
        
        return ConnectedLayer(self, input, output, self.params)
    

class MaxPool2dLayer(object):
    
    def __init__(self, pool_shape):
        self.pool_shape = pool_shape
        self.params = []
    
    def connect(self, input):
        output = pool_2d(input,
                         self.pool_shape,
                         ignore_border=False)
        
        return ConnectedLayer(self, input, output, self.params)


class MaxPool2dLayer2(object):
    def __init__(self, pool_shape):
        self.pool_shape = pool_shape
        self.params = []

    def connect(self, input):
        output = pool_2d(input,
                         self.pool_shape,
                         ignore_border=True)

        return ConnectedLayer(self, input, output, self.params)

class UpsamplingLayer(object):
    
    def __init__(self, block_shape, mode='dense'):
        self.block_shape = block_shape
        self.mode = mode
        self.params = []
    
    def connect(self, input):
        output = upsample(input, self.block_shape, self.mode)
        return ConnectedLayer(self, input, output, self.params)
    

class InnerProductLayer(object):
    
    def __init__(self, num_inputs, num_outputs, rng=_rng):
        
        w_bound = np.sqrt(6. / num_inputs)
        self.weights = theano.shared(
            np.array(
                rng.uniform(low=-w_bound, high=w_bound, size=(num_inputs, num_outputs)),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        
        self.bias = theano.shared(
            np.zeros(num_outputs, dtype=theano.config.floatX),
            borrow=True
        )
        
        self.params = [self.weights, self.bias]
    
    def connect(self, input):
        
        output = tt.dot(input, self.weights) + self.bias
        
        return ConnectedLayer(self, input, output, self.params)
    
    def get_conv3d(self, layer_input, filter_shape):
        """A convolutional operation with the weights of the inner product layer."""
        
        conv_weights = tt.reshape(self.weights.T, filter_shape)[:, :, ::-1, ::-1, ::-1]
        
        conv_out = conv3d(
            layer_input.dimshuffle([0, 2, 1, 3, 4]),
            conv_weights.dimshuffle([0, 2, 1, 3, 4]),
            border_mode='valid'
        ).dimshuffle([0, 2, 1, 3, 4])
        
        return conv_out + self.bias.dimshuffle('x', 0, 'x', 'x', 'x')


class InnerProductLayer2(object):
    def __init__(self, num_inputs, num_outputs, rng=_rng):
        self.weights = theano.shared(
            np.array(
                rng.normal(loc=0.0, scale=0.01, size=(num_inputs, num_outputs)),
                dtype=theano.config.floatX
            ),
            borrow=True
        )


        self.bias = theano.shared(
            np.zeros(num_outputs, dtype=theano.config.floatX),
            borrow=True
        )

        self.params = [self.weights, self.bias]

    def connect(self, input):
        output = tt.dot(input, self.weights) + self.bias

        return ConnectedLayer(self, input, output, self.params)

    def get_conv3d(self, layer_input, filter_shape):
        """A convolutional operation with the weights of the inner product layer."""

        conv_weights = tt.reshape(self.weights.T, filter_shape)[:, :, ::-1, ::-1, ::-1]

        conv_out = conv3d(
            layer_input.dimshuffle([0, 2, 1, 3, 4]),
            conv_weights.dimshuffle([0, 2, 1, 3, 4]),
            border_mode='valid'
        ).dimshuffle([0, 2, 1, 3, 4])

        return conv_out + self.bias.dimshuffle('x', 0, 'x', 'x', 'x')


class DropoutLayerOld(object):
    
    def __init__(self, prob_drop, rng=_rng):
        self.prob_drop = prob_drop
        self.prob_keep = 1 - prob_drop
        self.srng = RandomStreams(seed=rng.randint(999999))
        self.params = []
    
    def connect(self, input):
        
        mask = self.srng.binomial(n=1,
                                  p=self.prob_keep,
                                  size=input.shape,
                                  dtype=theano.config.floatX)
        output = mask * input / np.array(self.prob_keep, dtype=theano.config.floatX)
        return ConnectedLayer(self, input, output, self.params)

class DropoutLayer(object):
    def __init__(self, prob_drop, flag_on, rng=_rng):
        self.prob_drop = prob_drop
        self.prob_keep = 1 - prob_drop
        self.flag_on = flag_on
        self.srng = RandomStreams(seed=rng.randint(999999))
        self.params = []

    def connect(self, input):
        mask = self.srng.binomial(n=1,
                                  p=self.prob_keep,
                                  size=input.shape,
                                  dtype=theano.config.floatX)
        output = self.flag_on * tt.cast(mask, theano.config.floatX) * input + (1-self.flag_on) * self.prob_keep * input
        return ConnectedLayer(self, input, output, self.params)

class SoftmaxLayer(object):
    
    def __init__(self, layer_input):
        
        self.output = tt.nnet.softmax(layer_input)
        
        self.prediction = tt.argmax(self.output, axis=1)
        self.params = []
    
    def log_loss(self, real_y, weights=None):
        
        if weights is None:
            return -tt.mean(tt.log(self.output[tt.arange(real_y.shape[0]), real_y]))
        else:
            return -tt.mean(weights * tt.log(self.output[tt.arange(real_y.shape[0]), real_y]))
    
    def errors(self, real_y):
        return tt.mean(tt.neq(self.prediction, real_y))
    

class SSDLayer(object):
    
    def __init__(self, layer_input):
        
        self.output = layer_input
        self.params = []
    
    def loss(self, real_y, weights=None):
        
        if weights is None:
            diff = self.output - real_y
            return tt.mean(diff**2)
        else:
            diff = (self.output - real_y) * weights
            return tt.mean(diff**2)
    
