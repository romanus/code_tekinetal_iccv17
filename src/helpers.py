import os
import os.path
import numpy
import numpy as np

import logging

def ReLU(x):
    """
    Rectified linear unit
    :param x: input value
    :return: max(x, 0)
    """
    import theano.tensor as T
    return T.switch(x < 0, 0, x)


def LeakyReLU(a=0.33):
    """
    Leaky rectified linear unit with different scale
    :param a: scale
    :return: max(x, a*x)
    """
    import theano.tensor as T

    def inner(x):
        return T.switch(x < a*x, a*x, x)
    return inner


def InvReLU(x):
    """
    Rectified linear unit
    :param x: input value
    :return: max(x,0)
    """
    import theano.tensor as T
    x *= -1.
    return T.switch(x < 0, 0, x)


def TruncLin(x):
    """
    Truncated linear unit
    :param x: input value
    :return: max(min(x,1),-1)
    """
    import theano.tensor as T
    return T.switch(x < -1, -1, T.switch(x > 1, 1, x))


def TruncReLU(x):
    """
    Truncated rectified linear unit
    :param x: input value
    :return: max(min(x,1),0)
    """
    import theano.tensor as T
    return T.switch(x < 0, 0, T.switch(x > 1, 1, x))


def SlopeLin(slope):
    """
    Linear unit with different slopes
    :param slope: slope of negative quadrant
    :return: x if x > 0 else x/slope
    """
    import theano.tensor as T

    def inner(x):
        return T.switch(T.gt(x, 0), x, T.true_div(x, slope))
    return inner


def SlopeLinInv(slope):
    """
    Truncated linear unit
    :param slope: slope of negative quadrant
    :return: x if x > 0 else x*slope
    """
    import theano.tensor as T

    def inner(x):
        return T.switch(T.gt(x, 0), x, T.mul(x, slope))
    return inner


def SlopeLin2(x, slope):
    """
    Linear unit with different slopes
    :param slope: slope of negative quadrant
    :return: x if x > 0 else x/slope
    """

    import theano.tensor as T
    return T.switch(T.gt(x, 0), x, T.true_div(x, slope))


def huber(delta):
    """
    Huber loss, robust at 0
    :param delta: delta parameter
    :return: loss value
    """
    import theano.tensor as T

    def inner(target, output):
        d = target - output
        a = .5 * d**2
        b = delta * (T.abs_(d) - delta / 2.)
        l = T.switch(T.abs_(d) <= delta, a, b)
        return l
    return inner


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    # https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    arrays = [numpy.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = numpy.prod([x.size for x in arrays])
    if out is None:
        out = numpy.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = numpy.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out


def shuffle_many_inplace(arrays, random_state=None):
    """
    Shuffle given list of array consistently along first dimension
    :param arrays: list of arrays
    :param random_state: random state or seed
    :return: None
    """

    if random_state is None:
        rng = numpy.random.mtrand._rand
    elif isinstance(random_state, numpy.random.RandomState):
        rng = random_state
    else:
        raise ValueError("random_state must be None or numpy RandomState")

    assert all(i.shape[0] == arrays[0].shape[0] for i in arrays[1:])

    # Fisher-Yates Shuffle
    for oi in reversed(range(1, arrays[0].shape[0])):
        ni = rng.randint(oi+1)
        for a in arrays:
            a[[oi, ni]] = a[[ni, oi]]


def gaussian_kernel(kernel_shape, sigma):
    """
    Get 2D Gaussian kernel
    :param kernel_shape: kernel size
    :param sigma: sigma of Gaussian distribution
    :return: 2D Gaussian kernel
    """
    kern = numpy.zeros((kernel_shape, kernel_shape), dtype='float32')

    def gauss(x, y, s):
        Z = 2. * numpy.pi * s ** 2.
        return 1. / Z * numpy.exp(-(x ** 2. + y ** 2.) / (2. * s ** 2.))

    mid = numpy.floor(kernel_shape / 2.)
    for i in xrange(0, kernel_shape):
        for j in xrange(0, kernel_shape):
            kern[i, j] = gauss(i - mid, j - mid, sigma)

    return kern / kern.sum()

def plot_joints(ax, joints, *args, **kwargs):
    
    radius = np.max(np.abs(joints))
    
    rescale = not ax.collections
    
    ax.scatter(joints[:,0], joints[:,2], -joints[:,1], *args, **kwargs)
    
    if rescale:
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_zlim(-radius, radius)

def plot_pose(ax, pose, joint_style='ro', bone_style='b-', ):
    
    bones = [[0, 1], [1, 2], [2, 3],
             [0, 4], [4, 5], [5, 6],
             [0, 7], [7, 8], [8, 9], [9, 10],
             [8, 11], [11, 12], [12, 13],
             [8, 14], [14, 15], [15, 16]]
    
    actors = [] 
    pose = np.reshape(pose, (17, 3))
    for bone in bones:
        if bone == [0, 1]:
            bone_style = 'b-'
        elif bone == [1, 2]:
            bone_style = 'g-'
        elif bone == [2, 3]:
            bone_style = 'g-'
        elif bone == [0, 4]:
            bone_style = 'b-'
        elif bone == [4, 5]:
            bone_style = 'r-'
        elif bone == [5, 6]:
            bone_style = 'r-'
        elif bone == [0, 7]:
            bone_style = 'b-'
        elif bone == [7, 8]:
            bone_style = 'b-'
        elif bone == [8, 9]:
            bone_style = 'b-'
        elif bone == [9, 10]:
            bone_style = 'b-'
        elif bone == [8, 11]:
            bone_style = 'r-'
        elif bone == [11, 12]:
            bone_style = 'r-'
        elif bone == [12, 13]:
            bone_style = 'r-'
        elif bone == [8, 14]:
            bone_style = 'g-'
        elif bone == [14, 15]:
            bone_style = 'g-'
        elif bone == [15, 16]:
            bone_style = 'g-'
        actors.extend(ax.plot3D(pose[bone, 0], pose[bone, 2], -pose[bone, 1], bone_style, linewidth=3))

    ax.set_axis_off()
    actors.extend(ax.plot3D(pose[:, 0], pose[:, 2], -pose[:, 1], joint_style))

def makedirs(path):
    if not os.path.exists( path ):
        os.makedirs( path )

def config_logger(log_file):
    
    class MyFormatter(logging.Formatter):
        
        info_format = "\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s"
        error_format = "\x1b[31;1m%(asctime)s [%(name)s] [%(levelname)s]\x1b[0m %(message)s"
        
        def format(self, record):
            
            if record.levelno > logging.INFO:
                self._style._fmt = self.error_format
            else:
                self._style._fmt = self.info_format
            
            return super(MyFormatter, self).format(record)
    
    rootLogger = logging.getLogger()
    
    fileHandler = logging.FileHandler(log_file)
    fileFormatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)s]> %(message)s")
    fileHandler.setFormatter(fileFormatter)
    rootLogger.addHandler(fileHandler)
    
    consoleHandler = logging.StreamHandler()
    consoleFormatter = MyFormatter()
    consoleHandler.setFormatter(consoleFormatter)
    rootLogger.addHandler(consoleHandler)
    
    rootLogger.setLevel(logging.INFO)
