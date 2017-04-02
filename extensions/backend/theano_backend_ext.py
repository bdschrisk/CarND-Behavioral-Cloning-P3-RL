from theano import tensor as T

def resize(x, size, axis, interpolation='nearest_neighbor'):
    '''Resize the tensor.
    `size` must be an integer or a tuple of integers or a list of integers.
    If `axis` is a tuple or a list, then the `axis` length and the `size` length must be same.
    `interpolation` must be 'nearest_neighbor' or 'linear'. The default is 'nearest_neighbor'.
    '''
    if type(axis) in (tuple, list):
        if len(axis) != len(size):
            raise ValueError('The axis length and the size length are not same.')
        for s, a in zip(size, axis):
            x = resize(x, s, a, interpolation)
        return x
    else:
        if interpolation == 'nearest_neighbor':
            orig_size = x.shape[axis]
            idx_f = (orig_size - 1) / T.maximum(1, size - 1).astype(x.dtype) * T.arange(size)
            idx_i = T.round(idx_f).astype("int32")
            idx_i = tuple(idx_i if i == axis else slice(None) for i in range(x.ndim))
            return x[idx_i]
        elif interpolation == 'linear':
            orig_size = x.shape[axis]
            idx_f = (orig_size - 1) / T.maximum(1, size - 1).astype(x.dtype) * T.arange(size)
            idx_i0 = idx_f.astype("int32")
            idx_i1 = T.minimum(orig_size - 1, idx_i0 + 1)
            delta = (idx_f - idx_i0).astype(x.dtype)
            delta = delta.reshape(tuple(-1 if i == axis else 1 for i in range(x.ndim)))
            idx_i0 = tuple(idx_i0 if i == axis else slice(None) for i in range(x.ndim))
            idx_i1 = tuple(idx_i1 if i == axis else slice(None) for i in range(x.ndim))
            return x[idx_i0] * (1 - delta) + x[idx_i1] * delta
        else:
            raise ValueError('Invalid interpolation: ' + str(interpolation))