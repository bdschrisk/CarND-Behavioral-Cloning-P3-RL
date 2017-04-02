import tensorflow as tf

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
            x_dtype = x.dtype.base_dtype
            x_ndim = len(x.get_shape())
            orig_size = x.get_shape()[axis]
            ratio = tf.cast(orig_size - 1, x_dtype) / tf.cast(tf.maximum(1, size - 1), x_dtype)
            idx_f = ratio * tf.cast(tf.range(size), x_dtype)
            idx_i = tf.to_int32(tf.round(idx_f))
            x = tf.transpose(x, [axis] + list(range(axis)) + list(range(axis + 1, x_ndim)))
            x = tf.gather(x, idx_i)
            x = tf.transpose(x, list(range(1, axis + 1)) + [0] + list(range(axis + 1, x_ndim)))
            return x

        elif interpolation == 'linear':
            x_dtype = x.dtype.base_dtype
            x_ndim = len(x.get_shape())
            orig_size = x.get_shape()[axis]
            ratio = tf.cast(orig_size - 1, x_dtype) / tf.cast(tf.maximum(1, size - 1), x_dtype)
            idx_f = ratio * tf.cast(tf.range(size), x_dtype)
            idx_i0 = tf.to_int32(idx_f)
            idx_i1 = tf.minimum(orig_size - 1, idx_i0 + 1)
            delta = idx_f - tf.cast(idx_i0, x_dtype)
            delta = tf.reshape(delta, [-1] + [1] * (x_ndim - 1))
            perm_from = [axis] + list(range(axis)) + list(range(axis + 1, x_ndim))
            x0 = tf.transpose(x, perm_from)
            x1 = tf.transpose(x, perm_from)
            x0 = tf.gather(x0, idx_i0)
            x1 = tf.gather(x1, idx_i1)
            x = x0 * (1 - delta) + x1 * delta
            x = tf.transpose(x, list(range(1, axis + 1)) + [0] + list(range(axis + 1, x_ndim)))

            return x
        else:
            raise ValueError('Invalid interpolation: ' + str(interpolation))