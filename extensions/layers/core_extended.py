from keras.layers.core import Layer
import keras.backend as K

if (K.backend() == 'theano'):
    import extensions.backend.theano_backend_ext as T
else:
    import extensions.backend.tensorflow_backend_ext as T

class Resize(Layer):
    '''Resizes an output to a certain shape using interpolation.

    # Arguments
        size: Target size. An integer or a tuple of integers. They must be positive integers.
        axis: Resize axis. An integer or a tuple of integers.
            Do not include the samples dimension (batch size).
            If axis is a tuple, then the axis length and the size length must be same.
        interpolation: 'nearest_neighbor' or 'linear'. The default is 'nearest_neighbor'.

    # Example

    ```python
        # as first layer in a Sequential model
        model = Sequential()
        model.add(Resize(7, axis=1, interpolation='linear', input_shape=(5, 4)))
        # now: model.output_shape == (None, 7, 4)
        # note: `None` is the batch dimension
        model.add(Resize((3, 6), axis=(1, 2), interpolation='nearest_neighbor'))
        # now: model.output_shape == (None, 3, 6)
    ```
    '''

    def __init__(self, size, axis, interpolation='nearest_neighbor', **kwargs):
        super(Resize, self).__init__(**kwargs)
        self.size = size if type(size) in (tuple, list) else (size,)
        self.axis = axis if type(axis) in (tuple, list) else (axis,)
        self.interpolation = interpolation

        if not all(v > 0 for v in self.size):
            raise ValueError('The size of "Resize" must be positive (got ' + str(size) + ').')
        if 0 in self.axis:
            raise ValueError('The axis of "Resize" must not include 0 (got ' + str(axis) + ').')
        if len(self.size) != len(self.axis):
            raise ValueError('The length of size and the length of axis of "Resize" must be same.')
        if interpolation not in ('nearest_neighbor', 'linear'):
            raise ValueError('Unknown interpolation of "Resize" (got ' + str(interpolation) + ').')

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        for a, s in zip(self.axis, self.size):
            output_shape[a] = s
        return tuple(output_shape)

    def call(self, x, mask=None):
        return T.resize(x, size=self.size, axis=self.axis, interpolation=self.interpolation)

    def get_config(self):
        config = {'size': self.size,
                  'axis': self.axis,
                  'interpolation': self.interpolation}
        base_config = super(Resize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
