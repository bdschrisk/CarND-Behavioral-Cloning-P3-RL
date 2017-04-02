import keras.backend as K

### Custom Keras Metrics ###

def qmax(y_true, y_pred):
    return K.max(y_pred)
