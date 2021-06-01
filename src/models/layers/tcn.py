from tensorflow.keras.layers import SpatialDropout1D, Activation, Convolution1D, add
from tensorflow.keras.layers import BatchNormalization, Lambda


def TCN(x, nb_filters=64, kernel_size=1,nb_stacks=1, dilations=(1, 2, 4, 8), padding='causal',
        use_skip_connections=True, dropout_rate=0.2, return_sequences=False, activation='linear',
        kernel_initializer='he_normal', use_batch_norm=False):
  """
  Temporal Convolutional Layer (TCN)
  Referred from: https://github.com/philipperemy/keras-tcn

    Input shape:
        A tensor of shape (batch_size, timesteps, n_nodes)
    Args:
        nb_filters          : Number of filters to use in the convolutional layers.
        kernel_size         : Size of the kernel to use in each convolutional layer.
        dilations           : List of the dilation rate. Example is: [1, 2, 4, 8, 16, 32, 64].
        nb_stacks           : Number of stacks of residual blocks to use.
        padding             : Padding to use in the convolutional layers, 'causal' or 'same'.
        use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
        return_sequences    : Boolean. Whether to return the last output in the output sequence, or the full sequence.
        activation          : Activation used in the residual blocks o = Activation(x + F(x)).
        dropout_rate        : Float between 0 and 1. Fraction of the input units to drop.
        kernel_initializer  : Initializer for the kernel weights matrix (Conv1D).
        use_batch_norm      : Whether to use batch normalization in the residual layers or not.
    Return:
        A TCN layer
  """
  # x = tf.transpose(x, perm=[0,2,1])
  x = Convolution1D(nb_filters, 1, padding=padding, kernel_initializer=kernel_initializer)(x)
  skip_connections = []

  # Stack multiple tcn blocks with dilation rate
  for s in range(nb_stacks):
    for d in dilations:
      prev_x = x
      for k in range(3):
        x = Convolution1D(filters=nb_filters,
                   kernel_size=kernel_size,
                   dilation_rate=d,
                   kernel_initializer=kernel_initializer,
                   padding=padding)(x)
        if use_batch_norm:
          x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout1D(rate=dropout_rate)(inputs=x, training=None)
      
      prev_x = Convolution1D(nb_filters, 1, padding='same')(prev_x)
      skipout = add([prev_x, x])
      skipout = Activation(activation)(skipout)
      skip_connections.append(skipout)

    if use_skip_connections:
      x = add(skip_connections)

    if not return_sequences:
      x = Lambda(lambda tt:tt[:,-1,:])(x)
      
    return x