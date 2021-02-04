import keras.backend as K
import keras
from keras import optimizers
from keras.engine.topology import Layer
from keras.layers import Activation, Lambda, Conv1D, SpatialDropout1D, Dense, BatchNormalization

# Define TCN layer
class tcn_layer:
    """
        Input shape:
            A tensor of shape (batch_size, timesteps, input_dim).
        Args:
            nb_filters: The number of filters to use in the convolutional layers.
            kernel_size: The size of the kernel to use in each convolutional layer.
            dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : The number of stacks of residual blocks to use.
            padding: The padding to use in the convolutional layers, 'causal' or 'same'.
            use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
            return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
            activation: The activation used in the residual blocks o = Activation(x + F(x)).
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm: Whether to use batch normalization in the residual layers or not.
        Returns:
            A TCN layer.
        """

    def __init__(self, nb_filters=64, kernel_size=1,nb_stacks=1, dilations=(1, 2, 4, 8), padding='causal',
                 use_skip_connections=True, dropout_rate=0.0, return_sequences=False, activation='linear',
                 kernel_initializer='he_normal', use_batch_norm=False):
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.activation = activation
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.use_batch_norm = use_batch_norm

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")

        if not isinstance(nb_filters, int):
            print('An interface change occurred after the version 2.1.2.')
            print('Before: tcn.TCN(x, return_sequences=False, ...)')
            print('Now should be: tcn.TCN(return_sequences=False, ...)(x)')
            print('The alternative is to downgrade to 2.1.2 (pip install keras-tcn==2.1.2).')
            raise Exception()

    def __call__(self, inputs, training=None):
        x = inputs
        # x = tf.transpose(x, perm=[0,2,1])
        # 1D FCN.
        x = Conv1D(self.nb_filters, 1, padding=self.padding, kernel_initializer=self.kernel_initializer)(x)
        skip_connections = []
        for s in range(self.nb_stacks):
            for d in self.dilations:

                # Residual block for WaveNet TCN
                prev_x = x
                for k in range(2):
                    x = Conv1D(filters=self.nb_filters,
                            kernel_size=self.kernel_size,
                            dilation_rate=d,
                            kernel_initializer=self.kernel_initializer,
                            padding=self.padding)(x)
                    if self.use_batch_norm:
                        x = BatchNormalization()(x)  
                    x = Activation('relu')(x)
                    x = SpatialDropout1D(rate=self.dropout_rate)(inputs=x, training=training)

                # 1x1 conv to match the shapes (channel dimension)
                prev_x = Conv1D(self.nb_filters, 1, padding='same')(prev_x)
                skip_out = keras.layers.add([prev_x, x])
                skip_out = Activation(self.activation)(skip_out)
                skip_connections.append(skip_out)
        if self.use_skip_connections:
            x = keras.layers.add(skip_connections)
        print('TCN layer, ', x.shape)
        if not self.return_sequences:
            x = Lambda(lambda tt: tt[:, -1, :])(x)
        return x
