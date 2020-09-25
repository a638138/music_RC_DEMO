"""Embedding layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import Layer
from keras.legacy import interfaces
from keras.utils.generic_utils import to_list


class neg_sampling(Layer):
    """Turns positive integers (indexes) into dense vectors of fixed size.
    eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

    # Arguments
        input_dim: int > 0. Size of the vocabulary,
            i.e. maximum integer index + 1.
        output_dim: int >= 0. Dimension of the dense embedding.
        embeddings_initializer: Initializer for the `embeddings` matrix
            (see [initializers](../initializers.md)).
        embeddings_regularizer: Regularizer function applied to
            the `embeddings` matrix
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        embeddings_constraint: Constraint function applied to
            the `embeddings` matrix
            (see [constraints](../constraints.md)).
        mask_zero: Whether or not the input value 0 is a special "padding"
            value that should be masked out.
            This is useful when using [recurrent layers](recurrent.md)
            which may take variable length input.
            If this is `True` then all subsequent layers
            in the model need to support masking or an exception will be raised.
            If mask_zero is set to True, as a consequence, index 0 cannot be
            used in the vocabulary (input_dim should equal size of
            vocabulary + 1).
        input_length: Length of input sequences, when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).

    # Input shape
        3D tensor with shape: `(batch_size, sequence_length, tag_list)`.

    # Output shape
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.

    # References
        - [A Theoretically Grounded Application of Dropout in
           Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """

    @interfaces.legacy_embedding_support
    def __init__(self, input_dim, output_dim,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 mode='Train',
                 **kwargs):
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        super(neg_sampling, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mode = mode
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.mask_zero = mask_zero
        self.supports_masking = True
        self.input_length = input_length

    def build(self, input_shape):
        self.weight = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name='neg_sampling_weight',
            dtype=self.dtype,
            trainable=True
        )
        self.bias = self.add_weight(
            shape=(self.input_dim, 1),
            initializer='zeros',
            name='neg_sampling_bias',
            trainable=True,
            dtype=self.dtype)

        super(neg_sampling, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.mode == 'train':
            return (input_shape[0][0], input_shape[0][1])
        else:
            return (input_shape[0][0], self.input_dim)

    def call(self, x):
        target, song_h = x
        if K.dtype(target) != 'int32':
            target = K.cast(target, 'int32')
        if self.mode == 'train': 
            weight = K.gather(self.weight, target)
            result = K.batch_dot(song_h, weight, axes=2)
            bias = K.gather(self.bias, target)
#             print(f'weight:{weight.shape}')
#             print(f'bias:{bias.shape}')
#             print(f'result:{result.shape}')
            bias = K.squeeze(bias, axis=2)
            result = K.squeeze(result, axis=1)
            result = result+bias
 
        else:
            song_h = K.squeeze(song_h, axis=1)
            result = K.dot(song_h, K.transpose(self.weight))+K.transpose(self.bias)

        return result

    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'output_dim': self.output_dim,
                  'embeddings_initializer':
                      initializers.serialize(self.embeddings_initializer),
                  'embeddings_regularizer':
                      regularizers.serialize(self.embeddings_regularizer),
                  'activity_regularizer':
                      regularizers.serialize(self.activity_regularizer),
                  'embeddings_constraint':
                      constraints.serialize(self.embeddings_constraint),
                  'mask_zero': self.mask_zero,
                  'input_length': self.input_length}
        base_config = super(neg_sampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
