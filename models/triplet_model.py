import tensorflow as tf
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Input, Lambda, concatenate
from keras.models import Model
from keras.optimizers import Adam


class TripletModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def get_embedded(self, model):
        pass

    def get_model(self):
        positive_example = Input(self.input_shape)
        negative_example = Input(self.input_shape)
        anchor_example = Input(self.input_shape)

        # Retrieving the base model
        base_model = self._get_embedding_model()

        positive_embed = base_model(positive_example)
        negative_embed = base_model(negative_example)
        anchor_embed = base_model(anchor_example)

        # Stacking output
        merged = concatenate([anchor_embed, positive_embed, negative_embed], axis=-1)

        model = Model(
            inputs=[anchor_embed, positive_embed, negative_embed],
            outputs=merged,
            name="triple_siamese",
        )

        # compiling final model
        model.compile(optimizer=Adam(), loss=triplet_loss)

        return model

    def _get_embedding_model(self):

        embed = InceptionV3(
            include_top=False, weights="imagenet", input_shape=self.input_shape
        )

        for layer in embed.layers:
            layer.trainable = False

        x = embed.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(2048, activation="relu")(x)
        x = Dense(2048, activation="relu")(x)
        base_model = Dense(1024, activation="relu")(x)

        return Model(embed.input, base_model, name="base_model")


def triplet_loss(y_true, y_pred, N=1024, beta=1024, epsilon=1e-8):
    """
    Implementation of the triplet loss function
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    N  --  The number of dimension 
    beta -- The scaling factor, N is recommended
    epsilon -- The Epsilon value to prevent ln(0)
    
    
    Returns:
    loss -- real number, value of the loss
    """
    anchor = tf.convert_to_tensor(y_pred[:, 0:N])
    positive = tf.convert_to_tensor(y_pred[:, N : N * 2])
    negative = tf.convert_to_tensor(y_pred[:, N * 2 : N * 3])

    # distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    # distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    # Non Linear Values

    # -ln(-x/N+1)
    pos_dist = -tf.log(-tf.divide((pos_dist), beta) + 1 + epsilon)
    neg_dist = -tf.log(-tf.divide((N - neg_dist), beta) + 1 + epsilon)

    # compute loss
    loss = neg_dist + pos_dist

    return loss
