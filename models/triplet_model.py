import tensorflow as tf
from keras import backend as K
from keras import regularizers
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
            inputs=[anchor_example, positive_example, negative_example],
            outputs=merged,
            name="triple_siamese",
        )

        # compiling final model
        model.compile(optimizer=Adam(), loss=triplet_loss_v2)

        return model

    def _get_embedding_model(self):

        embed = InceptionV3(
            include_top=False, weights="imagenet", input_shape=self.input_shape
        )

        for layer in embed.layers:
            layer.trainable = False

        x = embed.output
        x = GlobalAveragePooling2D()(x)

        x = Dense(
            2048,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.01),
            activity_regularizer=regularizers.l2(0.01),
        )(x)
        x = Dense(1024, activation="relu")(x)
        base_model = Dense(
            128,
            activation="linear",
            kernel_regularizer=regularizers.l2(0.01),
            activity_regularizer=regularizers.l2(0.01),
        )(x)

        return Model(embed.input, base_model, name="base_model")

    def compile_model(self, model):
        return model.compile(optimizer=Adam(), loss=triplet_loss_v2)


def triplet_loss_v2(y_true, y_preds):
    # Alpha = 0.2

    reshape_triplets_embeddings = tf.reshape(y_preds, [-1, 3, 128])
    an, pn, nn = tf.split(reshape_triplets_embeddings, 3, 1)
    a = tf.reshape(an, [-1, 128])
    p = tf.reshape(pn, [-1, 128])
    n = tf.reshape(nn, [-1, 128])
    p_dist = K.sum(K.square(a - p), axis=-1)
    n_dist = K.sum(K.square(a - n), axis=-1)
    return K.sum(K.maximum(p_dist - n_dist + 0.5, 0), axis=0)
