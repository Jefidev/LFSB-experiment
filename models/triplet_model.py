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
        merged = concatenate([positive_embed, negative_embed, anchor_embed], axis=-1)

        model = Model(
            [anchor_example, positive_example, negative_example],
            outputs=merged,
            name="triple_siamese",
        )

        # compiling final model
        model.compile(optimizer=Adam(), loss=triplet_loss, metrics=[accuracy])

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


def triplet_loss(y_true, y_pred):
    margin = K.constant(1)
    return K.mean(
        K.maximum(
            K.constant(0),
            K.square(y_pred[:, 0, 0])
            - 0.5 * (K.square(y_pred[:, 1, 0]) + K.square(y_pred[:, 2, 0]))
            + margin,
        )
    )


def accuracy(y_true, y_pred):
    return K.mean(y_pred[:, 0, 0] < y_pred[:, 1, 0])


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
