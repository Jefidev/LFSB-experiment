from loguru import logger

from models.base_model import BaseModel


class TripletLossModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)

    def train(self, train):
        logger.info("Beginning train")

    def test(self, test):
        logger.info("Testing model. Results saved in {}".format(self.results_path))

    def build_new(self):
        input_shape = self.config["input_shape"]

        positive_example = Input(input_shape)
        negative_example = Input(input_shape)
        anchor_example = Input(input_shape)

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

    def load(self):
        logger.info("Loading model at {}".format(self.file))

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
            self.config["embedding_size"],
            activation="linear",
            kernel_regularizer=regularizers.l2(0.01),
            activity_regularizer=regularizers.l2(0.01),
        )(x)

        return Model(embed.input, base_model, name="base_model")

    def compile_model(self, model):
        return model.compile(optimizer=Adam(), loss=self.triplet_loss_v2)

    def _triplet_loss_v2(y_true, y_preds):
        # Alpha = 0.2

        embedding_size = self.config["embedding_size"]

        reshape_triplets_embeddings = tf.reshape(y_preds, [-1, 3, embedding_size])
        an, pn, nn = tf.split(reshape_triplets_embeddings, 3, 1)
        a = tf.reshape(an, [-1, embedding_size])
        p = tf.reshape(pn, [-1, embedding_size])
        n = tf.reshape(nn, [-1, embedding_size])
        p_dist = K.sum(K.square(a - p), axis=-1)
        n_dist = K.sum(K.square(a - n), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + 0.2, 0), axis=0)
