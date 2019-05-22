import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Input, Lambda, concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam
from loguru import logger

from models.base_model import BaseModel
from utils.customCallback import ModelUpdater


class TripletLossModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)

    def train(self, train):
        x, _ = train[0]
        input_shape = x[0][0].shape

        logger.info("Model input shape is {}".format(input_shape))

        model = None

        if self.config.get("load_existing"):
            model = self.load()
        else:
            model = self.build_new(input_shape)

        logger.info("Beginning train")

        early_stop = EarlyStopping(monitor="loss", patience=5)
        check = ModelCheckpoint(
            self.file,
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            period=1,
            monitor="loss",
        )

        model_switcher = ModelUpdater(model, train)

        model.fit_generator(
            train, epochs=100, callbacks=[early_stop, check, model_switcher], steps_per_epoch=10
        )

        logger.info("Training complete")

    def test(self, test):
        logger.info("Testing model. Results saved in {}".format(self.results_path))
        model = self.load()

        logger.info("Extractig embedding network from triplet network")

        embed_input = model.layers[3].get_input_at(-1)
        embed_output = model.layers[3].get_output_at(-1)

        embedding_model = Model(embed_input, embed_output)

        logger.info("Start predict")
        preds = embedding_model.predict_generator(test)

        logger.info("Saving preds")
        preds.dump("{}/preds.np".format(self.results_path))

        logger.info("Saving labels")
        label.dump("{}/label.np".format(self.results_path))

    def build_new(self, input_shape):

        positive_example = Input(input_shape)
        negative_example = Input(input_shape)
        anchor_example = Input(input_shape)

        # Retrieving the base model
        base_model = self._get_embedding_model(input_shape)

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
        model = self.compile_model(model)

        return model

    def load(self):
        logger.info("Loading model at {}".format(self.file))
        return load_model(self.file)

    def _get_embedding_model(self, input_shape):

        embed = VGG16(
            include_top=False, weights="imagenet", input_shape=input_shape
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
        model.compile(optimizer=Adam(), loss=self._triplet_loss_v2)
        return model

    def _triplet_loss_v2(self, y_true, y_preds):
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
