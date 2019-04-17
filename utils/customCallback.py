from keras.callbacks import Callback
from keras.models import Model

from loguru import logger


class ModelUpdater(Callback):
    def __init__(self, model, train_seq):
        self.model = model
        self.train_sequence = train_seq

        super().__init__()

    def on_batch_end(self, batch, logs={}):
        embed_input = self.model.layers[3].get_input_at(-1)
        embed_output = self.model.layers[3].get_output_at(-1)

        embedding_model = Model(embed_input, embed_output)

        self.train_sequence.set_model(embedding_model)
        logger.info("Model updated")

    def on_epoch_end(self, batch, logs={}):
        self.train_sequence.reset_data()
