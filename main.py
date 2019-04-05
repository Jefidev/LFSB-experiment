import json

from keras.callbacks import EarlyStopping, ModelCheckpoint
from loguru import logger

from config_loader.data_source_processor import DataSourceProcessor
from models.triplet_model import TripletModel

logger.info("Loading config")

with open("ressources/config.json", "r") as c:
    config = json.load(c)

random = config["randomstate"]
name = config["experiment_name"]
result_dir = config["results"]


# Defining test and train
process_source = DataSourceProcessor(config["datasource"], random)


train_sequence = process_source.get_training_set()
test_sequence = process_source.get_test_set()

x, _ = train_sequence[0]
input_shape = x[0][0].shape
print(input_shape)

# Loading model
logger.info("Retrieving model")
model = TripletModel(input_shape).get_model()
model.summary()

# Train the model
logger.info("Starting training")

save_path = "{}/{}.h5".format(result_dir, name)

early_stop = EarlyStopping(patience=3)
check = ModelCheckpoint(
    save_path,
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode="auto",
    period=1,
)

t = train_sequence[0]
# logger.info("Train sequence shape : {}".format(t[0].shape))
# logger.info("First sequence shape : {}".format(t[0][0].shape))

model.fit_generator(train_sequence, epochs=100, callbacks=[early_stop, check])

logger.info("Training complete")

# embed_model = model.get_layer("base_model")
# embed_model.summary()
