import json

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from loguru import logger

from config_loader.data_source_processor import DataSourceProcessor
from models.triplet_model import TripletModel, triplet_loss_v2

logger.info("Loading config")

with open("ressources/config.json", "r") as c:
    config = json.load(c)

random = config["randomstate"]
name = config["experiment_name"]
result_dir = config["results"]
load = config.get("load", None)


# Defining test and train
process_source = DataSourceProcessor(config["datasource"], random)


train_sequence = process_source.get_training_set()
test_sequence = process_source.get_test_set()

x, _ = train_sequence[0]
input_shape = x[0][0].shape
print(input_shape)

# Loading model
if load:
    logger.info("Loading from file {}".format(load["model_file"]))
    model = load_model(
        load["model_file"], custom_objects={"triplet_loss_v2": triplet_loss_v2}
    )
else:
    logger.info("Retrieving model")
    model = TripletModel(input_shape).get_model()
    model.summary()

# Train the model
if load == None or load["train"]:
    logger.info("Starting training")

    save_path = "{}/{}.h5".format(result_dir, name)

    early_stop = EarlyStopping(monitor="loss", patience=3)
    check = ModelCheckpoint(
        save_path,
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        period=1,
        monitor="loss",
    )
    model.fit_generator(train_sequence, epochs=100, callbacks=[early_stop, check])
    logger.info("Training complete")

# embed_model = model.get_layer("base_model")
# embed_model.summary()
