import json

from keras.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint
from keras.models import Model, load_model
from loguru import logger

from config_loader.data_source_processor import DataSourceProcessor
from data_sequence.triplet_sequence import update_model
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
    model.summary()

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

    lcallback = LambdaCallback(on_batch_end=update_model(train_sequence, model))

    model.fit_generator(
        train_sequence, epochs=100, callbacks=[early_stop, check, lcallback], steps_per_epoch=400
    )
    logger.info("Training complete")


# Extracting the embedding network
logger.info("Extractig embedding network from triplet network")

embed_input = model.layers[3].get_input_at(-1)
embed_output = model.layers[3].get_output_at(-1)

embedding_model = Model(embed_input, embed_output)

# Predict embedding for test
logger.info("Start predict")
preds = embedding_model.predict_generator(test_sequence)
label = test_sequence.y.as_matrix()

logger.info("Saving preds")
preds.dump("./results/preds.np")

logger.info("Saving labels")
label.dump("./results/label.np")
