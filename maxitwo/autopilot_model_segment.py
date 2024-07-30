import datetime
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from maxitwo.config import INPUT_SHAPE_SEGMENTATION, SIMULATOR_NAMES
from maxitwo.utils.dataset_utils import DataGenerator

''' Autopilot model which uses segmented images to drive'''
class AutopilotModelSegment:
    def __init__(self, env_name: str, 
                input_shape: Tuple[int] = INPUT_SHAPE_SEGMENTATION, 
                predict_throttle: bool = True):
        # cropped input_shape: height, width, channels. Allow for mixed datasets
        assert env_name in SIMULATOR_NAMES or env_name == "mixed", "Unknown simulator name {}. Choose among {}".format(
            env_name, SIMULATOR_NAMES
        )
        self.input_shape = input_shape
        self.env_name = env_name
        self.predict_throttle = predict_throttle
        self.model = None

    def build_model(self, keep_probability: float = 0.5, use_dropout = False, drop_rate = 0.2) -> Sequential:
        """
        Modified NVIDIA model
        """
        if use_dropout:
            print(f"selected dropout rate: {drop_rate}")
            model = Sequential()
            model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=self.input_shape))
            model.add(Conv2D(24, (5, 5), activation="elu", strides=(2, 2)))
            model.add(Dropout(drop_rate))
            model.add(Conv2D(36, (5, 5), activation="elu", strides=(2, 2)))
            model.add(Dropout(drop_rate))
            model.add(Conv2D(48, (5, 5), activation="elu", strides=(2, 2)))
            model.add(Dropout(drop_rate))
            model.add(Conv2D(64, (3, 3), activation="elu"))
            model.add(Dropout(drop_rate))
            model.add(Conv2D(64, (3, 3), activation="elu"))
            model.add(Dropout(drop_rate))
            model.add(Flatten())
            model.add(Dense(100, activation="elu"))
            model.add(Dropout(drop_rate))
            model.add(Dense(50, activation="elu"))
            model.add(Dropout(drop_rate))
            model.add(Dense(10, activation="elu"))
            model.add(Dropout(drop_rate))
            if self.predict_throttle:
                model.add(Dense(2))
            else:
                model.add(Dense(1))

            model.summary()
        else:
            model = Sequential()
            model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=self.input_shape))
            model.add(Conv2D(24, (5, 5), activation="elu", strides=(2, 2)))
            model.add(Conv2D(36, (5, 5), activation="elu", strides=(2, 2)))
            model.add(Conv2D(48, (5, 5), activation="elu", strides=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation="elu"))
            model.add(Conv2D(64, (3, 3), activation="elu"))
            model.add(Dropout(keep_probability))
            model.add(Flatten())
            model.add(Dense(100, activation="elu"))
            model.add(Dense(50, activation="elu"))
            model.add(Dense(10, activation="elu"))

            if self.predict_throttle:
                model.add(Dense(2))
            else:
                model.add(Dense(1))

            model.summary()

        return model

    def load(self, model_path: str) -> None:
        assert os.path.exists(model_path), "Model path {} not found".format(model_path)
        with tf.device("cpu:0"):
            self.model = load_model(filepath=model_path)

    def train_model(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        save_path: str,
        model_name: str,
        save_best_only: bool = True,
        keep_probability: float = 0.5,
        learning_rate: float = 1e-4,
        nb_epoch: int = 200,
        batch_size: int = 128,
        early_stopping_patience: int = 3,
        save_plots: bool = True,
        preprocess: bool = True,
        fake_images: bool = False,
        use_dropout: bool = False,
        drop_rate: float = 0.2
    ) -> None:
        os.makedirs(save_path, exist_ok=True)
        self.model = self.build_model(keep_probability=keep_probability,
                                    use_dropout=use_dropout,
                                    drop_rate=drop_rate)
        filename = "{}-{}-{}.h5".format(self.env_name, model_name, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        if fake_images:
            filename = "{}-fake-{}-{}.h5".format(
                self.env_name, model_name, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            )

        checkpoint = ModelCheckpoint(
            os.path.join(save_path, filename), monitor="val_loss", verbose=0, save_best_only=save_best_only, mode="auto"
        )

        self.model.compile(loss="mean_squared_error", optimizer=Adam(lr=learning_rate))

        early_stopping = EarlyStopping(monitor="val_loss", patience=early_stopping_patience)

        train_generator = DataGenerator(
            X=X_train,
            y=y_train,
            batch_size=batch_size,
            is_training=True,
            env_name=self.env_name,
            input_shape=self.input_shape,
            predict_throttle=self.predict_throttle,
            preprocess=preprocess,
            fake_images=fake_images,
        )
        validation_generator = DataGenerator(
            X=X_val,
            y=y_val,
            batch_size=batch_size,
            is_training=False,
            env_name=self.env_name,
            input_shape=self.input_shape,
            predict_throttle=self.predict_throttle,
            preprocess=preprocess,
            fake_images=fake_images,
        )

        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=nb_epoch,
            use_multiprocessing=False,
            max_queue_size=10,
            workers=8,
            callbacks=[checkpoint, early_stopping],
            verbose=1,
        )

        if save_plots:

            plt.figure()
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.title("model loss")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(["train", "val"], loc="upper left")

            plt.savefig(
                os.path.join(
                    save_path, "{}-loss-{}.pdf".format(self.env_name, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
                ),
                format="pdf",
            )
