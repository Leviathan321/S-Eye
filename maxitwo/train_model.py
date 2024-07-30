import argparse
import datetime
import os

import numpy as np

from maxitwo.autopilot_model import AutopilotModel
from maxitwo.config import INPUT_SHAPE, SIMULATOR_NAMES
from maxitwo.global_log import GlobalLog
from maxitwo.utils.dataset_utils import load_archive_into_dataset
from maxitwo.utils.randomness import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="Random seed", type=int, default=-1)
parser.add_argument("--archive-path", help="Archive path", type=str, default="logs")
parser.add_argument("--env-name", help="Simulator name", type=str, choices=[*SIMULATOR_NAMES, "mixed"], required=True)
parser.add_argument(
    "--archive-names", nargs="+", help="Archive name to analyze (with extension, .npz)", type=str, required=True
)
parser.add_argument(
    "--model-save-path", help="Path where model will be saved", type=str, default=os.path.join("maxitwo","logs", "models")
)
parser.add_argument("--model-name", help="Model name (without the extension)", type=str, required=True)
parser.add_argument("--predict-throttle", help="Predict steering and throttle", action="store_true", default=False)
parser.add_argument("--no-preprocess", help="Do not preprocess data during training", action="store_true", default=False)
parser.add_argument("--test-split", help="Test split", type=float, default=0.2)
parser.add_argument("--keep-probability", help="Keep probability (dropout)", type=float, default=0.5)
parser.add_argument("--learning-rate", help="Learning rate", type=float, default=1e-4)
parser.add_argument("--nb-epoch", help="Number of epochs", type=int, default=5)
parser.add_argument("--batch-size", help="Batch size", type=int, default=128)
parser.add_argument(
    "--early-stopping-patience",
    help="Number of epochs of no validation loss improvement used to stop training",
    type=int,
    default=3,
)
parser.add_argument(
    "--fake-images",
    help="Whether the training is performed on images produced by the cyclegan. The fake images contained on the archives are already cropped.",
    action="store_true",
    default=False
)
parser.add_argument("--use-dropout", help="Use MCD.", type=bool, default=False) # flag to use MCD
parser.add_argument("--drop-rate", help="Drop rate for MCD layers.", type=float, default=0.2)
parser.add_argument("--drop-rate-start", help="Start .", type=float, default=0.2)
parser.add_argument("--drop-rate-end", help="End.", type=float, default=0.5)

parser.add_argument("--num-ensembles", help="Specifiy the number of model to generate for DeepEnsemble based testing.", type=int, default=32)

args = parser.parse_args()

if __name__ == "__main__":

    logg = GlobalLog("train_model")

    if args.seed == -1:
        args.seed = np.random.randint(2**30 - 1)

    logg.info("Random seed: {}".format(args.seed))
    set_random_seed(seed=args.seed)

    train_data, test_data, train_labels, test_labels = load_archive_into_dataset(
        archive_path=args.archive_path,
        archive_names=args.archive_names,
        seed=args.seed,
        test_split=args.test_split,
        predict_throttle=args.predict_throttle,
        env_name=None if args.env_name != "mixed" else "mixed",
    )

    autopilot_model = AutopilotModel(env_name=args.env_name, 
                                     input_shape=INPUT_SHAPE, 
                                     predict_throttle=args.predict_throttle)
    
if args.use_dropout:
    # generate drop out models with different dropout rates
    STEP = 0.05
    time =  datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    for drop_rate in np.arange(args.drop_rate_start, args.drop_rate_end + STEP, STEP):
        print(drop_rate)
        autopilot_model.train_model(
            X_train=train_data,
            X_val=test_data,
            y_train=train_labels,
            y_val=test_labels,
            save_path=args.model_save_path + os.sep + time,
            model_name=args.model_name + f"_{drop_rate}",
            save_best_only=True,
            keep_probability=args.keep_probability,
            learning_rate=args.learning_rate,
            nb_epoch=args.nb_epoch,
            batch_size=args.batch_size,
            early_stopping_patience=args.early_stopping_patience,
            save_plots=True,
            preprocess=not args.no_preprocess,
            fake_images=args.fake_images,
            use_dropout = args.use_dropout,
            drop_rate = drop_rate
        )
else:
    # generate DE models
    time =  datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    for i in range(args.num_ensembles):
        print(f"Generating {i}.th model out of {args.num_ensembles} ensemble models")
        autopilot_model.train_model(
            X_train=train_data,
            X_val=test_data,
            y_train=train_labels,
            y_val=test_labels,
            save_path=args.model_save_path + os.sep + time,
            model_name=args.model_name + f"_{i}",
            save_best_only=True,
            keep_probability=args.keep_probability,
            learning_rate=args.learning_rate,
            nb_epoch=args.nb_epoch,
            batch_size=args.batch_size,
            early_stopping_patience=args.early_stopping_patience,
            save_plots=True,
            preprocess=not args.no_preprocess,
            fake_images=args.fake_images,
            use_dropout = args.use_dropout,
            drop_rate = args.drop_rate
        )