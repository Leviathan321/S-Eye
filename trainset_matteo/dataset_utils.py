import os
from typing import Dict, List, Tuple
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from maxitwo.config import BEAMNG_SIM_NAME, DONKEY_SIM_NAME, IMAGE_HEIGHT, IMAGE_WIDTH, UDACITY_SIM_NAME
from maxitwo.global_log import GlobalLog
import sys 
import logging as log
from maxitwo.utils.dataset_utils import load_archive

SIMULATOR_NAMES = [BEAMNG_SIM_NAME, DONKEY_SIM_NAME, UDACITY_SIM_NAME]

# path to repo where we colourmask based segmentation
#sys.path.insert(0,r"C:\Users\sorokin\Documents\testing\segment")
sys.path.insert(0,r"/home/lev/Projects/testing/Multi-Simulation/segment")

import segment_by_mask

''' Apply manual colour mask segmentation to retrieve data set for training/validation'''
''' y's are not steeering labels but the segmented images '''

def load_archive_into_datasets_segmented(
    archive_path: str,
    archive_names: List[str],
    seed: int,
    test_split: float = 0.2,
    predict_throttle: bool = False,
    env_name: str = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    logg = GlobalLog("load_archive_into_dataset")

    if env_name == "mixed":
        X_train, X_test, y_train, y_test = [], [], [], []
        for sim_name in SIMULATOR_NAMES:
            filtered_archive_names = list(filter(lambda an: sim_name in an, archive_names))
            assert (
                len(filtered_archive_names) <= 1
            ), "There must be at most one archive name that contains {}. Found: {}".format(sim_name, filtered_archive_names)

            if len(filtered_archive_names) == 1:

                filtered_archive_name = filtered_archive_names[0]
                numpy_dict = load_archive(archive_path=archive_path, archive_name=filtered_archive_name)
                obs = numpy_dict["observations"]

                # apply colour mask, actions are segmented images
                segmented_images = segment_by_mask.segment_from_arrays(obs, 
                        sim_name, 
                        output_folder = None,
                        image_names = None,
                        output_format = "jpg",
                        do_preprocess = True,
                        do_write = False)

                actions = segmented_images
                               
                print(f"Image segmented.")
                input()

                # if len(actions.shape) > 2:
                #     actions = actions.squeeze(axis=1)

                # if not predict_throttle:
                #     actions = actions[:, 0]

                X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
                    obs, actions, test_size=test_split, random_state=seed
                )
                logg.info(
                    "Training set size for {}: {}. Validation set size for {}: {}".format(
                        sim_name, X_train_i.shape, sim_name, X_test_i.shape
                    )
                )
                logg.info("Adding sim name dimension")
                X_train.append([X_train_i, sim_name])
                X_test.append([X_test_i, sim_name])
                y_train.append(y_train_i)
                y_test.append(y_test_i)

        assert len(X_train) > 0, "Training data must not be empty"

        X_train = np.concatenate(X_train)
        X_test = np.concatenate(X_test)
        y_train = np.concatenate(y_train)
        y_test = np.concatenate(y_test)

        logg.info("Mixed training set size: {}. Mixed validation set size: {}".format(X_train.shape, X_test.shape))

        return X_train, X_test, y_train, y_test

    obs = []
    actions = []
    for i in range(len(archive_names)):
        numpy_dict = load_archive(archive_path=archive_path, archive_name=archive_names[i])
        obs_i = numpy_dict["observations"]

        # apply colour mask, actions are segmented images
        segmented_images = segment_by_mask.segment_from_arrays(obs_i, 
                sim_name, 
                output_folder = None,
                image_names = None,
                do_preprocess = True,
                do_write = False)

        actions_i = segmented_images
                        
        print(f"Image segmented.")
        input()

        # if len(actions.shape) > 2:
        #     actions = actions.squeeze(axis=1)

        # if not predict_throttle:
        #     actions = actions[:, 0]

        obs.append(obs_i)
        actions.append(actions_i)

    obs = np.concatenate(obs)
    actions = np.concatenate(actions)

    X = obs

    if len(actions.shape) > 2:
        actions = actions.squeeze(axis=1)

    if not predict_throttle:
        y = actions[:, 0]
    else:
        y = actions

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=seed)
    logg.info("Training set size: {}. Validation set size: {}".format(X_train.shape, X_test.shape))

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    #archive_path = r"C:\\Users\\sorokin\\Downloads\\training_datasets\\"
    archive_path = r"~/Downloads/training_datasets/"

    seed = 1
    test_split = 0
    predict_throttle = False
    env_name = "mixed"

    ######################

    if env_name == "udacity":
        archive_names =  [r"udacity-2022_05_31_12_17_56-archive-agent-autopilot-seed-0-episodes-50.npz"]
    elif env_name == "donkey":
        archive_names = [r"donkey-2022_05_31_12_45_57-archive-agent-autopilot-seed-0-episodes-50.npz"]
    elif env_name == "beamng":
        archive_names = [r"beamng-2022_05_31_14_34_55-archive-agent-autopilot-seed-0-episodes-50.npz"]
    elif env_name == "mixed":
        archive_names = [r"udacity-2022_05_31_12_17_56-archive-agent-autopilot-seed-0-episodes-50.npz",
                        r"donkey-2022_05_31_12_45_57-archive-agent-autopilot-seed-0-episodes-50.npz",
                        r"beamng-2022_05_31_14_34_55-archive-agent-autopilot-seed-0-episodes-50.npz"]
    ##################
    if seed == -1:
        seed = np.random.randint(2**30 - 1)

    log.info("Random seed: {}".format(seed))

    X_train, X_test, y_train, y_test = load_archive_into_datasets_segmented(
                    archive_path,
                    archive_names,
                    seed,
                    test_split = 0.2,
                    predict_throttle = False,
                    env_name = env_name)