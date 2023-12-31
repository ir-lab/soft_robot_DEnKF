import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import random
import tensorflow as tf
import time
import pickle
import pdb
import tensorflow_probability as tfp
import csv
import cv2
import diff_EKF
from dataloader import DataLoader

DataLoader = DataLoader()

"""
define the training loop
"""


def run_filter(mode):
    tf.keras.backend.clear_session()
    dim_x = 7
    if mode == True:
        # define batch_size
        batch_size = 2
        window_size = 8

        # define number of ensemble
        num_ensemble = 32

        # load the model
        EKF_model = diff_EKF.EKF(batch_size)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        # dummy init to the model
        gt_pre, gt_now, raw_sensor = DataLoader.load_train_data(
            "./dataset/train_dataset_52.pkl", batch_size, window_size, norm=True
        )
        states = DataLoader.format_EKF_init_state(gt_pre[0], batch_size, dim_x)
        _ = EKF_model(raw_sensor[0], states)

        # # load pretrained weights
        # model.load_weights("DEnKF_v1.01_UR5003.h5")
        # EKF_model.layer[4].set_weights(model.layer[3].get_weights())
        # EKF_model.layer[4].trainable = False

        epoch = 20
        counter = 0
        for k in range(epoch):
            print("end-to-end wholemodel")
            print(
                "========================================= working on epoch %d =========================================: "
                % (k)
            )
            #
            steps = int(70000 / batch_size)
            for step in range(steps):
                counter = counter + 1
                # dummy init to the model
                gt_pre, gt_now, raw_sensor = DataLoader.load_train_data(
                    "./dataset/train_dataset_52.pkl", batch_size, window_size, norm=True
                )
                with tf.GradientTape(persistent=True) as tape:
                    loss = 0
                    #### apply multiple steps for EKF ####
                    for i in range(window_size):
                        start = time.time()
                        if i == 0:
                            states = DataLoader.format_EKF_init_state(
                                gt_pre[i], batch_size, dim_x
                            )
                            out = EKF_model(raw_sensor[i], states)
                        else:
                            out = EKF_model(raw_sensor[i], states)
                        state_h = out[0]  # state output
                        P = out[1]  # covariance matrix
                        loss = get_loss._mse(gt_now[i] - state_h)  # end-to-end state
                        states = (out[0], out[1])  # update state
                        end = time.time()
                    if step % 2 == 0:
                        print(
                            "Training loss at step %d: %.4f (took %.3f seconds) "
                            % (step, float(loss), float(end - start))
                        )
                        # with train_summary_writer.as_default():
                        #     tf.summary.scalar("total_loss", loss, step=counter)
                        print("---")
                grads = tape.gradient(loss, EKF_model.trainable_weights)
                optimizer.apply_gradients(zip(grads, EKF_model.trainable_weights))

            if (k + 1) % epoch == 0:
                EKF_model.save_weights(
                    "./models/dEKF_"
                    + version
                    + "_"
                    + name[index]
                    + str(epoch).zfill(3)
                    + ".h5"
                )
                print("model is saved at this epoch")
            if (k + 1) % 2 == 0:
                EKF_model.save_weights(
                    "./models/dEKF_"
                    + version
                    + "_"
                    + name[index]
                    + str(k).zfill(3)
                    + ".h5"
                )
                print("model is saved at this epoch")

                # define batch_size
                test_batch_size = 1

                # load the model
                model_test = diff_EKF.EKF(test_batch_size)
                (
                    test_gt_pre,
                    test_gt_now,
                    test_raw_sensor,
                ) = DataLoader.load_test_data(
                    "./dataset/test_dataset_52.pkl",
                    test_batch_size,
                    window_size=100,
                    norm=True,
                )

                # load init state
                inputs = test_raw_sensor[0]
                init_states = DataLoader.format_EKF_init_state(
                    test_gt_pre[0], test_batch_size, dim_x
                )

                dummy = model_test(inputs, init_states)
                model_test.load_weights(
                    "./models/dEKF_"
                    + version
                    + "_"
                    + name[index]
                    + str(k).zfill(3)
                    + ".h5"
                )
                for layer in model_test.layers:
                    layer.trainable = False
                model_test.summary()

                """
                run a test demo and save the state of the test demo
                """
                N = test_gt_now.shape[0]
                data = {}
                data_save = []
                gt_save = []

                for t in range(N):
                    if t == 0:
                        states = init_states
                    raw_sensor = test_raw_sensor[t]
                    out = model_test(raw_sensor, states)
                    if t % 10 == 0:
                        print("---")
                        print(out[1])
                        print(test_gt_now[t])
                    states = (out[0], out[1])
                    state_out = np.array(out[0])
                    gt_out = np.array(test_gt_now[t])
                    data_save.append(state_out)
                    gt_save.append(gt_out)
                data["state"] = data_save
                data["gt"] = gt_save
                with open(
                    "./output/dEKF_"
                    + version
                    + "_"
                    + name[index]
                    + str(k).zfill(3)
                    + ".pkl",
                    "wb",
                ) as f:
                    pickle.dump(data, f)

    else:
        k_list = [5]


"""
load loss functions
"""
get_loss = diff_EKF.getloss()

"""
load data for training
"""
global name
name = ["Tensegrity"]

global index
index = 0

global version
version = "dEKF_v1.0"
old_version = version

# os.system("sudo rm -rf /tf/experiments/loss/" + version + "/")

# train_log_dir = "/tf/experiments/loss/" + version
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)


def main():
    training = True
    run_filter(training)

    # training = False
    # run_filter(training)


if __name__ == "__main__":
    main()
