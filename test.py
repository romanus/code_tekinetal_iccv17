    
import logging
import numpy as np
import os.path
import datetime
import argparse

from src.data import load_testing, load_training
from src.training.sampler import AsyncLoaderSampler
from src.net.trainable_fusion import FusionNet
from src.helpers import plot_pose

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

logger = logging.getLogger(__name__)

def run(args):

    # Get initial parameters
    minibatch_size = 1
    image_shape    = (128, 128)
    heatmap_shape  = (64, 64)
    dpi            = 36.
    xinch          = image_shape[0] / dpi
    yinch          = image_shape[1] / dpi
    limits_x       = (-900, 900)
    limits_y       = (-900, 900)
    limits_z       = (-900, 900)

    # Define the network
    network = FusionNet(args.nettype)
    network.load_state(args.netpath)

    # Sample test data
    testing_data  = load_testing(val_ims=args.val_ims, val_hms=args.val_hms, keys=["val_x", "val_y3d", "heatmaps"])
    testing_sampler = AsyncLoaderSampler(testing_data,
                                         minibatch_size=minibatch_size,
                                         loadblock_size=50,
                                         training=0,
                                         minibatch_shapes=[(-1, 3) + image_shape, (-1, 51), (-1, 16) + heatmap_shape])

    # Print some testing information
    logger.info("Testing data information...")
    logger.info("{} data points:".format(testing_sampler.num_samples//testing_sampler.sampling))
    logger.info("\t{} minibatches".format(testing_sampler.num_batches))
    logger.info("\t{} data points per minibatch".format(testing_sampler.minibatch_size))
    
    # Visualizer
    fig = plt.figure(figsize=(xinch*5, yinch))
    ax_img = fig.add_subplot(1, 4, 1)
    ax_hm = fig.add_subplot(1, 4, 2)
    ax_pose_gt = fig.add_subplot(1, 4, 4, projection='3d')
    ax_pose_gt.view_init(elev=10, azim=-94)
    ax_pose_pred = fig.add_subplot(1, 4, 3, projection='3d')
    ax_pose_pred.view_init(elev=10, azim=-94)
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    logger.info("Parsing data, predicting pose and visualizing...")
    count = 0
    for sampler in testing_sampler.iterable():

        # Predict
        prediction = network.compute_testing_output(sampler[0], sampler[1])

        # # Visualize
        ax_img.cla()
        ax_hm.cla()
        ax_pose_gt.cla()
        ax_pose_pred.cla()
        ax_img.imshow(np.clip(np.transpose(sampler[0][0], (1, 2, 0)), 0.0, 1.0))
        ax_img.axis('off')
        ax_img.set_aspect('equal')
        ax_img.set_title('Image', fontsize='medium')
        ax_hm.imshow(np.clip(np.sum(sampler[1][0], axis=0).squeeze(), 0.0, 1.0))
        ax_hm.axis('off')
        ax_hm.set_aspect('equal')
        ax_hm.set_title('Heatmap', fontsize='medium')
        plot_pose(ax_pose_gt, sampler[2], joint_style='b.', bone_style='b-')
        ax_pose_gt._axis3don = False
        ax_pose_gt.get_xaxis().set_ticklabels([])
        ax_pose_gt.get_yaxis().set_ticklabels([])
        ax_pose_gt.set_aspect('equal')
        ax_pose_gt.set_title('GT', fontsize='medium')
        ax_pose_gt.set_xlim(limits_x)
        ax_pose_gt.set_ylim(limits_y)
        ax_pose_gt.set_zlim(limits_z)
        plot_pose(ax_pose_pred, prediction, joint_style='b.', bone_style='b-')
        ax_pose_pred._axis3don = False
        ax_pose_pred.get_xaxis().set_ticklabels([])
        ax_pose_pred.get_yaxis().set_ticklabels([])
        ax_pose_pred.set_aspect('equal')
        ax_pose_pred.set_title('Prediction', fontsize='medium')
        ax_pose_pred.set_xlim(limits_x)
        ax_pose_pred.set_ylim(limits_y)
        ax_pose_pred.set_zlim(limits_z)
        plt.pause(1)

        # Compute error
        residuals = prediction - sampler[2]
        sqrdiff = np.square(residuals)
        num_samples = sqrdiff.shape[0]
        num_joints = sqrdiff.shape[1] / 3
        sqrdiff = np.reshape(sqrdiff, (num_samples, num_joints, 3))
        error = np.mean(np.sqrt(np.sum(sqrdiff, 2)))
        logger.info("Pose error {}/{}: {}.".format(count, testing_sampler.num_batches, error))
        count += 1


def main():
    logging.basicConfig(level=logging.INFO, format="\x1b[32;1m[%(asctime)s:%(name)s:%(levelname)s]\x1b[0m %(message)s")

    # Arguments to parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nettype', type=int, default=1)
    parser.add_argument('--netpath', type=str, default='model/h36m_model_A_14.npz')
    parser.add_argument('--val_ims', type=str, default='data/h36m_data_imgs_A_14_val.h5')
    parser.add_argument('--val_hms', type=str, default='data/h36m_data_hg_preds_A_14_val.h5')
    args = parser.parse_args()

    run(args)

if __name__ == "__main__":
    main()
