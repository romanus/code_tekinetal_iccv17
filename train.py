
import logging
import numpy as np
import os.path
import datetime
import argparse

from src.data import load_testing, load_training
from src.training.sampler import AsyncLoaderSampler
from src.net.trainable_fusion import FusionNet
from src.training import solvers, trainer
from src.helpers import makedirs

logger = logging.getLogger(__name__)

def run(args):

    # Get initial parameters
    minibatch_size = 128
    image_shape    = (128, 128)
    heatmap_shape  = (64, 64)

    # Define the network
    network = FusionNet(args.type, args.ibeta, args.reg_factor)

    # Get the loaders for the training and testing data
    training_data = load_training(train_ims=args.train_ims, train_hms=args.train_hms, keys=["train_x", "train_y3d", "heatmaps"])
    testing_data  = load_testing(val_ims=args.val_ims, val_hms=args.val_hms, keys=["val_x", "val_y3d", "heatmaps"])

    training_sampler = AsyncLoaderSampler(training_data,
                                          minibatch_size=minibatch_size,
                                          loadblock_size=50,
                                          training=1,
                                          minibatch_shapes=[(-1, 3) + image_shape, (-1, 51), (-1, 16) + heatmap_shape])
    testing_sampler  = AsyncLoaderSampler(testing_data,
                                          minibatch_size=minibatch_size,
                                          loadblock_size=50,
                                          training=0,
                                          minibatch_shapes=[(-1, 3) + image_shape, (-1, 51), (-1, 16) + heatmap_shape])

    # Print some training information
    logger.info("{} data points:".format(training_sampler.num_samples))
    logger.info("\t{} minibatches".format(training_sampler.iters_per_epoch))
    logger.info("\t{} data points per minibatch".format(training_sampler.minibatch_size))
   
    # Get the solver
    solver = solvers.AdamSolver(network.params, network.grads, [network.x1, network.x2, network.y], network.cost)

    # Create directory for saved models
    output_dir = "saved"
    if not os.path.exists(output_dir):
        makedirs(output_dir)

    # Get the trainer
    mytrainer = trainer.Trainer(network,
                                solver,
                                0.001,
                                training_sampler, testing_sampler,
                                validation_every=training_sampler.iters_per_epoch,
                                save_every=10*training_sampler.iters_per_epoch,
                                save_model_path=output_dir)

    # Train
    num_iters = 200 * training_sampler.iters_per_epoch
    best_iter, best_error = mytrainer.train(num_iters, print_every=10)

    # Print the results of the training
    logger.info("Best model obtained at iteration: {}".format(best_iter))
    logger.info("\tBest validation error: {}".format(best_error))

def main():
    
    logging.basicConfig(level=logging.INFO, format="\x1b[32;1m[%(asctime)s:%(name)s:%(levelname)s]\x1b[0m %(message)s")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=int, default=1)
    parser.add_argument('--reg_factor', type=int, default=5000)
    parser.add_argument('--ibeta', type=float, default=4)
    parser.add_argument('--train_ims', type=str,  default='data/h36m_data_imgs_A_14_train.h5')
    parser.add_argument('--train_hms', type=str, default='data/h36m_data_hg_preds_A_14_train.h5')
    parser.add_argument('--val_ims', type=str, default='data/h36m_data_imgs_A_14_val.h5')
    parser.add_argument('--val_hms', type=str, default='data/h36m_data_hg_preds_A_14_val.h5')
    args = parser.parse_args()

    run(args)

if __name__ == "__main__":
    main()
