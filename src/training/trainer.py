
import os.path
import time
import logging

import numpy as np
import theano
import theano.tensor as tt
import summary
import scipy.io

logger = logging.getLogger(__name__)

def grouper(iterable, n):
    return (iterable[i : i + n] for i in xrange(0, len(iterable) - n, n))

class Tester(object):
    
    def __init__(self, network, testing_sampler):
        
        self.network = network
        self.sampler = testing_sampler
        
        self.iters = []
        self.costs = []
        self.errors = []
        self.constraints = []
        
        self.best_iter = -1
        self.best_cost = np.inf
        self.best_error = np.inf
        self.procrustes_error = np.inf
    
    def validate(self, niter=None, save_model_path=None):
        
        current_cost = []
        current_error = []
        current_pred = []
        current_gt = []

        # Map
        for minibatch_x, minibatch_heatmap, minibatch_y in self.sampler.iterable(): 
            cost = self.network.compute_cost(minibatch_x, minibatch_heatmap, minibatch_y)
            error = self.network.compute_error(minibatch_x, minibatch_heatmap, minibatch_y)
            pred = self.network.compute_testing_output(minibatch_x, minibatch_heatmap)

            current_cost.append(cost)
            current_error.append(error)
            current_pred.append(pred)
            current_gt.append(minibatch_y)

            # import IPython
            # IPython.embed()

        # Reduce
        current_cost = np.mean(current_cost)
        current_error = np.mean(current_error)

        if niter is not None:
            self.iters.append(niter)
            self.costs.append(current_cost)
            self.errors.append(current_error)
            
            data_shape = np.shape(current_pred)
            current_pred = np.array(np.reshape(np.float32(current_pred), (data_shape[0] * data_shape[1], data_shape[2])))
            current_gt = np.array(np.reshape(np.float32(current_gt), (data_shape[0] * data_shape[1], data_shape[2])))
            current_pred = np.array(np.reshape(np.float32(current_pred), (data_shape[0] * data_shape[1], data_shape[2]/3, 3)))
            current_gt = np.array(np.reshape(np.float32(current_gt), (data_shape[0] * data_shape[1], data_shape[2]/3 , 3)))

            # Compute the error after Procrustes
            total_procrustes_error = 0
            for i in range(data_shape[0]*data_shape[1]):
                d, transformed, tform = self.procrustes(current_gt[i, :, :], current_pred[i, :, :], scaling=True)
                sqrdiff = np.square(current_gt[i, :, :] - transformed)
                procrustes_error = np.mean(np.sqrt(np.sum(sqrdiff, 1)))
                total_procrustes_error += procrustes_error
            mean_procrustes_error = total_procrustes_error / (data_shape[0]*data_shape[1])

            if current_error < self.best_error:
                filename_pred = os.path.join(save_model_path, "pred.mat")
                filename_gt = os.path.join(save_model_path, "gt.mat")
                logger.info("Saving predictions to {}...".format(filename_pred))
                scipy.io.savemat(filename_pred, mdict={'pred': current_pred})
                scipy.io.savemat(filename_gt, mdict={'gt': current_gt})

                self.best_cost = current_cost
                self.best_iter = niter
                self.best_error = current_error
                self.procrustes_error = mean_procrustes_error
        
        return current_cost, current_error, mean_procrustes_error

    def procrustes(self, X, Y, scaling=True, reflection='best'):

        n, m = X.shape
        ny, my = Y.shape
        muX = X.mean(0)
        muY = Y.mean(0)
        X0 = X - muX
        Y0 = Y - muY
        ssX = (X0 ** 2.).sum()
        ssY = (Y0 ** 2.).sum()

        # centred Frobenius norm
        normX = np.sqrt(ssX)
        normY = np.sqrt(ssY)

        # scale to equal (unit) norm
        X0 /= normX
        Y0 /= normY

        if my < m:
            Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

        # optimum rotation matrix of Y
        A = np.dot(X0.T, Y0)
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        V = Vt.T
        T = np.dot(V, U.T)

        if reflection is not 'best':
            # does the current solution use a reflection?
            have_reflection = np.linalg.det(T) < 0
            # if that's not what was specified, force another reflection
            if reflection != have_reflection:
                V[:, -1] *= -1
                s[-1] *= -1
                T = np.dot(V, U.T)

        traceTA = s.sum()

        if scaling:
            # optimum scaling of Y
            b = traceTA * normX / normY
            # standarised distance between X and b*Y*T + c
            d = 1 - traceTA ** 2
            # transformed coords
            Z = normX * traceTA * np.dot(Y0, T) + muX

        else:
            b = 1
            d = 1 + ssY / ssX - 2 * traceTA * normY / normX
            Z = normY * np.dot(Y0, T) + muX

        # transformation matrix
        if my < m:
            T = T[:my, :]
        c = muX - b * np.dot(muY, T)

        # transformation values
        tform = {'rotation': T, 'scale': b, 'translation': c}

        return d, Z, tform

class Trainer(object):
    
    def __init__(self,
                 network,
                 solver,
                 learning_rate,
                 training_sampler,
                 testing_sampler=None,
                 validation_every=None,
                 save_every=None,
                 save_model_path=None):
        
        self.network = network
        self.sampler = training_sampler

        self.solver = solver
        self.learning_rate = learning_rate
        self.iter = 0
        self.iters_per_epoch = self.sampler.iters_per_epoch
        
        self.save_model_path = save_model_path
        
        self.training_iters = []
        self.training_costs = []
        self.training_constraints = []
        self.training_times = []
        self.testing_iters = []
        self.testing_costs = []
        self.testing_errors = []
        self.testing_constraints = []

        # Summary for registering data
        self.summary = summary.Summary()
        
        self.tester = None
        if testing_sampler is not None:
            self.tester = Tester(network, testing_sampler)
            self.testing_iters = self.tester.iters
            self.testing_costs = self.tester.costs
            self.testing_errors = self.tester.errors
            self.testing_constraints = self.tester.constraints
        
        self.validation_every = validation_every
        if self.validation_every is None:
            self.validation_every = self.iters_per_epoch
        
        self.save_every = save_every
        if self.save_every is None:
            self.save_every = self.iters_per_epoch
        
        self.saved_at = []
            
    
    def save_network_and_solver(self):
        
        if self.save_model_path is None:
            return
        
        if self.iter in self.saved_at:
            # Do not save same model twice
            return

        filename_fuse = os.path.join(self.save_model_path, "net{:05}.gz".format(self.iter))
        logger.info("Saving fusion network to {}...".format(filename_fuse))
        self.network.save_state(filename_fuse)

        filename = os.path.join(self.save_model_path, "solver_{:05}.gz".format(self.iter))
        logger.info("Saving solver to {}...".format(filename))
        self.solver.save(filename)

        logger.info("Saving summary to {}...".format(filename))
        self.summary.save(os.path.join(self.save_model_path, "summary.npy"), backup=True)
        
        self.saved_at.append(self.iter)

    def load_iter(self, niter):

        if self.save_model_path is None:
            raise ValueError("`save_model_path` not set; cannot load a previous state at given iteration")

        network_filename = os.path.join(self.save_model_path, "net{:05}.gz.npz".format(niter))
        solver_filename = os.path.join(self.save_model_path, "solver_{:05}.gz.npz".format(niter))
        summary_filename = os.path.join(self.save_model_path, "summary.npy")

        self.load(network_filename, solver_filename, summary_filename)

        self.iter = niter

    def load(self, network_filename, solver_filename, summary_filename):

        logger.info("Loading network from {}...".format(network_filename))
        self.network.load_state(network_filename)

        logger.info("Loading solver state from {}...".format(solver_filename))
        self.solver.load(solver_filename)

        logger.info("Loading summary from {}...".format(summary_filename))
        self.summary.load(summary_filename)
    
    def run_validation(self):
        
        if self.tester is None:
            return
        
        logger.info("Validating network at iteration {}...".format(self.iter))
        
        val_cost, val_error, proc_error = self.tester.validate(self.iter,self.save_model_path)
        logger.info("\tValidation cost: {}".format(val_cost))
        logger.info("\tValidation error: {}".format(val_error))
        logger.info("\tValidation error after procrustes: {}".format(proc_error))

        self.summary.register("testing_cost", self.iter, val_cost)
        self.summary.register("testing_error", self.iter, val_error)

        best = False
        if self.tester.best_iter == self.iter:
            logger.info("\tBest model so far!")
            best = True
        
        if self.save_model_path is not None:
            np.savez(os.path.join(self.save_model_path, "costs.npz"),
                training_iters=self.training_iters,
                training_costs=self.training_costs,
                training_times=self.training_times,
                testing_iters=self.testing_iters,
                testing_costs=self.testing_costs,
                testing_errors=self.testing_errors,
                testing_constraints=self.testing_constraints)

        return best, val_error
    
    def train(self, niters, print_every=0, maxtime=np.inf):
        
        tic = time.clock()
        best_iter = 0
        best_error = np.inf

        learning_rate = self.learning_rate

        for _ in xrange(niters):
            
            num_epoch, epoch_iter = divmod(self.iter, self.iters_per_epoch)
            
            # New epoch
            if epoch_iter == 0:
                logger.info("Starting epoch {}.".format(num_epoch))
                self.sampler.shuffle()
            
            # Get the minibatch
            minibatch_x, minibatch_heatmap, minibatch_y = self.sampler.get_minibatch(self.iter)

            # Solver step
            cost = self.solver.step(learning_rate, minibatch_x, minibatch_heatmap, minibatch_y)
            # mean_cosine = self.network.compute_mean_cosine(minibatch_x, minibatch_heatmap)
            # ortho_cost = self.network.compute_ortho_cost(minibatch_x, minibatch_heatmap)
            constraints = 0
            
            time_elapsed = time.clock() - tic

            # Register data
            self.summary.register("training_cost", self.iter, cost)
            self.summary.register("training_time", self.iter, time_elapsed)
                        
            self.training_iters.append(self.iter)
            self.training_costs.append(cost)
            self.training_constraints.append(constraints)
            self.training_times.append(time_elapsed)
            
            # logger.info(info)
            if print_every > 0 and self.iter % print_every == 0:
                logger.info("Iteration {} ({}/{} in epoch {})...".format(
                                self.iter, epoch_iter, self.iters_per_epoch, num_epoch))
                logger.info("\tTraining loss: {}".format(cost))
                # logger.info("\tMean cosine: {}".format(mean_cosine))
                # logger.info("\tOrthogonality training loss: {}".format(ortho_cost))

            self.iter += 1
            
            if time_elapsed > maxtime:
                logger.info("Maximum time reached!")
                break
            
            # Validation
            if self.iter % self.validation_every == 0:
                best_flag, error = self.run_validation()
                
                if best_flag:
                    self.save_network_and_solver()
                    best_iter = self.iter
                    best_error = error

            # Save model and solver
            if self.iter % self.save_every == 0:
                self.save_network_and_solver()

        self.summary.register("best", best_iter, best_error)
        self.save_network_and_solver()
        return best_iter, best_error

        
