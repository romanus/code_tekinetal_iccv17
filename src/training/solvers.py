
import warnings

import numpy as np
import theano
import theano.tensor as tt

floatX = theano.config.floatX

class MinresQLPSolver(object):
    
    def __init__(self,
                 residuals,
                 params,
                 args,
                 lmb=None,
                 maxiter=100):
        
        raise NotImplementedError
        
        def compute_JJp(xs):
            
            Jp = tt.Rop(residuals, params, xs)
            JJp = tt.Lop(residuals, params, Jp)
            
            if lmb is not None:
                JJp = [JJp_i + lmb * p_i for JJp_i, p_i in zip(JJp, xs)]
            
            return JJp, []
        
        Jr = tt.Lop(residuals, params, residuals)
        Jr = [-Jr_i for Jr_i in Jr]
        
        param_shapes = [param_i.get_value().shape for param_i in params]
        
        # self.cg = ConjugateGradient(compute_JJp, Jr, param_shapes, args, args)
        self.cg = MinresQLP(
            compute_JJp, Jr, param_shapes, args, args,
            maxit=maxiter
        )
        
        self.update_params = theano.function([], [],
                                updates=[(param_i, param_i + x_i) for param_i, x_i in zip(params, self.cg.xs)])
    
    def step(self, *args, **kwargs):
        
        eta = kwargs.get("eta", None)
        if eta is not None:
            self.eta.set_value(eta)
        
        constr_weight = kwargs.get("constr_weight", None)
        if constr_weight is not None:
            self.constr_weight.set_value(constr_weight)
        
        extras, _, self.flag, self.iters = self.cg.execute(args, args)
        
        # Update params
        if not kwargs.get("do_not_update", False):
            self.update_params()
        
        # Do it several times. For fun.
        for i in xrange(self.solver_iters - 1):
            aux, _, self.flag, self.iters = self.cg.execute(args, args)
            print aux[0], np.median(np.abs(aux[1]))
            if not kwargs.get("do_not_update", False):
                self.update_params()
        
        return extras
    
    def message(self):
        
        return messages[self.flag + 1]
    
    def save(self, filename):
        # Nothing to save
        pass
    
    def load(self, filename):
        # Nothing to load
        pass

class ConstraintsMinresQLPSolver(MinresQLPSolver):
    
    def __init__(self,
                 grads_or_residuals,
                 constraints,
                 params,
                 args,
                 shape_constraints,
                 costs=[],
                 init_eta=None,
                 maxiter=100,
                 mode="LM"):
        
        warnings.warn("Using obsolete class {}. Use GradientMinresQLPSolver or LMMinresQLPSolver instead.".format(self.__class__.__name__))
        
        # if mode == "LM":
        residuals = grads_or_residuals
        # else:
        # grads = grads_or_residuals
        
        if init_eta is not None:
            self.eta = theano.shared(np.array(init_eta, dtype=floatX), name='eta')
        
        def compute_Ax(xs):
            
            if mode == "LM":
                Jp = tt.Rop(residuals, params, xs[:-1])
                JJp = tt.Lop(residuals, params, Jp)
                
                if init_eta is not None:
                    JJp = [JJp_i + self.eta * p_i for JJp_i, p_i in zip(JJp, xs[:-1])]
            elif mode == "gradient":
                JJp = [self.eta * p_i for p_i in xs[:-1]]
            else:
                raise "Unknown mode '{}'".format(mode)
            
            Al = tt.Lop(constraints, params, xs[-1])
            Ap = tt.Rop(constraints, params, xs[:-1])
            Mp = map(tt.add, JJp, Al) + [Ap]
            
            return Mp, []
        
        # if mode == "LM":
        Jr = tt.Lop(residuals, params, residuals)
        # else:
        # Jr = grads
        b = Jr + [constraints]
        b = [-b_i for b_i in b]
        
        param_shapes = [param_i.get_value().shape for param_i in params]
        param_shapes.append(shape_constraints)
        
        # args = residuals_args + constraints_args
        # self.cg = ConjugateGradient(compute_Ax, b, param_shapes, args, args)
        self.cg = MinresQLP(
            compute_Ax, b, param_shapes, args, args,
            maxit=maxiter,
            extra_functions=costs
        )
        
        self.update_params = theano.function([], [],
                                updates=[(param_i, param_i + x_i) for param_i, x_i in zip(params, self.cg.xs)])
    

class GradientMinresQLPSolver(MinresQLPSolver):
    
    def __init__(self,
                 params,
                 grads,
                 args,
                 costs=[],
                 constraints=[],
                 constraint_shapes=[],
                 init_eta=1e4,
                 mu=None,
                 maxiter=100,
                 solver_iters=1):
        
        assert len(constraints) == len(constraint_shapes)
        
        len_params = len(params)
        param_shapes = [param_i.get_value().shape for param_i in params] + constraint_shapes
        
        Jr = grads
        if mu is not None:
            self.prev_xs = [theano.shared(
                                np.zeros(param_i.get_value().shape, dtype=param_i.get_value().dtype),
                                borrow=True)
                            for param_i in params]
            
            Jr = [Jr_i - mu * prev_x_i for Jr_i, prev_x_i in zip(Jr, self.prev_xs)]
        
        # Jr = [theano.shared(np.zeros(sh_i, dtype=floatX)) for sh_i in param_shapes[:len_params]]
        b = Jr + constraints
        b = [-b_i for b_i in b]
        
        self.eta = theano.shared(np.array(init_eta, dtype=floatX))
        
        def compute_Ax(xs):
            
            JJp = [self.eta * p_i for p_i in xs[:len_params]]
            
            if constraints:
                Al = tt.Lop(constraints, params, xs[len_params:])
                Ap = tt.Rop(constraints, params, xs[:len_params])
                JJp = map(tt.add, JJp, Al) + Ap
            
            return JJp, []
        
        self.cg = MinresQLP(
            compute_Ax, b, param_shapes, args, args,
            maxit=maxiter,
            extra_functions=costs
        )
        
        # self.xs = [tt.TensorType(floatX, (False,) * i.ndim)() for i in params]
        # self.xs += [tt.TensorType(floatX, (False,) * len(i))() for i in constraint_shapes]
        self.compute_Ax = theano.function(args, compute_Ax(self.cg.xs)[0], on_unused_input='warn')
        self.compute_b = theano.function(args, b, on_unused_input='warn')
        
        self.update_params = theano.function([], [],
                                updates=[(param_i, param_i + x_i) for param_i, x_i in zip(params, self.cg.xs)])
        
        self.solver_iters = solver_iters
    

class AdamMinresQLPSolver(object):
    
    def __init__(self,
                 params,
                 grads,
                 args,
                 costs=[],
                 constraints=[],
                 constraint_shapes=[],
                 init_eta=1e3,
                 beta1=0.9,
                 beta2=0.999,
                 eps=1e-9,
                 maxiter=100,
                 solver_iters=1):
        
        assert len(constraints) == len(constraint_shapes)
        
        self.beta1 = beta1
        self.beta2 = beta2
        
        len_params = len(params)
        param_shapes = [param_i.get_value().shape for param_i in params] + constraint_shapes
        
        self.m = [theano.shared(np.zeros(param_i.get_value().shape, dtype=param_i.get_value().dtype), borrow=True)
                    for param_i in params]
        self.v = [theano.shared(np.zeros(param_i.get_value().shape, dtype=param_i.get_value().dtype), borrow=True)
                    for param_i in params]
        self.it = theano.shared(np.array(0.0, dtype=floatX))
        self.eta = theano.shared(np.array(init_eta, dtype=floatX))
        
        # Jr = [grad_i for grad_i in grads]
        
        b = self.m + constraints
        b = [-b_i for b_i in b]
        
        # fix = tt.sqrt(1 - self.beta2 ** self.it) / (1 - self.beta1 ** self.it)
        fix = (1 - self.beta1 ** self.it) / tt.sqrt(1 - self.beta2 ** self.it)
        
        def compute_Ax(xs):
            
            JJp = [self.eta * fix * (tt.sqrt(v_i) + eps) * p_i for v_i, p_i in zip(self.v, xs[:len_params])]
            
            if constraints:
                Al = tt.Lop(constraints, params, xs[len_params:])
                Ap = tt.Rop(constraints, params, xs[:len_params])
                JJp = map(tt.add, JJp, Al) + Ap
            
            return JJp, []
        
        self.cg = MinresQLP(
            compute_Ax, b, param_shapes, args, args,
            maxit=maxiter,
            extra_functions=costs
        )
        
        m_update = [(m_i, self.beta1 * m_i + (1 - self.beta1) * grad_i)
                        for m_i, grad_i in zip(self.m, grads)]
        v_update = [(v_i, self.beta2 * v_i + (1 - self.beta2) * grad_i ** 2)
                        for v_i, grad_i in zip(self.v, grads)]
        it_update = [(self.it, self.it + 1.0)]
        
        self.update_mv = theano.function(args, [], updates=m_update + v_update + it_update,
                                    allow_input_downcast=True,
                                    name="update_mv",
                                    on_unused_input="warn")
        
        self.update_params = theano.function([], [],
                                updates=[(param_i, param_i + x_i) for param_i, x_i in zip(params, self.cg.xs)])
        
        self.solver_iters = solver_iters
    
    def step(self, *args, **kwargs):
        
        self.update_mv(*args)
        
        extras, _, self.flag, self.iters = self.cg.execute(args, args)
        
        # Update params
        if not kwargs.get("do_not_update", False):
            self.update_params()
        
        # Do it several times. For fun.
        for i in xrange(self.solver_iters - 1):
            self.update_mv(*args)
            aux, _, self.flag, self.iters = self.cg.execute(args, args)
            print aux[0], np.median(np.abs(aux[1]))
            if not kwargs.get("do_not_update", False):
                self.update_params()
        
        return extras
    
    def message(self):
        return messages[self.flag + 1]
    
    def save(self, filename):
        np.savez(filename,
            m=[m_i.get_value() for m_i in self.m],
            v=[v_i.get_value() for v_i in self.v],
            it=self.it.get_value())
    
    def load(self, filename):
        
        with np.load(filename) as aux:
            
            for m_i, aux_i in zip(self.m, aux["m"]):
                m_i.set_value(aux_i)
            for v_i, aux_i in zip(self.v, aux["v"]):
                v_i.set_value(aux_i)
            
            self.it.set_value(aux["it"])
    

class LMMinresQLPSolver(MinresQLPSolver):
    
    def __init__(self,
                 params,
                 residuals,
                 args,
                 costs=[],
                 constraints=[],
                 constraint_shapes=[],
                 init_eta=None,
                 init_constr_weight=1.0,
                 maxiter=100,
                 solver_iters=1):
        
        assert len(constraints) == len(constraint_shapes)
        
        len_params = len(params)
        param_shapes = [param_i.get_value().shape for param_i in params] + constraint_shapes
        
        Jr = tt.Lop(residuals, params, residuals)
        
        self.eta = theano.shared(np.array(init_eta, dtype=floatX))
        self.constr_weight = theano.shared(np.array(init_constr_weight, dtype=floatX))
        
        b = Jr + [c * self.constr_weight for c in constraints]
        b = [-b_i for b_i in b]
        
        def compute_Ax(xs):
            
            Jp = tt.Rop(residuals, params, xs[:len_params])
            JJp = tt.Lop(residuals, params, Jp)
            
            if init_eta is not None:
                JJp = [JJp_i + self.eta * p_i for JJp_i, p_i in zip(JJp, xs[:len_params])]
            
            if constraints:
                Al = tt.Lop(constraints, params, xs[len_params:])
                Ap = tt.Rop([c * self.constr_weight for c in constraints], params, xs[:len_params])
                JJp = map(tt.add, JJp, Al) + Ap
            
            return JJp, []
        
        self.cg = MinresQLP(
            compute_Ax, b, param_shapes, args, args,
            maxit=maxiter,
            extra_functions=costs
        )
        
        self.solver_iters = solver_iters
        
        self.compute_Ax = theano.function(args, compute_Ax(self.cg.xs)[0], on_unused_input='warn')
        self.compute_b = theano.function(args, b, on_unused_input='warn')
        
        self.update_params = theano.function([], [],
                                updates=[(param_i, param_i + x_i) for param_i, x_i in zip(params, self.cg.xs)])

    

class AdagradSolver(object):
    
    def __init__(self,
                 params,
                 grads,
                 grad_vars,
                 cost=[],
                 eps=1e-8,
                 init_lr=1e-3):
        
        self.eps = eps
        
        self.m = [theano.shared(np.zeros(param_i.get_value().shape, dtype=param_i.get_value().dtype), borrow=True)
                    for param_i in params]
        self.v = [theano.shared(np.zeros(param_i.get_value().shape, dtype=param_i.get_value().dtype), borrow=True)
                    for param_i in params]
        self.learning_rate = theano.shared(np.array(init_lr, dtype=floatX), name='lr')
        
        m_update = [(m_i, grad_i)
                        for m_i, grad_i in zip(self.m, grads)]
        v_update = [(v_i, v_i + grad_i ** 2)
                        for v_i, grad_i in zip(self.v, grads)]
        
        param_update = [(param_i, param_i - self.learning_rate * m_i / (tt.sqrt(v_i) + eps))
                        for param_i, m_i, v_i in zip(params, self.m, self.v)]
        
        self.f_propagation = theano.function(grad_vars, cost, updates=m_update + v_update)
        self.f_update = theano.function([], [], updates=param_update)
    
    def step(self, *vars, **kwargs):
        
        if kwargs.has_key("learning_rate"):
            self.learning_rate.set_value(kwargs.get("learning_rate"))
        
        cost = self.f_propagation(*vars)
        self.f_update()
        
        return cost
    
    def save(self, filename):
        
        np.savez(filename,
                 m=[m_i.get_value() for m_i in self.m],
                 v=[v_i.get_value() for v_i in self.v])
    
    def load(self, filename):
        
        aux = np.load(filename)
        
        for m_i, aux_i in zip(self.m, aux["m"]):
            m_i.set_value(aux_i)
        for v_i, aux_i in zip(self.v, aux["v"]):
            v_i.set_value(aux_i)
    

#class AdamSolver(object):
    
    #def __init__(self,
                 #params,
                 #grads,
                 #grad_vars,
                 #cost=[],
                 #beta1=0.9,
                 #beta2=0.999,
                 #eps=1e-8,
                 #init_lr=1e-3):
        
        #self.beta1 = beta1
        #self.beta2 = beta2
        #self.eps = eps
        
        #self.m = [theano.shared(np.zeros(param_i.get_value().shape, dtype=param_i.get_value().dtype), borrow=True)
                    #for param_i in params]
        #self.v = [theano.shared(np.zeros(param_i.get_value().shape, dtype=param_i.get_value().dtype), borrow=True)
                    #for param_i in params]
        #self.it = theano.shared(np.asarray(0.0, dtype=floatX))
        #self.learning_rate = theano.shared(np.array(init_lr, dtype=floatX), name='lr')
        
        #m_update = [(m_i, self.beta1 * m_i + (1 - self.beta1) * grad_i)
                        #for m_i, grad_i in zip(self.m, grads)]
        #v_update = [(v_i, self.beta2 * v_i + (1 - self.beta2) * grad_i ** 2)
                        #for v_i, grad_i in zip(self.v, grads)]
        #it_update = [(self.it, self.it + 1.0)]
        
        #fix = tt.sqrt(1 - self.beta2 ** self.it) / (1 - self.beta1 ** self.it)
        
        #param_update = [(param_i, param_i - self.learning_rate * fix * m_i / (tt.sqrt(v_i) + eps))
                        #for param_i, m_i, v_i in zip(params, self.m, self.v)]
        
        #self.f_propagation = theano.function(grad_vars, cost, updates=m_update + v_update + it_update)
        #self.f_update = theano.function([], [], updates=param_update)

    #def step(self, learning_rate, *vars):
        
        ##if kwargs.has_key("learning_rate"):
            ##self.learning_rate.set_value(kwargs.get("learning_rate"))
        
        #cost = self.f_propagation(*vars)
        #self.f_update()
        
        #return cost
    
    ##def step(self, *vars, **kwargs):
        
        ##if kwargs.has_key("learning_rate"):
            ##self.learning_rate.set_value(kwargs.get("learning_rate"))
        
        ##cost = self.f_propagation(*vars)
        ##self.f_update()
        
        ##return cost
    
    #def save(self, filename):
        
        #np.savez(filename,
                 #m=[m_i.get_value() for m_i in self.m],
                 #v=[v_i.get_value() for v_i in self.v],
                 #it=self.it.get_value())
    
    #def load(self, filename):
        
        #aux = np.load(filename)
        
        #for m_i, aux_i in zip(self.m, aux["m"]):
            #m_i.set_value(aux_i)
        #for v_i, aux_i in zip(self.v, aux["v"]):
            #v_i.set_value(aux_i)
        
        #self.it.set_value(aux["it"])


class SGDSolver(object):
    def __init__(self,
                 params,
                 grads,
                 grad_vars,
                 cost=[],
                 momentum=0.9,
                 init_lr=1e-3):

        self.momentum = momentum
        self.m = [theano.shared(np.zeros(param_i.get_value().shape, dtype=param_i.get_value().dtype), borrow=True)
                  for param_i in params]

        self.learning_rate = theano.shared(np.array(init_lr, dtype=floatX), name='lr')
        m_update = [(m_i, self.momentum * m_i - self.learning_rate * grad_i) for m_i, grad_i in zip(self.m, grads)]

        param_update = [(param_i, param_i + m_i)
                        for param_i, m_i in zip(params, self.m)]

        self.f_propagation = theano.function(grad_vars, cost, updates=m_update)
        self.f_update = theano.function([], [], updates=param_update)

    def step(self, *vars, **kwargs):

        if kwargs.has_key("learning_rate"):
            self.learning_rate.set_value(kwargs.get("learning_rate"))

        cost = self.f_propagation(*vars)
        self.f_update()

        return cost

    def save(self, filename):

        np.savez(filename, m=[m_i.get_value() for m_i in self.m])

    def load(self, filename):

        aux = np.load(filename)

        for m_i, aux_i in zip(self.m, aux["m"]):
            m_i.set_value(aux_i)

# class SGDSolver(object):
#
#     def __init__(self,
#                  params,
#                  grads,
#                  grad_vars,
#                  cost=[],
#                  momentum=0.9):
#
#         self.momentum = momentum
#         # self.m = [theano.shared(np.zeros(param_i.get_value().shape, dtype=param_i.get_value().dtype), borrow=True)
#         #           for param_i in params]
#         self.m = [theano.shared(np.zeros(param_i.get_value().shape, dtype=theano.config.floatX), borrow=True)
#                   for param_i in params]
#
#         self.learning_rate = tt.scalar('lr')
#         m_update = [(m_i, self.momentum * m_i - self.learning_rate * grad_i) for m_i, grad_i in zip(self.m, grads)]
#
#         param_update = [(param_i, param_i + m_i)
#                         for param_i, m_i in zip(params, self.m)]
#
#         self.f_propagation = theano.function(grad_vars, cost, updates=m_update, allow_input_downcast=True)
#         self.f_update = theano.function([self.learning_rate], [], updates=param_update, allow_input_downcast=True)
#
#
#     def step(self, learning_rate, *vars):
#
#         cost = self.f_propagation(*vars)
#         self.f_update(learning_rate)
#
#         return cost
#
#     # def step(self, learning_rate, minibatch_x, minibatch_heatmap, minibatch_y):
#     #
#     #     cost = self.f_propagation([minibatch_x, minibatch_heatmap, minibatch_y])
#     #     self.f_update(learning_rate)
#     #
#     #     return cost
#
#     def save(self, filename):
#
#         np.savez(filename, m=[m_i.get_value() for m_i in self.m])
#
#     def load(self, filename):
#
#         aux = np.load(filename)
#
#         for m_i, aux_i in zip(self.m, aux["m"]):
#             m_i.set_value(aux_i)

class AdamSolver(object):
    
    def __init__(self,
                 params,
                 grads,
                 grad_vars,
                 cost=[],
                 #cost2d=[],
                 #cost3d=[],
                 beta1=0.9,
                 beta2=0.999,
                 eps=1e-8):
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        #self.m = [theano.shared(np.zeros(param_i.get_value().shape, dtype=param_i.get_value().dtype), borrow=True)
                    #for param_i in params]
        #self.v = [theano.shared(np.zeros(param_i.get_value().shape, dtype=param_i.get_value().dtype), borrow=True)
                    #for param_i in params]
        self.m = [theano.shared(np.zeros(param_i.get_value().shape, dtype=theano.config.floatX), borrow=True)
                    for param_i in params]
        self.v = [theano.shared(np.zeros(param_i.get_value().shape, dtype=theano.config.floatX), borrow=True)
                    for param_i in params]        
        self.it = theano.shared(np.asarray(0.0, dtype=theano.config.floatX))
        self.learning_rate = tt.scalar('lr')
        
        m_update = [(m_i, self.beta1 * m_i + (1 - self.beta1) * grad_i)
                        for m_i, grad_i in zip(self.m, grads)]
        v_update = [(v_i, self.beta2 * v_i + (1 - self.beta2) * grad_i ** 2)
                        for v_i, grad_i in zip(self.v, grads)]
        it_update = [(self.it, self.it + 1.0)]
        
        fix = tt.sqrt(1 - self.beta2 ** self.it) / (1 - self.beta1 ** self.it)
        
        param_update = [(param_i, param_i - self.learning_rate * fix * m_i / (tt.sqrt(v_i) + eps))
                        for param_i, m_i, v_i in zip(params, self.m, self.v)]
        
        self.f_propagation = theano.function(grad_vars, cost, updates=m_update + v_update + it_update, allow_input_downcast=True)
        #self.f_propagation2d = theano.function([grad_vars[0], grad_vars[2]], cost2d)
        #self.f_propagation3d = theano.function([grad_vars[0], grad_vars[1]], cost3d)        
        self.f_update = theano.function([self.learning_rate], [], updates=param_update, allow_input_downcast=True)
    
    def step(self, learning_rate, *vars):
        
        cost = self.f_propagation(*vars)
        self.f_update(learning_rate)
        
        return cost   
    
    #def step2d(self, minibatch_x, minibatch_y2d):
        
        #cost2d = self.f_propagation2d(minibatch_x, minibatch_y2d)
        
        #return cost2d
    
    #def step3d(self, minibatch_x, minibatch_y3d):
        
        #cost3d = self.f_propagation3d(minibatch_x, minibatch_y3d)
                
        #return cost3d        

    def save(self, filename):
        
        np.savez(filename,
                 m=[m_i.get_value() for m_i in self.m],
                 v=[v_i.get_value() for v_i in self.v],
                 it=self.it.get_value())
    
    def load(self, filename):
        
        aux = np.load(filename)
        
        for m_i, aux_i in zip(self.m, aux["m"]):
            m_i.set_value(aux_i)
        for v_i, aux_i in zip(self.v, aux["v"]):
            v_i.set_value(aux_i)
        
        self.it.set_value(aux["it"])

    def load_heatmap_solver(self, filename):
        
        aux = np.load(filename)
        #print len(aux["m"])
        for m_i, aux_i in zip(self.m[:16], aux["m"]):
            m_i.set_value(aux_i)
        for v_i, aux_i in zip(self.v[:16], aux["v"]):
            v_i.set_value(aux_i)
        
        self.it.set_value(aux["it"])
        print 'Loaded solver parameters from {}'.format(filename)
        
        
    def load_lift_solver(self, filename):
        
        aux = np.load(filename)
        #print len(aux["m"])
        
        for m_i, aux_i in zip(self.m[16:], aux["m"]):
            m_i.set_value(aux_i)
        for v_i, aux_i in zip(self.v[16:], aux["v"]):
            v_i.set_value(aux_i)
        
        self.it.set_value(aux["it"])
        print 'Loaded solver parameters from {}'.format(filename)
