# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.robo.rbayes -- TODO
============================================

.. module:: robo
    :platform: Unix
    :synopsis: TODO

TODO: Write long description
"""

import george
import numpy

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Integer, Real, Space

from robo.priors.default_priors import DefaultPrior
from robo.models.wrapper_bohamiann import WrapperBohamiann
from robo.models.gaussian_process import GaussianProcess
from robo.models.gaussian_process_mcmc import GaussianProcessMCMC
from robo.models.random_forest import RandomForest
from robo.maximizers.scipy_optimizer import SciPyOptimizer
from robo.maximizers.random_sampling import RandomSampling
from robo.maximizers.differential_evolution import DifferentialEvolution
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.acquisition_functions.ei import EI
from robo.acquisition_functions.pi import PI
from robo.acquisition_functions.log_ei import LogEI
from robo.acquisition_functions.lcb import LCB
from robo.acquisition_functions.marginalization import MarginalizationGPMCMC
from robo.initial_design import init_latin_hypercube_sampling


def build_bounds(space):
    lower = numpy.zeros(len(space.keys()))
    upper = numpy.zeros(len(space.keys()))
    for i, (name, dim) in enumerate(space.items()):
        lower[i], upper[i] = dim.interval()
        if dim.prior_name == 'reciprocal':
            lower[i] = numpy.log(lower[i])
            upper[i] = numpy.log(upper[i])

    return lower, upper


def build_optimizer(model, maximizer="random", acquisition_func="log_ei", maximizer_seed=1):
    """
    General interface for Bayesian optimization for global black box
    optimization problems.
    Parameters
    ----------
    maximizer: {"random", "scipy", "differential_evolution"}
        The optimizer for the acquisition function.
    acquisition_func: {"ei", "log_ei", "lcb", "pi"}
        The acquisition function
    maximizer_seed: int
        Seed for random number generator of the acquisition function maximizer
    Returns
    -------
        Optimizer
        :param maximizer_seed:
        :param acquisition_func:
        :param maximizer:
        :param model:
    """

    if acquisition_func == "ei":
        a = EI(model)
    elif acquisition_func == "log_ei":
        a = LogEI(model)
    elif acquisition_func == "pi":
        a = PI(model)
    elif acquisition_func == "lcb":
        a = LCB(model)
    else:
        raise ValueError("'{}' is not a valid acquisition function"
                         .format(acquisition_func))

    if isinstance(model, GaussianProcessMCMC):
        acquisition_func = MarginalizationGPMCMC(a)
    else:
        acquisition_func = a

    maximizer_rng = numpy.random.RandomState(maximizer_seed)
    if maximizer == "random":
        max_func = RandomSampling(acquisition_func, model.lower, model.upper, rng=maximizer_rng)
    elif maximizer == "scipy":
        max_func = SciPyOptimizer(acquisition_func, model.lower, model.upper, rng=maximizer_rng)
    elif maximizer == "differential_evolution":
        max_func = DifferentialEvolution(acquisition_func, model.lower, model.upper,
                                         rng=maximizer_rng)
    else:
        raise ValueError("'{}' is not a valid function to maximize the "
                         "acquisition function".format(maximizer))

    # NOTE: Internal RNG of BO won't be used.
    # NOTE: Nb of initial points won't be used within BO, but rather outside
    bo = BayesianOptimization(lambda: None, model.lower, model.upper,
                              acquisition_func, model, max_func,
                              initial_points=None, rng=None,
                              initial_design=init_latin_hypercube_sampling,
                              output_path=None)

    return bo


def build_model(lower, upper, model_type="gp_mcmc", model_seed=1, prior_seed=1):
    """
    General interface for Bayesian optimization for global black box
    optimization problems.
    Parameters
    ----------
    lower: numpy.ndarray (D,)
        The lower bound of the search space
    upper: numpy.ndarray (D,)
        The upper bound of the search space
    model_type: {"gp", "gp_mcmc", "rf", "bohamiann", "dngo"}
        The model for the objective function.
    model_seed: int
        Seed for random number generator of the model
    prior_seed: int
        Seed for random number generator of the prior
    Returns
    -------
        Model
    """
    assert upper.shape[0] == lower.shape[0], "Dimension miss match"
    assert numpy.all(lower < upper), "Lower bound >= upper bound"

    cov_amp = 2
    n_dims = lower.shape[0]

    initial_ls = numpy.ones([n_dims])
    exp_kernel = george.kernels.Matern52Kernel(initial_ls,
                                               ndim=n_dims)
    kernel = cov_amp * exp_kernel

    prior = DefaultPrior(len(kernel) + 1, numpy.random.RandomState(prior_seed))

    n_hypers = 3 * len(kernel)
    if n_hypers % 2 == 1:
        n_hypers += 1

    # NOTE: Some models do not support RNG properly and rely on global RNG state
    #       so we need to seed here as well...
    numpy.random.seed(model_seed)
    model_rng = numpy.random.RandomState(model_seed)
    if model_type == "gp":
        model = GaussianProcess(kernel, prior=prior, rng=model_rng,
                                normalize_output=False, normalize_input=True,
                                lower=lower, upper=upper)
    elif model_type == "gp_mcmc":
        model = GaussianProcessMCMC(kernel, prior=prior,
                                    n_hypers=n_hypers,
                                    chain_length=200,
                                    burnin_steps=100,
                                    normalize_input=True,
                                    normalize_output=False,
                                    rng=model_rng, lower=lower, upper=upper)

    elif model_type == "rf":
        model = RandomForest(rng=model_rng)

    elif model_type == "bohamiann":
        model = WrapperBohamiann()

    elif model_type == "dngo":
        from pybnn.dngo import DNGO
        model = DNGO()

    else:
        raise ValueError("'{}' is not a valid model".format(model_type))

    return model


class RoBO(BaseAlgorithm):
    """TODO: Class docstring"""

    requires = 'real'

    def __init__(self, space: Space,
                 model_type='gp_mcmc',
                 maximizer="random",
                 acquisition_func="log_ei",
                 n_init=20,
                 model_seed=0,
                 prior_seed=0,
                 init_seed=0,
                 maximizer_seed=0,
                 **kwargs):

        super(RoBO, self).__init__(space, **kwargs)

        self.suggest_count = 0
        self.n_init = n_init
        self.model_type = model_type

        lower, upper = build_bounds(space)

        model = build_model(lower, upper, model_type, model_seed, prior_seed)
        self.model = model
        self.robo = build_optimizer(
            model, maximizer=maximizer, acquisition_func=acquisition_func,
            maximizer_seed=maximizer_seed)

        self.seed_rng(init_seed)

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        :param seed: Integer seed for the random number generator.

        .. note:: This methods does nothing if the algorithm is deterministic.
        """
        #
        # Look at method state dict in olympus version for rng values to seed
        # Not good to set all with same seed
        # sample five different
        #

        self.rng = numpy.random.RandomState(seed)

        size = 4
        rand_nums = numpy.random.randint(1, 10e8, size)

        self.robo.rng = numpy.random.RandomState(rand_nums[0])
        self.robo.maximize_func.rng.seed(rand_nums[1])
        self.robo.model.rng.seed(rand_nums[2])
        self.model.prior.rng.seed(rand_nums[3])

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        # TODO: Adapt this to your algo

        s_dict = super(RoBO, self).state_dict

        s_dict.update({'rng_state': self.rng.get_state(),
                       'global_numpy_rng_state': numpy.random.get_state(),
                       'maximizer_rng_state': self.robo.maximize_func.rng.get_state(),
                       'prior_rng_state': self.robo.model.prior.rng.get_state(),
                       'model_rng_state': self.robo.model.rng.get_state()})

        if self.model_type == 'gp_mcmc':
            if hasattr(self.robo.model, 'p0'):
                s_dict['model_p0'] = self.robo.model.p0.tolist()
        else:

            s_dict['model_kernel_parameter_vector'] = self.robo.model.kernel.get_parameter_vector().tolist()
            s_dict['noise'] = self.robo.model.noise

        return s_dict

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        # TODO: Adapt this to your algo

        super(RoBO, self).set_state(state_dict)

        self.rng.set_state(state_dict['rng_state'])
        numpy.random.set_state(state_dict['global_numpy_rng_state'])
        self.robo.maximize_func.rng.set_state(state_dict['maximizer_rng_state'])
        self.robo.model.rng.set_state(state_dict['model_rng_state'])
        self.robo.model.prior.rng.set_state(state_dict['prior_rng_state'])

    def suggest(self, num=1):
        """Suggest a `num`ber of new sets of parameters.

        TODO: document how suggest work for this algo

        Parameters
        ----------
        num: int, optional
            Number of points to suggest. Defaults to 1.

        Returns
        -------
        list of points or None
            A list of lists representing points suggested by the algorithm. The algorithm may opt
            out if it cannot make a good suggestion at the moment (it may be waiting for other
            trials to complete), in which case it will return None.

        Notes
        -----
        New parameters must be compliant with the problem's domain `orion.algo.space.Space`.

        """
        self.suggest_count += 1
        if self.suggest_count > self.n_init:

            pass
        else:
            return self.space.sample(num, seed=tuple(self.rng.randint(0, 1000000, size=3)))

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        TODO: document how observe work for this algo

        Parameters
        ----------
        points : list of tuples of array-likes
           Points from a `orion.algo.space.Space`.
           Evaluated problem parameters by a consumer.
        results : list of dicts
           Contains the result of an evaluation; partial information about the
           black-box function at each point in `params`.

        Result
        ------
        objective : numeric
           Evaluation of this problem's objective function.
        gradient : 1D array-like, optional
           Contains values of the derivatives of the `objective` function
           with respect to `params`.
        constraint : list of numeric, optional
           List of constraints expression evaluation which must be greater
           or equal to zero by the problem's definition.

        """

        # TODO: Adapt this to your algo
        pass
