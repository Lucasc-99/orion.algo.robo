#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform integration tests for `orion.algo.robo`."""
import os

import numpy
import pytest

from orion.algo.space import Integer, Real, Space
from orion.client import create_experiment
import orion.core.cli
from orion.core.worker.primary_algo import PrimaryAlgo
from orion.testing.state import OrionState


def rosenbrock_function(x, y):
    """Evaluate a n-D rosenbrock function."""
    z = x - 34.56789
    r = 4 * z**2 + 23.4
    return [dict(name='objective', type='objective', value=r)]


MODEL_TYPES = ['gp', 'gp_mcmc', 'bohamiann']


@pytest.fixture(scope='session')
def database():
    """Return Mongo database object to test with example entries."""
    from pymongo import MongoClient
    client = MongoClient(username='user', password='pass', authSource='orion_test')
    database = client.orion_test
    yield database
    client.close()


@pytest.fixture()
def clean_db(database):
    """Clean insert example experiment entries to collections."""
    database.experiments.drop()
    database.trials.drop()
    database.workers.drop()
    database.resources.drop()


@pytest.fixture()
def space():
    """Return an optimization space"""
    space = Space()
    dim1 = Integer('yolo1', 'uniform', -3, 6)
    space.register(dim1)
    dim2 = Real('yolo2', 'uniform', 0, 1)
    space.register(dim2)

    return space


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_seeding(space, model_type, mocker):
    """Verify that seeding makes sampling deterministic"""
    optimizer = PrimaryAlgo(space, {'robo': {'model_type': model_type}})

    optimizer.seed_rng(1)
    a = optimizer.suggest(1)[0]
    assert not numpy.allclose(a, optimizer.suggest(1)[0])

    optimizer.seed_rng(1)
    assert numpy.allclose(a, optimizer.suggest(1)[0])


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_seeding_bo(space, model_type, mocker):
    """Verify that seeding BO makes sampling deterministic"""
    n_init = 3
    optimizer = PrimaryAlgo(space, {'robo': {'model_type': model_type, 'n_init': n_init}})
    optimizer.seed_rng(1)

    spy = mocker.spy(optimizer.algorithm.robo, 'choose_next')

    samples = []
    for i in range(n_init + 2):
        a = optimizer.suggest(1)[0]
        optimizer.observe([a], [i / n_init])
        samples.append([a])

    assert spy.call_count == 2

    optimizer = PrimaryAlgo(space, {'robo': {'model_type': model_type, 'n_init': n_init}})
    optimizer.seed_rng(1)

    spy = mocker.spy(optimizer.algorithm.robo, 'choose_next')

    for i in range(n_init + 2):
        b = optimizer.suggest(1)[0]
        optimizer.observe([b], [i / n_init])
        samples[i].append(b)

    assert spy.call_count == 2

    for pair in samples:
        assert numpy.allclose(*pair)


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_set_state(space, model_type):
    """Verify that resetting state makes sampling deterministic"""
    optimizer = PrimaryAlgo(space, {'robo': {'model_type': model_type}})

    optimizer.seed_rng(1)
    state = optimizer.state_dict
    a = optimizer.suggest(1)[0]
    assert not numpy.allclose(a, optimizer.suggest(1)[0])

    optimizer.set_state(state)
    assert numpy.allclose(a, optimizer.suggest(1)[0])


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_set_state_bo(space, model_type, mocker):
    """Verify that resetting state during BO makes sampling deterministic"""
    n_init = 3
    optimizer = PrimaryAlgo(space, {'robo': {'model_type': model_type, 'n_init': n_init}})

    spy = mocker.spy(optimizer.algorithm.robo, 'choose_next')

    for i in range(n_init + 2):
        a = optimizer.suggest(1)[0]
        optimizer.observe([a], [i / n_init])

    assert spy.call_count == 2

    state = optimizer.state_dict
    a = optimizer.suggest(1)[0]
    assert not numpy.allclose(a, optimizer.suggest(1)[0])

    optimizer.set_state(state)
    assert numpy.allclose(a, optimizer.suggest(1)[0])


def test_optimizer(monkeypatch):
    """Check functionality of BayesianOptimizer wrapper for single shaped dimension."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(["hunt", "--name", "exp", "--max-trials", "5", "--config",
                             "./benchmark/robo.yaml",
                             "./benchmark/rosenbrock.py",
                             "-x~uniform(-5, 5)"])


def test_int(monkeypatch):
    """Check support of integer values."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(["hunt", "--name", "exp", "--max-trials", "5", "--config",
                             "./benchmark/robo.yaml",
                             "./benchmark/rosenbrock.py",
                             "-x~uniform(-5, 5, discrete=True)"])


def test_categorical():
    """Check support of categorical values."""
    with OrionState(experiments=[], trials=[]):

        exp = create_experiment(
            name="exp",
            space={
                'x': 'choices([-5, -2, 0, 2, 5])',
                'y': 'uniform(-50, 50, shape=2)'
            },
            algorithms={
                'robo': {
                    'model_type': 'gp',
                    'n_init': 2
                }
            },
            debug=True
        )

        for _ in range(10):
            trial = exp.suggest()
            assert trial.params['x'] in [-5, -2, 0, 2, 5]
            exp.observe(trial, [dict(name='objective', type='objective', value=0)])

    assert False


def test_optimizer_two_inputs(monkeypatch):
    """Check functionality of BayesianOptimizer wrapper for 2 dimensions."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(["hunt", "--name", "exp", "--max-trials", "5", "--config",
                             "./benchmark/robo.yaml",
                             "./benchmark/rosenbrock.py",
                             "-x~uniform(-5, 5)", "-y~uniform(-10, 10)"])


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_optimizer_actually_optimize(model_type):
    """Check if the optimizer has better optimization than random search."""
    best_random_search = 25.0

    with OrionState(experiments=[], trials=[]):

        exp = create_experiment(
            name="exp", space={'x': 'uniform(-50, 50, precision=6)'},
            max_trials=20,
            algorithms={
                'robo': {
                    'model_type': model_type,
                    'n_init': 5
                }
            },
            debug=True
        )

        exp.workon(rosenbrock_function, y=0)

        objective = exp.stats['best_evaluation']

        assert best_random_search > objective
