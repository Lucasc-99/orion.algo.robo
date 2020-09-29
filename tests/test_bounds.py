"""Perform tests for function `orion.algo.robo.rbayes.build_bounds`."""
import numpy
import numpy.testing
import pytest

import orion.algo.base  # noqa
from orion.algo.robo.rbayes import build_bounds
from orion.algo.space import Categorical, Integer, Real, Space
from orion.core.worker.transformer import build_required_space


@pytest.fixture()
def space():
    """Return an optimization space"""
    space = Space()
    dim1 = Integer('yolo1', 'uniform', -3, 6)
    space.register(dim1)
    dim2 = Real('yolo2', 'uniform', 0, 1)
    space.register(dim2)

    return space


@pytest.fixture()
def multidim_space(space):
    """Return a search space with a multi-dimensional dimension"""
    # dim3 = Integer('multi1', 'uniform', [-3, 0], [6, 1], shape=2)
    # space.register(dim3)
    dim4 = Integer('multi2', 'uniform', 2, 2, shape=2)
    space.register(dim4)

    return space


@pytest.fixture()
def cat_space(space):
    """Return a search space with a categorical dimension"""
    dim3 = Categorical('cat2d', ['hello', 'kitty'])
    space.register(dim3)
    dim4 = Categorical('cat3d', ['hello', 'kitty', 'cat'])
    space.register(dim4)

    return space


def test_simple_space(space):
    """Test basic space with single real/int dimensions"""
    lower, upper = build_bounds(space)
    numpy.testing.assert_equal(lower, numpy.array([-3, 0]))
    numpy.testing.assert_equal(upper, numpy.array([3, 1]))


def test_multidim_space(multidim_space):
    """Test that multidim is flattened"""
    lower, upper = build_bounds(multidim_space)
    numpy.testing.assert_equal(lower, numpy.array([2, 2, -3, 0]))
    numpy.testing.assert_equal(upper, numpy.array([4, 4, 3, 1]))


def test_categorical(cat_space):
    """Test that categorical is mapped properly to vector embedding space"""
    lower, upper = build_bounds(build_required_space('real', cat_space))
    # First dimension is 2d category which is mapped to (0, 1) with < 0.5 threshold
    # Three next dimensions are the one-hot dimensions of 3d category
    numpy.testing.assert_equal(lower, numpy.array([0, 0, 0, 0, -3, 0]))
    numpy.testing.assert_equal(upper, numpy.array([1, 1, 1, 1, 3, 1]))
