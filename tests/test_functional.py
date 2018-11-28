import pytest
import numpy as np


from .. import functional


def test_softmax():
    scores = np.asarray([-3.44, 1.16, -0.81, 3.91]).reshape(1, 4)
    exp_probs = np.asarray([0.0006, 0.0596, 0.0083, 0.9315]).reshape(1, 4)
    assert np.allclose(functional.softmax(scores), exp_probs, rtol=1e-02)


def test_relu():
    arr_a = np.array([45, -12])
    assert np.array_equal(functional.relu(arr_a), np.array([45, 0]))
    arr_b = np.array(([-1, 0], [0.0001, -0.01]))
    assert np.allclose(functional.relu(arr_b), np.array(([0, 0], [0.0001, 0])))


def test_relu_derivative():
    arr_a = np.array([45, -12])
    assert np.array_equal(functional.relu_derivative(arr_a), np.array([1, 0]))
    arr_b = np.array(([-1, 0], [0.0001, -0.01]))
    assert np.allclose(functional.relu_derivative(
        arr_b), np.array(([0, 0], [1, 0])))


def test_sigmoid():
    arr_a = np.array([45, -12, 0])
    sig_a = np.array([1, 6.144e-06, 0.5])
    assert np.allclose(functional.sigmoid(arr_a), sig_a)
    arr_b = np.array(([-1, 0], [0.0001, -0.01]))
    sig_b = np.array(([0.2689, 0.5], [0.5, 0.5024]))
    assert np.allclose(functional.sigmoid(arr_b), sig_b, rtol=1e-02)


def test_sigmoid_derivative():
    arr_a = np.array([45, -12, 0])
    assert np.allclose(functional.sigmoid_derivative(
        arr_a), np.array([0, 6.144e-06, 0.25]))
    arr_b = np.array(([-1, 0], [0.0001, -0.01]))
    dsig_b = np.array(([0.1965, 0.25], [0.25, 0.2499]))
    assert np.allclose(functional.sigmoid_derivative(
        arr_b), dsig_b, rtol=1e-02)


def test_cross_entropy():
    y_hat_a = np.array([0.25, 0.25, 0.5]).reshape(1, 3)
    y_a = np.array([0, 0, 1]).reshape(1, 3)
    assert np.isclose(functional.cross_entropy(
        y_hat_a, y_a), 0.6931, rtol=1e-04)
    y_hat_b = np.array(([0.25, 0.25, 0.5], [0.1, 0.1, 0.9]))
    y_b = np.array(([0, 0, 1], [1, 0, 0]))
    assert np.isclose(functional.cross_entropy(
        y_hat_b, y_b), 1.4978, rtol=1e-04)


def test_cross_entropy_derivative():
    y_hat_a = np.array([0.25, 0.25, 0.5]).reshape(1, 3)
    y_a = np.array([0, 0, 1]).reshape(1, 3)
    grad_a = np.array([0.25, 0.25, -0.5]).reshape(1, 3)
    assert np.allclose(
        functional.cross_entropy_derivative(y_hat_a, y_a), grad_a)
    y_hat_b = np.array(([0.25, 0.25, 0.5], [0.1, 0.1, 0.9]))
    y_b = np.array(([0, 0, 1], [1, 0, 0]))
    grad_b = np.array(([0.25, 0.25, -0.5], [-0.9, 0.1, 0.9]))
    assert np.allclose(
        functional.cross_entropy_derivative(y_hat_b, y_b), grad_b)
