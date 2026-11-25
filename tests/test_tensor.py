"""The conversion of NumPy input to a tensor does not warn for an array that is not writable, which a broadcast or a frozen array is."""
import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.torch_cpu


def test_a_non_writable_array_becomes_a_tensor_with_no_warning_and_the_same_values():
    from wobbler._tensor import as_tensor

    frozen = np.broadcast_to(np.array([1.5, 2.5, 3.5]), (4, 3))
    assert not frozen.flags.writeable
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tensor = as_tensor(frozen, "cpu")
    assert tensor.shape == (4, 3) and tensor.dtype.is_floating_point
    np.testing.assert_array_equal(tensor.numpy(), frozen)


def test_the_tensor_of_a_non_writable_array_does_not_share_its_memory():
    from wobbler._tensor import as_tensor

    frozen = np.array([1.0, 2.0])
    frozen.flags.writeable = False
    tensor = as_tensor(frozen, "cpu")
    tensor[0] = 9.0
    assert frozen[0] == 1.0


def test_a_writable_float64_array_is_still_shared_without_a_copy():
    from wobbler._tensor import as_tensor

    array = np.array([1.0, 2.0])
    tensor = as_tensor(array, "cpu")
    tensor[0] = 9.0
    assert array[0] == 9.0
