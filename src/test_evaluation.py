import numpy as np

def test_second_moment_loss():
    # Test case 1
    tt = np.array([1, 2, 3, 4, 5])
    mu = np.array([0, 0, 0, 0, 0])
    sigma = np.array([1, 1, 1, 1, 1])
    logp = np.array([0, 0, 0, 0, 0])
    expected_result = 0.0
    assert np.isclose(second_moment_loss(tt, mu, sigma, logp), expected_result)

    # Test case 2
    tt = np.array([1, 2, 3, 4, 5])
    mu = np.array([1, 2, 3, 4, 5])
    sigma = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    logp = np.array([1, 2, 3, 4, 5])
    expected_result = 0.0
    assert np.isclose(second_moment_loss(tt, mu, sigma, logp), expected_result)

    # Test case 3
    tt = np.array([1, 2, 3, 4, 5])
    mu = np.array([0, 0, 0, 0, 0])
    sigma = np.array([1, 1, 1, 1, 1])
    logp = np.array([1, 2, 3, 4, 5])
    expected_result = 5.0
    assert np.isclose(second_moment_loss(tt, mu, sigma, logp), expected_result)

    # Add more test cases here...

test_second_moment_loss()