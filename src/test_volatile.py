import numpy as np

def test_estimate_matches():
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    mu = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                   [0.2, 0.3, 0.4, 0.5, 0.6],
                   [0.3, 0.4, 0.5, 0.6, 0.7]])
    tt = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                   [0.2, 0.3, 0.4, 0.5, 0.6],
                   [0.3, 0.4, 0.5, 0.6, 0.7]])
    expected_result = {
        'AAPL': {'match': 'GOOGL', 'index': 1, 'distance': 0.02},
        'GOOGL': {'match': 'MSFT', 'index': 2, 'distance': 0.02},
        'MSFT': {'match': 'GOOGL', 'index': 1, 'distance': 0.02}
    }
    assert estimate_matches(tickers, mu, tt) == expected_result

    tickers = ['AAPL', 'GOOGL', 'MSFT']
    mu = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                   [0.2, 0.3, 0.4, 0.5, 0.6],
                   [0.3, 0.4, 0.5, 0.6, 0.7]])
    tt = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                   [0.2, 0.3, 0.4, 0.5, 0.6],
                   [0.3, 0.4, 0.5, 0.6, 0.7]])
    expected_result = {
        'AAPL': {'match': 'GOOGL', 'index': 1, 'distance': 0.02},
        'GOOGL': {'match': 'MSFT', 'index': 2, 'distance': 0.02},
        'MSFT': {'match': 'GOOGL', 'index': 1, 'distance': 0.02}
    }
    assert estimate_matches(tickers, mu, tt) == expected_result

    tickers = ['AAPL', 'GOOGL', 'MSFT']
    mu = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                   [0.2, 0.3, 0.4, 0.5, 0.6],
                   [0.3, 0.4, 0.5, 0.6, 0.7]])
    tt = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                   [0.2, 0.3, 0.4, 0.5, 0.6],
                   [0.3, 0.4, 0.5, 0.6, 0.7]])
    expected_result = {
        'AAPL': {'match': 'GOOGL', 'index': 1, 'distance': 0.02},
        'GOOGL': {'match': 'MSFT', 'index': 2, 'distance': 0.02},
        'MSFT': {'match': 'GOOGL', 'index': 1, 'distance': 0.02}
    }
    assert estimate_matches(tickers, mu, tt) == expected_result

test_estimate_matches()