import numpy as np

def test_download():
    # Test case 1
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    start = '2022-01-01'
    end = '2022-01-10'
    interval = '1d'
    expected_result = {
        'tickers': ['AAPL', 'GOOGL', 'MSFT'],
        'dates': pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06', '2022-01-07', '2022-01-08', '2022-01-09', '2022-01-10']),
        'price': np.array([[182.51, 182.51, 182.51, 182.51, 182.51, 182.51, 182.51, 182.51, 182.51, 182.51],
                           [3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0],
                           [300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0]]),
        'volume': np.array([[1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000],
                            [500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000],
                            [100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000]]),
        'currencies': ['USD', 'USD', 'USD'],
        'exchange_rates': {},
        'default_currency': 'USD',
        'sectors': {'AAPL': 'Technology', 'GOOGL': 'Technology', 'MSFT': 'Technology'},
        'industries': {'AAPL': 'Consumer Electronics', 'GOOGL': 'Internet Content & Information', 'MSFT': 'Software—Infrastructure'}
    }
    assert download(tickers, start, end, interval) == expected_result

    # Test case 2
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    start = '2022-01-01'
    end = '2022-01-10'
    interval = '1d'
    expected_result = {
        'tickers': ['AAPL', 'GOOGL', 'MSFT'],
        'dates': pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06', '2022-01-07', '2022-01-08', '2022-01-09', '2022-01-10']),
        'price': np.array([[182.51, 182.51, 182.51, 182.51, 182.51, 182.51, 182.51, 182.51, 182.51, 182.51],
                           [3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0],
                           [300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0]]),
        'volume': np.array([[1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000],
                            [500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000],
                            [100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000]]),
        'currencies': ['USD', 'USD', 'USD'],
        'exchange_rates': {},
        'default_currency': 'USD',
        'sectors': {'AAPL': 'Technology', 'GOOGL': 'Technology', 'MSFT': 'Technology'},
        'industries': {'AAPL': 'Consumer Electronics', 'GOOGL': 'Internet Content & Information', 'MSFT': 'Software—Infrastructure'}
    }
    assert download(tickers, start, end, interval) == expected_result

    # Test case 3
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    start = '2022-01-01'
    end = '2022-01-10'
    interval = '1d'
    expected_result = {
        'tickers': ['AAPL', 'GOOGL', 'MSFT'],
        'dates': pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06', '2022-01-07', '2022-01-08', '2022-01-09', '2022-01-10']),
        'price': np.array([[182.51, 182.51, 182.51, 182.51, 182.51, 182.51, 182.51, 182.51, 182.51, 182.51],
                           [3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0],
                           [300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0]]),
        'volume': np.array([[1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000],
                            [500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000],
                            [100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000]]),
        'currencies': ['USD', 'USD', 'USD'],
        'exchange_rates': {},
        'default_currency': 'USD',
        'sectors': {'AAPL': 'Technology', 'GOOGL': 'Technology', 'MSFT': 'Technology'},
        'industries': {'AAPL': 'Consumer Electronics', 'GOOGL': 'Internet Content & Information', 'MSFT': 'Software—Infrastructure'}
    }
    assert download(tickers, start, end, interval) == expected_result

test_download()