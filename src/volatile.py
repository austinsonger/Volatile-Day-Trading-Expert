#!/usr/bin/env python
import os.path
import matplotlib.pyplot as plt
import csv
import datetime as dt
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS
import multitasking
import numpy as np
from download import download
from models import train_msis_mcs
from plotting import plot_industry_estimates, plot_market_estimates, plot_matches, plot_sector_estimates, plot_stock_estimates
from tools import convert_currency, extract_hierarchical_info
from typing import List, Dict, Tuple


def softplus(x: np.array) -> np.array:
    return np.log(1 + np.exp(x))

def estimate_logprice_statistics(mu: np.array, sigma: np.array, tt: np.array) -> Tuple[np.array, np.array]:
    return np.dot(mu, tt), softplus(sigma)

def estimate_price_statistics(mu: np.array, sigma: np.array) -> Tuple[np.array, np.array]:
    return np.exp(mu + sigma ** 2 / 2), np.sqrt(np.exp(2 * mu + sigma ** 2) * (np.exp(sigma ** 2) - 1))

def rate(scores: np.array, lower_bounds: Dict[str, int] = None) -> List[str]:
    if lower_bounds is None:
        lower_bounds = {"HIGHLY BELOW TREND": 3, "BELOW TREND": 2, "ALONG TREND": -2, "ABOVE TREND": -3}
    rates = []
    for score in scores:
        if score > lower_bounds["HIGHLY BELOW TREND"]:
            rates.append("HIGHLY BELOW TREND")
        elif score > lower_bounds["BELOW TREND"]:
            rates.append("BELOW TREND")
        elif score > lower_bounds["ALONG TREND"]:
            rates.append("ALONG TREND")
        elif score > lower_bounds["ABOVE TREND"]:
            rates.append("ABOVE TREND")
        else:
            rates.append("HIGHLY ABOVE TREND")
    return rates

from typing import List, Dict, Union
import numpy as np

def estimate_matches(tickers: List[str], mu: np.array, tt: np.array) -> Dict[str, Dict[str, Union[str, int, float]]]:
    """
    Estimates the matches for a given list of tickers based on the provided mu and tt arrays.

    Args:
        tickers (List[str]): A list of tickers.
        mu (np.array): A numpy array representing mu.
        tt (np.array): A numpy array representing tt.

    Returns:
        Dict[str, Dict[str, Union[str, int, float]]]: A dictionary containing the matches for each ticker.
            Each match is represented as a dictionary with the following keys:
                - "match": The ticker of the matched stock.
                - "index": The index of the matched stock.
                - "distance": The distance between the ticker and the matched stock.
    """
    dtt = np.arange(1, tt.shape[0])[:, None] * tt[1:] / tt[1, None]
    dlogp_est = np.dot(mu[:, 1:],  dtt)
    num_stocks = len(tickers)

    try:
        assert num_stocks <= 2000
        match_dist = np.sum((dlogp_est[:, None] - dlogp_est[None]) ** 2, 2)
        match_minidx = np.argsort(match_dist, 1)[:, 1]
        match_mindist = np.sort(match_dist, 1)[:, 1]
        matches = {tickers[i]: {"match": tickers[match_minidx[i]],
                              "index": match_minidx[i],
                              "distance": match_mindist[i]} for i in range(num_stocks)}
    except:
        num_threads = min([len(tickers), multitasking.cpu_count() * 2])
        multitasking.set_max_threads(num_threads)

        matches = {}

        @multitasking.task
        def _estimate_one(i, tickers, dlogp_est):
            match_dist = np.sum((dlogp_est[i] - dlogp_est) ** 2, 1)
            match_minidx = np.argsort(match_dist)[1]
            match_mindist = np.sort(match_dist)[1]
            matches[tickers[i]] = {"match": tickers[match_minidx], "index": match_minidx, "distance": match_mindist}

        for i in range(num_stocks):
            _estimate_one(i, tickers, dlogp_est)

    return matches

def estimate_clusters(tickers: List[str], mu: np.array, tt: np.array) -> List[int]:
    dtt = np.arange(1, tt.shape[0])[:, None] * tt[1:] / tt[1, None]
    dlogp_est = np.dot(mu[:, 1:],  dtt)
    num_stocks = len(tickers)

    num_threads = min([len(tickers), multitasking.cpu_count() * 2])
    multitasking.set_max_threads(num_threads)

    clusters = []

    def _unite_clusters(clusters):
        k = 0
        flag = 0
        while k < len(clusters):
            for j in range(k + 1, len(clusters)):
                if clusters[j] & clusters[k]:
                    clusters[j] = clusters[j].union(clusters[k])
                    flag = 1
                    break
            if flag:
                del clusters[k]
                flag = 0
            else:
                k += 1
        return clusters

    def _estimate_one(i, dlogp_est):
        dist = np.sum((dlogp_est[i] - dlogp_est) ** 2, 1)
        clusters.append(set(np.argsort(dist)[:2].tolist()))
        return _unite_clusters(clusters)

    for i in range(num_stocks):
        clusters = _estimate_one(i, dlogp_est)

    return [np.where([j in clusters[k] for k in range(len(clusters))])[0][0] for j in range(num_stocks)]

if __name__ == '__main__':
    cli = ArgumentParser('Volatile: your day-to-day trading companion.',
                         formatter_class=ArgumentDefaultsHelpFormatter)
    cli.add_argument('-s', '--symbols', type=str, nargs='+', help=SUPPRESS)
    cli.add_argument('--rank', type=str, default="rate",
                     help="If `rate`, stocks are ranked in the prediction table and in the stock estimation plot from "
                          "the highest below to the highest above trend; if `growth`, ranking is done from the largest"
                          " to the smallest trend growth at current date; if `volatility`, from the largest to the "
                          "smallest current volatility estimate.")
    cli.add_argument('--save-table', action='store_true',
                     help='Save prediction table in csv format.')
    cli.add_argument('--no-plots', action='store_true',
                     help='Plot estimates with their uncertainty over time.')
    cli.add_argument('--plot-losses', action='store_true',
                     help='Plot loss function decay over training iterations.')
    cli.add_argument('--cache', action='store_true',
                     help='Use cached data and parameters if available.')
    args = cli.parse_args()

    if args.rank.lower() not in ["rate", "growth", "volatility"]:
        raise Exception("{} not recognized. Please provide one between `rate` and `growth`.".format(args.rank))

    if args.cache and os.path.exists('data.pickle'):
        print('\nLoading last year of data...')
        with open('data.pickle', 'rb') as handle:
            data = pickle.load(handle)
        print('Data has been saved to {}/{}.'.format(os.getcwd(), 'data.pickle'))
    else:
        if args.symbols is None:
            with open("symbols_list.txt", "r") as my_file:
                args.symbols = my_file.readlines()[0].split(" ")
        print('\nDownloading last year of data...')
        data = download(args.symbols)

        with open('data.pickle', 'wb') as handle:
            pickle.dump(data, handle)

    tickers = data["tickers"]
    logp = np.log(data['price'])

    for i, curr in enumerate(data['currencies']):
        if curr != data['default_currency']:
            logp[i] = convert_currency(logp[i], np.array(data['exchange_rates'][curr]), type='forward')

    num_stocks, t = logp.shape

    info = extract_hierarchical_info(data['sectors'], data['industries'])

    if num_stocks > 1:
        print("\nTraining a model that discovers correlations...")
        order = 52
        info['tt'] = (np.linspace(1 / t, 1, t) ** np.arange(order + 1).reshape(-1, 1)).astype('float32')
        info['order_scale'] = np.ones((1, order + 1), dtype='float32')
        phi_m, psi_m, phi_s, psi_s, phi_i, psi_i, phi, psi = train_msis_mcs(logp, info, num_steps=50000)
        print("Training completed.")

        print("\nEstimate top matches...")
        matches = estimate_matches(tickers, phi.numpy(), info['tt'])
        print("Top matches estimation completed.")

    print("\nTraining a model that estimates and predicts trends...")
    horizon = 5
    order = 2
    info['tt'] = (np.linspace(1 / t, 1, t) ** np.arange(order + 1).reshape(-1, 1)).astype('float32')
    info['order_scale'] = np.linspace(1 / (order + 1), 1, order + 1)[::-1].astype('float32')[None, :]
    phi_m, psi_m, phi_s, psi_s, phi_i, psi_i, phi, psi = train_msis_mcs(logp, info, plot_losses=args.plot_losses)
    print("Training completed.")

    logp_est, std_logp_est = estimate_logprice_statistics(phi.numpy(), psi.numpy(), info['tt'])
    tt_pred = ((1 + (np.arange(1 + horizon) / t)) ** np.arange(order + 1).reshape(-1, 1)).astype('float32')
    logp_pred, std_logp_pred = estimate_logprice_statistics(phi.numpy(), psi.numpy(), tt_pred)
    logp_ind_est, std_logp_ind_est = estimate_logprice_statistics(phi_i.numpy(), psi_i.numpy(), info['tt'])
    logp_sec_est, std_logp_sec_est = estimate_logprice_statistics(phi_s.numpy(), psi_s.numpy(), info['tt'])
    logp_mkt_est, std_logp_mkt_est = estimate_logprice_statistics(phi_m.numpy(), psi_m.numpy(), info['tt'])

    scores = (logp_pred[:, horizon] - logp[:, -1]) / std_logp_pred.squeeze()
    growth = np.dot(phi.numpy()[:, 1:], np.arange(1, order + 1)) / t

    for i, curr in enumerate(data['currencies']):
        if curr != data['default_currency']:
            logp[i] = convert_currency(logp[i], np.array(data['exchange_rates'][curr]), type='backward')
            logp_est[i] = convert_currency(logp_est[i], np.array(data['exchange_rates'][curr]), type='backward')

    p_est, std_p_est = estimate_price_statistics(logp_est, std_logp_est)
    p_pred, std_p_pred = estimate_price_statistics(logp_pred, std_logp_pred)
    p_ind_est, std_p_ind_est = estimate_price_statistics(logp_ind_est, std_logp_ind_est)
    p_sec_est, std_p_sec_est = estimate_price_statistics(logp_sec_est, std_logp_sec_est)
    p_mkt_est, std_p_mkt_est = estimate_price_statistics(logp_mkt_est, std_logp_mkt_est)

    volatility = std_p_est[:, -1] / data['price'][:, -1]

    if args.rank == "rate":
        rank = np.argsort(scores)[::-1]
    elif args.rank == "growth":
        rank = np.argsort(growth)[::-1]
    elif args.rank == "volatility":
        rank = np.argsort(volatility)[::-1]

    ranked_tickers = np.array(tickers)[rank]
    ranked_scores = scores[rank]
    ranked_p = data['price'][rank]
    ranked_currencies = np.array(data['currencies'])[rank]
    ranked_growth = growth[rank]
    ranked_volatility = volatility[rank]
    if num_stocks > 1:
        ranked_matches = np.array([matches[ticker]["match"] for ticker in ranked_tickers])

    ranked_rates = rate(ranked_scores)

    if not args.no_plots:
        plot_market_estimates(data, p_mkt_est, std_p_mkt_est)
        plot_sector_estimates(data, info, p_sec_est, std_p_sec_est)
        plot_industry_estimates(data, info, p_ind_est, std_p_ind_est)
        plot_stock_estimates(data, p_est, std_p_est, args.rank, rank, ranked_rates)
        if num_stocks > 1:
            plot_matches(data, matches)

    print("\nPREDICTION TABLE")
    ranked_sectors = [name if name[:2] != "NA" else "Not Available" for name in np.array(list(data["sectors"].values()))[rank]]
    ranked_industries = [name if name[:2] != "NA" else "Not Available" for name in np.array(list(data["industries"].values()))[rank]]

    strf = "{:<15} {:<26} {:<42} {:<16} {:<22} {:<11} {:<15} {:<4}"
    num_dashes = 159
    separator = num_dashes * "-"
    print(num_dashes * "-")
    print(strf.format("SYMBOL", "SECTOR", "INDUSTRY", "PRICE", "RATE", "GROWTH", "VOLATILITY", "MATCH"))
    print(separator)
    for i in range(num_stocks):
        print(strf.format(ranked_tickers[i], ranked_sectors[i], ranked_industries[i],
                          f"{np.round(ranked_p[i, -1], 2)} {ranked_currencies[i]}", ranked_rates[i],
                          f"{'+' if ranked_growth[i] >= 0 else ''}{np.round(100 * ranked_growth[i], 2)}%",
                          np.round(ranked_volatility[i], 2),
                          ranked_matches[i] if num_stocks > 1 else "None"))
        print(separator)
        if i < num_stocks - 1 and ranked_rates[i] != ranked_rates[i + 1]:
            print(separator)

    if args.save_table:
        tab_name = 'prediction_table.csv'
        table = zip(["SYMBOL"] + ranked_tickers.tolist(),
                    ['SECTOR'] + ranked_sectors,
                    ['INDUSTRY'] + ranked_industries,
                    ["PRICE"] + [f"{np.round(ranked_p[i, -1], 2)} {ranked_currencies[i]}" for i in range(num_stocks)],
                    ["RATE"] + ranked_rates,
                    ["GROWTH"] + ranked_growth.tolist(),
                    ["VOLATILITY"] + ranked_volatility.tolist(),
                    ["MATCH"] + (ranked_matches.tolist() if num_stocks > 1 else ["None"]))

        with open(tab_name, 'w') as file:
            wr = csv.writer(file)
            for row in table:
                wr.writerow(row)
        print('\nThe prediction table printed above has been saved to {}/{}.'.format(os.getcwd(), tab_name))
