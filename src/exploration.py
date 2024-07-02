#!/usr/bin/env python
import numpy as np
import datetime as dt
import csv
import pickle
import logging
import argparse
from download import download
from volatile import estimate_logprice_statistics, estimate_price_statistics, rate
from tools import convert_currency, extract_hierarchical_info
from plotting import *
from models import *
import matplotlib.pyplot as plt
import os.path



logging.basicConfig(level=logging.INFO)

def load_data(cache):
    if cache and os.path.exists('data.pickle'):
        logging.info('Loading last year of data...')
        with open('data.pickle', 'rb') as handle:
            data = pickle.load(handle)
        logging.info('Data has been loaded from {}/{}.'.format(os.getcwd(), 'data.pickle'))
    else:
        if args.symbols is None:
            with open("symbols_list.txt", "r") as my_file:
                args.symbols = my_file.readlines()[0].split(" ")
        logging.info('Downloading last year of data...')
        try:
            data = download(args.symbols)
            with open('data.pickle', 'wb') as handle:
                pickle.dump(data, handle)
        except Exception as e:
            logging.error('Failed to download data: {}'.format(str(e)))
            data = None

    return data

def train_correlation_model(logp, info):
    logging.info("Training a model that discovers correlations...")
    order = 52
    info['tt'] = (np.linspace(1 / t, 1, t) ** np.arange(order + 1).reshape(-1, 1)).astype('float32')
    info['order_scale'] = np.ones((1, order + 1), dtype='float32')
    phi_m, psi_m, phi_s, psi_s, phi_i, psi_i, phi, psi = train_msis_mcs(logp, info, num_steps=50000)
    logging.info("Training completed.")
    return phi, psi

def train_trend_model(logp, info, plot_losses):
    logging.info("Training a model that estimates and predicts trends...")
    horizon = 5
    order = 2
    info['tt'] = (np.linspace(1 / t, 1, t) ** np.arange(order + 1).reshape(-1, 1)).astype('float32')
    info['order_scale'] = np.linspace(1 / (order + 1), 1, order + 1)[::-1].astype('float32')[None, :]
    phi_m, psi_m, phi_s, psi_s, phi_i, psi_i, phi, psi = train_msis_mcs(logp, info, plot_losses=args.plot_losses)
    logging.info("Training completed.")
    return phi, psi

def explore_stocks(data, p_est, std_p_est, num_rows, num_cols):
    num_stocks = len(data["tickers"])
    num_set = num_cols * num_rows
    prob = np.ones(num_stocks) / num_stocks
    idx_set = np.random.choice(num_stocks, num_set, p=prob, replace=False)
    idx_choice_all = set()
    j = 0
    stop_flag = False
    while True:
        j += 1
        plot_stocks_set_exploration(data, p_est, std_p_est, idx_set, num_rows=num_rows, num_cols=num_cols)
        choice_unknown = True
        while choice_unknown:
            choice = input("Round %d. Enter chosen stock(s), or NEXT, or RESTART, or STOP: " % j)
            if choice.upper() == "STOP":
                choice_unknown = False
                stop_flag = True
            elif choice.upper() == "NEXT":
                idx_set = np.random.choice(num_stocks, num_set, p=prob, replace=False)
                choice_unknown = False
            elif choice.upper() == "RESTART":
                idx_choice_all = set()
                prob = np.ones(num_stocks) / num_stocks
                idx_set = np.random.choice(num_stocks, num_set, p=prob, replace=False)
                choice_unknown = False
            else:
                choice = choice.replace(',', ' ').split()
                loc_choice = []
                tickers_set = np.array(data["tickers"])[idx_set]
                for c in choice:
                    where_c = np.where(tickers_set == c.upper())[0]
                    if len(where_c) == 0:
                        print("Choice {} not recognized.".format(c))
                    else:
                        loc_choice.append(where_c[0])
                if len(loc_choice) < len(choice):
                    print('Please choose stocks among the current choice set.')
                else:
                    idx_choice = idx_set[loc_choice]
                    idx_choice_all.update(idx_choice)
                    dist = np.mean([np.sum((dlogp_est[idx] - dlogp_est) ** 2, 1) for idx in idx_choice], 0)
                    prob = prob / (1 + dist)
                    prob /= prob.sum()
                    idx_set = np.random.choice(num_stocks, num_set, p=prob, replace=False)
                    choice_unknown = False
        plt.close()
        if stop_flag:
            break
    plot_chosen_stocks_exploration(data, p_est, std_p_est, idx_choice_all)

def main(args):
    data = load_data(args.cache)
    if data is None:
        return

    tickers = data["tickers"]
    logp = np.log(data['price'])

    for i, curr in enumerate(data['currencies']):
        if curr != data['default_currency']:
            logp[i] = convert_currency(logp[i], np.array(data['exchange_rates'][curr]), type='forward')

    num_stocks, t = logp.shape

    info = extract_hierarchical_info(data['sectors'], data['industries'])

    phi, psi = train_correlation_model(logp, info)

    tt = info['tt']
    dtt = np.arange(1, tt.shape[0])[:, None] * tt[1:] / tt[1, None]
    dlogp_est = np.dot(phi.numpy()[:, 1:], dtt)

    phi, psi = train_trend_model(logp, info, args.plot_losses)

    logp_est, std_logp_est = estimate_logprice_statistics(phi.numpy(), psi.numpy(), info['tt'])

    for i, curr in enumerate(data['currencies']):
        if curr != data['default_currency']:
            logp[i] = convert_currency(logp[i], np.array(data['exchange_rates'][curr]), type='backward')
            logp_est[i] = convert_currency(logp_est[i], np.array(data['exchange_rates'][curr]), type='backward')

    p_est, std_p_est = estimate_price_statistics(logp_est, std_logp_est)

    p = data["price"]
    currencies = data["currencies"]
    volume = data["volume"]
    lb, ub = compute_uncertainty_bounds(p_est, std_p_est)

    explore_stocks(data, p_est, std_p_est, num_rows=3, num_cols=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Volatile: your day-to-day trading companion.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--symbols', type=str, nargs='+', help=argparse.SUPPRESS)
    parser.add_argument('--rank', type=str, default="rate",
                        help="If `rate`, stocks are ranked in the prediction table and in the stock estimation plot from "
                             "the highest below to the highest above trend; if `growth`, ranking is done from the largest"
                             " to the smallest trend growth at current date; if `volatility`, from the largest to the "
                             "smallest current volatility estimate.")
    parser.add_argument('--save-table', action='store_true',
                        help='Save prediction table in csv format.')
    parser.add_argument('--no-plots', action='store_true',
                        help='Plot estimates with their uncertainty over time.')
    parser.add_argument('--plot-losses', action='store_true',
                        help='Plot loss function decay over training iterations.')
    parser.add_argument('--cache', action='store_true',
                        help='Use cached data and parameters if available.')
    args = parser.parse_args()

    main(args)
