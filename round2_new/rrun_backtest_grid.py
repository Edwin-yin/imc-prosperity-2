import copy
from typing import Dict, List, Any
import pandas as pd
import json
from collections import defaultdict
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datamodel import TradingState, Listing, OrderDepth, Trade, Observation, Order, UserId
from round_2_ink import Trader
from backtester import Backtester


def calculate_SQUID_INK_fair(order_depth):
    # assumes order_depth has orders in it
    best_ask = min(order_depth.sell_orders.keys())
    best_bid = max(order_depth.buy_orders.keys())
    filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 15]
    filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15]
    mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
    mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid

    mmmid_price = (mm_ask + mm_bid) / 2
    return mmmid_price


def calculate_RAINFOREST_RESIN_fair(order_depth):
    return 10000


listings = {
    'RAINFOREST_RESIN': Listing(symbol='RAINFOREST_RESIN', product='RAINFOREST_RESIN', denomination='SEASHELLS'),
    'SQUID_INK': Listing(symbol='SQUID_INK', product='SQUID_INK', denomination='SEASHELLS'),
    'KELP': Listing(symbol='KELP', product='KELP', denomination='SEASHELLS'),
    'CROISSANTS': Listing(symbol='CROISSANTS', product='CROISSANTS', denomination='SEASHELLS'),
    'JAMS': Listing(symbol='JAMS', product='JAMS', denomination='SEASHELLS'),
    'DJEMBES': Listing(symbol='DJEMBES', product='DJEMBES', denomination='SEASHELLS'),
    'PICNIC_BASKET1': Listing(symbol='PICNIC_BASKET1', product='PICNIC_BASKET1', denomination='SEASHELLS'),
    'PICNIC_BASKET2': Listing(symbol='PICNIC_BASKET2', product='PICNIC_BASKET2', denomination='SEASHELLS'),
}


import itertools

def generate_param_combinations(param_grid):
    param_names = param_grid.keys()
    param_values = param_grid.values()
    combinations = list(itertools.product(*param_values))
    return [dict(zip(param_names, combination)) for combination in combinations]
import os
from tqdm import tqdm

def run_backtests(trader, listings, position_limit, fair_calcs, market_data, trade_history, backtest_dir, param_grid, symbol):
    if not os.path.exists(backtest_dir):
        os.makedirs(backtest_dir)

    param_combinations = generate_param_combinations(param_grid[symbol])

    results = []
    backtest_dict = {}
    for params in tqdm(param_combinations, desc=f"Running backtests for {symbol}", unit="backtest"):
        trader.params = {symbol: params}
        backtester = Backtester(trader, listings, position_limit, fair_calcs, market_data, trade_history)
        backtester.run()

        param_str = "-".join([f"{key}={value}" for key, value in params.items()])
        log_filename = f"{backtest_dir}/{symbol}_{param_str}.log"
        backtester._log_trades(log_filename)

        results.append((params, backtester.pnl[symbol]))
        backtest_dict[param_combinations] = copy.deepcopy(backtester)

    return results, backtest_dict

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"
    KELP = 'KELP'
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    SYNTHETIC1 = "SYNTHETIC1"
    SYNTHETIC2 = "SYNTHETIC2"
    SYNTHETIC12 = "SYNTHETIC12"
    SPREAD1 = "SPREAD1"
    SPREAD2 = "SPREAD2"
    SPREAD12 = "SPREAD12"
    BASKET1BY2 = "BASKET1BY2"


params_grid = {
    # Product.RAINFOREST_RESIN: {
    #     "fair_value": 10000,
    #     "take_width": 1,
    #     "clear_threshold": 30,
    #     "clear_width": 0,
    #     # for making
    #     "disregard_edge": 0,  # disregards orders for joining or pennying within this value from fair
    #     "join_edge": 1,  # joins orders within this edge
    #     "default_edge": 2,
    #     "soft_position_limit": 50,
    # },

    Product.SQUID_INK: {
        "take_width": [2],
        "clear_width": [0],
        "clear_threshold": [0],
        "prevent_adverse": [False],
        "adverse_volume": [15],
        "reversion_beta": [-0.2],
        "disregard_edge": [1],
        "join_edge": [2],
        "default_edge": [4],
        "soft_position_limit": [0],
        "zscore_threshold": [3.0,4.0,5.0,6.0],
        "zscore_threshold_for_clean": [0.5,1.0,2.0],
        "price_std_window": [50,100],
        "final_timestamp": [950000]
    },}

import io
def _process_data_(file):
    with open(file, 'r') as file:
        log_content = file.read()
    sections = log_content.split('Sandbox logs:')[1].split('Activities log:')
    sandbox_log = sections[0].strip()
    activities_log = sections[1].split('Trade History:')[0]
    # sandbox_log_list = [json.loads(line) for line in sandbox_log.split('\n')]
    trade_history = json.loads(sections[1].split('Trade History:')[1])
    # sandbox_log_df = pd.DataFrame(sandbox_log_list)
    market_data_df = pd.read_csv(io.StringIO(activities_log), sep=";", header=0)
    trade_history_df = pd.json_normalize(trade_history)
    # print(sections[1])
    return market_data_df, trade_history_df



position_limit = {
        'RAINFOREST_RESIN': 50,
        'SQUID_INK': 50,
        'KELP': 50,
        'CROISSANTS': 250,
        'JAMS': 350,
        'DJEMBES': 60,
        'PICNIC_BASKET1': 60,
        'PICNIC_BASKET2': 100
    }

fair_calculations = {
        "RAINFOREST_RESIN": calculate_RAINFOREST_RESIN_fair,
        "SQUID_INK": calculate_SQUID_INK_fair
    }
# run
all_backtesters = []
backtest_dir = "backtestruns"
for i in range(6):
    if i in [0,1,2]:
        day = i - 1
    #day = -1
        market_data = pd.read_csv(f"./round-2-island-data-bottle/prices_round_2_day_{day}.csv", sep=";", header=0)
        trade_history = pd.read_csv(f"./round-2-island-data-bottle/trades_round_2_day_{day}.csv", sep=";", header=0)

        # market_data, trade_history = _process_data_('./webruns/aggress_time.log')
        # market_data, trade_history = _process_data_('./webruns/null_strategy.log')
        trader = Trader()
        #backtester = Backtester(trader, listings, position_limit, fair_calculations, market_data, trade_history,"trade_history_sim.log", False, None)
        results, runned_backtester = run_backtests(trader, listings, position_limit, fair_calculations, market_data, trade_history, backtest_dir, params_grid, "SQUID_INK")
        print(runned_backtester.pnl['SQUID_INK'])
        all_backtesters.append(copy.deepcopy(runned_backtester))
    elif i in [3,4,5]:
        listings3 = { 'RAINFOREST_RESIN': Listing(symbol='RAINFOREST_RESIN', product='RAINFOREST_RESIN', denomination='SEASHELLS'),
        'SQUID_INK': Listing(symbol='SQUID_INK', product='SQUID_INK', denomination='SEASHELLS'),
         'KELP': Listing(symbol='KELP', product='KELP', denomination='SEASHELLS'),
         'CROISSANTS': Listing(symbol='CROISSANTS', product='CROISSANTS', denomination='SEASHELLS'),
        'JAMS': Listing(symbol='JAMS', product='JAMS', denomination='SEASHELLS'),
        'DJEMBES': Listing(symbol='DJEMBES', product='DJEMBES', denomination='SEASHELLS'),
         'PICNIC_BASKET1': Listing(symbol='PICNIC_BASKET1', product='PICNIC_BASKET1', denomination='SEASHELLS'),
        'PICNIC_BASKET2': Listing(symbol='PICNIC_BASKET2', product='PICNIC_BASKET2', denomination='SEASHELLS'),

        'VOLCANIC_ROCK': Listing(symbol='VOLCANIC_ROCK', product='VOLCANIC_ROCK', denomination='SEASHELLS'),
        'VOLCANIC_ROCK_VOUCHER_9500': Listing(symbol='VOLCANIC_ROCK_VOUCHER_9500', product='VOLCANIC_ROCK_VOUCHER_9500', denomination='SEASHELLS'),
        'VOLCANIC_ROCK_VOUCHER_9750': Listing(symbol='VOLCANIC_ROCK_VOUCHER_9750', product='VOLCANIC_ROCK_VOUCHER_9750', denomination='SEASHELLS'),
         'VOLCANIC_ROCK_VOUCHER_10000': Listing(symbol='VOLCANIC_ROCK_VOUCHER_10000', product='VOLCANIC_ROCK_VOUCHER_10000', denomination='SEASHELLS'),
         'VOLCANIC_ROCK_VOUCHER_10250': Listing(symbol='VOLCANIC_ROCK_VOUCHER_10250', product='VOLCANIC_ROCK_VOUCHER_10250', denomination='SEASHELLS'),
         'VOLCANIC_ROCK_VOUCHER_10500': Listing(symbol='VOLCANIC_ROCK_VOUCHER_10500', product='VOLCANIC_ROCK_VOUCHER_10500', denomination='SEASHELLS'),
         }

        position_limit3 = {
        'RAINFOREST_RESIN': 50,
        'SQUID_INK': 50,
        'KELP': 50,
        'CROISSANTS': 250,
        'JAMS': 350,
        'DJEMBES': 60,
        'PICNIC_BASKET1': 60,
        'PICNIC_BASKET2': 100,
        'VOLCANIC_ROCK_VOUCHER_9500': 200,
        'VOLCANIC_ROCK_VOUCHER_9750': 200,
         'VOLCANIC_ROCK_VOUCHER_10000': 200,
        }

        day = i - 3
        #day = -1
        market_data = pd.read_csv(f"./round-3-island-data-bottle/prices_round_3_day_{day}.csv", sep=";", header=0)
        trade_history = pd.read_csv(f"./round-3-island-data-bottle/trades_round_3_day_{day}.csv", sep=";", header=0)
        trader = Trader()
        results, runned_backtester = run_backtests(trader, listings3, position_limit3, fair_calculations, market_data, trade_history, backtest_dir, params_grid, "SQUID_INK")
        #print(params_grid)
        #print(runned_backtester.pnl['SQUID_INK'])
        all_backtesters.append(runned_backtester)
