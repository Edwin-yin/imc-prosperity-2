import copy
from typing import Dict, List, Any
import pandas as pd
import json
from collections import defaultdict
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datamodel import TradingState, Listing, OrderDepth, Trade, Observation, Order, UserId


class Backtester:
    def __init__(self, trader, listings: Dict[str, Listing], position_limit: Dict[str, int], fair_marks, 
                 market_data: pd.DataFrame, trade_history: pd.DataFrame, file_name: str = None,
                 do_verification: bool =False, output_fair: str = None):
        self.trader = trader
        self.listings = listings
        self.market_data = market_data
        self.position_limit = position_limit
        self.fair_marks = fair_marks
        self.trade_history = trade_history.sort_values(by=['timestamp', 'symbol'])
        self.file_name = file_name

        self.observations = [Observation({}, {}) for _ in range(len(market_data))]

        self.current_position = {product: 0 for product in self.listings.keys()}
        self.position_history = {product: [] for product in self.listings.keys()}
        self.pnl_history = {product: [] for product in self.listings.keys()}
        self.pnl = {product: 0 for product in self.listings.keys()}
        self.cash = {product: 0 for product in self.listings.keys()}
        self.trades = []
        self.sandbox_logs = []
        self.do_verification = do_verification
        self.output_fair = output_fair
        self.fair_price_history = {product: [] for product in self.listings.keys()}
        
    def run(self):

        traderData = ""
        
        timestamp_group_md = self.market_data.groupby('timestamp')
        timestamp_group_th = self.trade_history.groupby('timestamp')
        
        mode = 'empty'  # The empty version is used
        self_trade_history_dict = {}
        bot_trade_history_dict = {}
        for timestamp, group in timestamp_group_th:
            bot_trades = []
            self_trades = []
            for _, row in group.iterrows():
                symbol = row['symbol']
                price = row['price']
                quantity = row['quantity']
                buyer = row['buyer'] if pd.notnull(row['buyer']) else ""
                seller = row['seller'] if pd.notnull(row['seller']) else ""
                trade = Trade(symbol, int(price), int(quantity), buyer, seller, timestamp)
                if buyer != 'SUBMISSION' and seller != 'SUBMISSION':
                    bot_trades.append(trade)
                else:
                    mode = 'webruns'
                    self_trades.append(trade)
            self_trade_history_dict[timestamp] = self_trades
            bot_trade_history_dict[timestamp] = bot_trades

        if mode == 'webruns':
            if self.output_fair:
                self.fair_price_history = {product: [0]*len(timestamp_group_md) for product in self.listings.keys()}
            for timestamp, group in timestamp_group_md:
                products = group['product'].tolist()
                order_depths_pnl = self._construct_order_depths(group)
                self_trades = self_trade_history_dict.get(timestamp, [])
                sandboxLog = ""
                if self.do_verification or self.output_fair:
                    for product in products:
                        self._mark_pnl(self.cash, self.current_position, order_depths_pnl, self.pnl, product)
                        self_compute_pnl = self.pnl[product]
                        web_pnl = float(group['profit_and_loss'][group['product'] == product].values)
                        if self.do_verification:
                            assert self_compute_pnl == web_pnl
                        if self.output_fair:
                            if self.current_position[product] != 0:
                                web_fair_value = (web_pnl - self.cash[product]) / self.current_position[product]
                                self.fair_price_history[product][int(timestamp/100)] = web_fair_value
                            else:
                                self.fair_price_history[product][int(timestamp/100)] = np.nan
                for trade in self_trades:
                    trade_volume = trade.quantity
                    product = trade.symbol
                    price = trade.price
                    if trade.buyer == 'SUBMISSION':
                        if abs(trade_volume + self.current_position[product]) <= int(self.position_limit[product]):
                            self.current_position[product] += trade_volume
                            self.cash[product] -= price * trade_volume
                        else:
                            sandboxLog += f"\nOrders for product {product} exceeded limit of {self.position_limit[product]} set"
                    else:
                        if abs(trade_volume + self.current_position[product]) <= int(self.position_limit[product]):
                            self.current_position[product] -= trade_volume
                            self.cash[product] += price * trade_volume
                        else:
                            sandboxLog += f"\nOrders for product {product} exceeded limit of {self.position_limit[product]} set"

                for product in products:
                    if product in self.listings:
                        self._mark_pnl(self.cash, self.current_position, order_depths_pnl, self.pnl, product)
                        self.pnl_history[product].append(self.pnl[product])
                for product in self.listings.keys():
                    self.position_history[product].append(self.current_position[product])
            if self.output_fair:
                for product in self.listings.keys():
                    if self.pnl[product] != 0:
                        product_df = copy.deepcopy(self.market_data[self.market_data['product'] == product].reset_index(drop=True))
                        product_df.loc[:, 'fair_price'] = self.fair_price_history[product]
                        product_df.to_csv(f'{self.output_fair}_{product}.csv', index=False)

            # self._add_trades(self_trade_history_dict, bot_trade_history_dict)
                # self._add_trades(own_trades, market_trades)
        else:
            for timestamp, group in timestamp_group_md:
                own_trades = defaultdict(list)
                market_trades = defaultdict(list)
                pnl_product = defaultdict(float)

                order_depths = self._construct_order_depths(group)
                order_depths_matching = self._construct_order_depths(group)
                order_depths_pnl = self._construct_order_depths(group)

                state = self._construct_trading_state(traderData, timestamp, self.listings, order_depths,
                                     dict(own_trades), dict(market_trades), self.current_position, self.observations)

                orders, conversions, traderData = self.trader.run(state)

                products = group['product'].tolist()
                sandboxLog = ""
                trades_at_timestamp = bot_trade_history_dict.get(timestamp, [])

                for product in products:
                    new_trades = []
                    for order in orders.get(product, []):
                        executed_orders = self._execute_order(timestamp, order, order_depths_matching, self.current_position, self.cash, bot_trade_history_dict, sandboxLog)
                        if len(executed_orders) > 0:
                            trades_done, sandboxLog = executed_orders
                            new_trades.extend(trades_done)
                    if len(new_trades) > 0:
                        own_trades[product] = new_trades
                    
                self.sandbox_logs.append({"sandboxLog": sandboxLog, "lambdaLog": "", "timestamp": timestamp})
            
                trades_at_timestamp = bot_trade_history_dict.get(timestamp, [])
                if trades_at_timestamp:
                    for trade in trades_at_timestamp:
                        product = trade.symbol
                        market_trades[product].append(trade)
                else:
                    for product in products:
                        market_trades[product] = []

                for product in products:
                    if product in self.listings:
                        self._mark_pnl(self.cash, self.current_position, order_depths_pnl, self.pnl, product)
                        self.pnl_history[product].append(self.pnl[product])

                self._add_trades(own_trades, market_trades)
                for product in self.listings.keys():
                    self.position_history[product].append(self.current_position[product])
        return self._log_trades(self.file_name)
    
    
    def _log_trades(self, filename: str = None):
        if filename is None:
            return 
        # FIXME: profit and loss has not been calculated yet
        # self.market_data['profit_and_loss'] = self.pnl_history

        output = ""
        output += "Sandbox logs:\n"
        for i in self.sandbox_logs:
            output += json.dumps(i, indent=2) + "\n"

        output += "\n\n\n\nActivities log:\n"
        market_data_csv = self.market_data.to_csv(index=False, sep=";")
        market_data_csv = market_data_csv.replace("\r\n", "\n")
        output += market_data_csv

        output += "\n\n\n\nTrade History:\n"
        output += json.dumps(self.trades, indent=2)

        with open(filename, 'w') as file:
            file.write(output)

            
    def _add_trades(self, own_trades: Dict[str, List[Trade]], market_trades: Dict[str, List[Trade]]):
        products = set(own_trades.keys()) | set(market_trades.keys())
        for product in products:
            self.trades.extend([self._trade_to_dict(trade) for trade in own_trades.get(product, [])])
        for product in products:
            self.trades.extend([self._trade_to_dict(trade) for trade in market_trades.get(product, [])])

    def _trade_to_dict(self, trade: Trade) -> dict[str, Any]:
        return {
            "timestamp": trade.timestamp,
            "buyer": trade.buyer,
            "seller": trade.seller,
            "symbol": trade.symbol,
            "currency": "SEASHELLS",
            "price": trade.price,
            "quantity": trade.quantity,
        }
        
    def _construct_trading_state(self, traderData, timestamp, listings, order_depths, 
                                 own_trades, market_trades, position, observations):
        state = TradingState(traderData, timestamp, listings, order_depths, 
                             own_trades, market_trades, position, observations)
        return state
    
        
    def _construct_order_depths(self, group):
        order_depths = {}
        for idx, row in group.iterrows():
            product = row['product']
            order_depth = OrderDepth()
            for i in range(1, 4):
                if f'bid_price_{i}' in row and f'bid_volume_{i}' in row:
                    bid_price = row[f'bid_price_{i}']
                    bid_volume = row[f'bid_volume_{i}']
                    if not pd.isna(bid_price) and not pd.isna(bid_volume):
                        order_depth.buy_orders[int(bid_price)] = int(bid_volume)
                if f'ask_price_{i}' in row and f'ask_volume_{i}' in row:
                    ask_price = row[f'ask_price_{i}']
                    ask_volume = row[f'ask_volume_{i}']
                    if not pd.isna(ask_price) and not pd.isna(ask_volume):
                        order_depth.sell_orders[int(ask_price)] = -int(ask_volume)
            order_depths[product] = order_depth
        return order_depths
    
        
        
    def _execute_buy_order(self, timestamp, order, order_depths, position, cash, trade_history_dict, sandboxLog):
        trades = []
        order_depth = order_depths[order.symbol]

        for price, volume in list(order_depth.sell_orders.items()):
            if price > order.price or order.quantity == 0:
                break

            trade_volume = min(abs(order.quantity), abs(volume))
            if abs(trade_volume + position[order.symbol]) <= int(self.position_limit[order.symbol]):
                trades.append(Trade(order.symbol, price, trade_volume, "SUBMISSION", "", timestamp))
                position[order.symbol] += trade_volume
                self.cash[order.symbol] -= price * trade_volume
                order_depth.sell_orders[price] += trade_volume
                order.quantity -= trade_volume
            else:
                sandboxLog += f"\nOrders for product {order.symbol} exceeded limit of {self.position_limit[order.symbol]} set"
            

            if order_depth.sell_orders[price] == 0:
                del order_depth.sell_orders[price]
        
        trades_at_timestamp = trade_history_dict.get(timestamp, [])
        new_trades_at_timestamp = []
        for trade in trades_at_timestamp:
            if trade.symbol == order.symbol:
                if trade.price < order.price:
                    trade_volume = min(abs(order.quantity), abs(trade.quantity))
                    trades.append(Trade(order.symbol, order.price, trade_volume, "SUBMISSION", "", timestamp))
                    order.quantity -= trade_volume
                    position[order.symbol] += trade_volume
                    self.cash[order.symbol] -= order.price * trade_volume
                    if trade_volume == abs(trade.quantity):
                        continue
                    else:
                        new_quantity = trade.quantity - trade_volume
                        new_trades_at_timestamp.append(Trade(order.symbol, order.price, new_quantity, "", "", timestamp))
                        continue
            new_trades_at_timestamp.append(trade)  

        if len(new_trades_at_timestamp) > 0:
            trade_history_dict[timestamp] = new_trades_at_timestamp

        return trades, sandboxLog
        
        
        
    def _execute_sell_order(self, timestamp, order, order_depths, position, cash, trade_history_dict, sandboxLog):
        trades = []
        order_depth = order_depths[order.symbol]
        
        for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
            if price < order.price or order.quantity == 0:
                break

            trade_volume = min(abs(order.quantity), abs(volume))
            if abs(position[order.symbol] - trade_volume) <= int(self.position_limit[order.symbol]):
                trades.append(Trade(order.symbol, price, trade_volume, "", "SUBMISSION", timestamp))
                position[order.symbol] -= trade_volume
                self.cash[order.symbol] += price * abs(trade_volume)
                order_depth.buy_orders[price] -= abs(trade_volume)
                order.quantity += trade_volume
            else:
                sandboxLog += f"\nOrders for product {order.symbol} exceeded limit of {self.position_limit[order.symbol]} set"

            if order_depth.buy_orders[price] == 0:
                del order_depth.buy_orders[price]

        trades_at_timestamp = trade_history_dict.get(timestamp, [])
        new_trades_at_timestamp = []
        for trade in trades_at_timestamp:
            if trade.symbol == order.symbol:
                if trade.price > order.price:
                    trade_volume = min(abs(order.quantity), abs(trade.quantity))
                    trades.append(Trade(order.symbol, order.price, trade_volume, "", "SUBMISSION", timestamp))
                    order.quantity += trade_volume
                    position[order.symbol] -= trade_volume
                    self.cash[order.symbol] += order.price * trade_volume
                    if trade_volume == abs(trade.quantity):
                        continue
                    else:
                        new_quantity = trade.quantity - trade_volume
                        new_trades_at_timestamp.append(Trade(order.symbol, order.price, new_quantity, "", "", timestamp))
                        continue
            new_trades_at_timestamp.append(trade)  

        if len(new_trades_at_timestamp) > 0:
            trade_history_dict[timestamp] = new_trades_at_timestamp
                
        return trades, sandboxLog
        
        
        
    def _execute_order(self, timestamp, order, order_depths, position, cash, trades_at_timestamp, sandboxLog):
        if order.quantity == 0:
            return []
        order_depth = order_depths[order.symbol]
        if order.quantity > 0:
            return self._execute_buy_order(timestamp, order, order_depths, position, cash, trades_at_timestamp, sandboxLog)
        else:
            return self._execute_sell_order(timestamp, order, order_depths, position, cash, trades_at_timestamp, sandboxLog)
    
    def _mark_pnl(self, cash, position, order_depths, pnl, product):
        order_depth = order_depths[product]
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid = (best_ask + best_bid)/2
        fair = mid
        if product in self.fair_marks:
            get_fair = self.fair_marks[product]
            fair = get_fair(order_depth)
        pnl[product] = cash[product] + fair * position[product]


if __name__ == '__main__':
    from round_2_ink import Trader


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
    # for day in [0]:
    # #day = -1
    #     market_data = pd.read_csv(f"./round-2-island-data-bottle/prices_round_2_day_{day}.csv", sep=";", header=0)
    #     trade_history = pd.read_csv(f"./round-2-island-data-bottle/trades_round_2_day_{day}.csv", sep=";", header=0)
    #     import io
    #     def _process_data_(file):
    #         with open(file, 'r') as file:
    #             log_content = file.read()
    #         sections = log_content.split('Sandbox logs:')[1].split('Activities log:')
    #         sandbox_log = sections[0].strip()
    #         activities_log = sections[1].split('Trade History:')[0]
    #         # sandbox_log_list = [json.loads(line) for line in sandbox_log.split('\n')]
    #         trade_history = json.loads(sections[1].split('Trade History:')[1])
    #         # sandbox_log_df = pd.DataFrame(sandbox_log_list)
    #         market_data_df = pd.read_csv(io.StringIO(activities_log), sep=";", header=0)
    #         trade_history_df = pd.json_normalize(trade_history)
    #         # print(sections[1])
    #         return market_data_df, trade_history_df
    #     # market_data, trade_history = _process_data_('./webruns/aggress_time.log')
    #     # market_data, trade_history = _process_data_('./webruns/null_strategy.log')
    #     trader = Trader()
    #     backtester = Backtester(trader, listings, position_limit, fair_calculations, market_data, trade_history,
    #                             "trade_history_sim.log", False, None)
    #     backtester.run()
    #     print(backtester.pnl)
    #     df_ink = market_data[market_data['product'] == 'SQUID_INK']
    #     fig, ax1 = plt.subplots()
    #     # 绘制第一条折线（左侧Y轴）
    #     max_pnl = max(backtester.pnl_history['SQUID_INK'])
    #     color = 'tab:blue'
    #     ax1.set_xlabel('Time')
    #     ax1.set_ylabel('pnl', color=color)
    #     ax1.plot(df_ink['timestamp'] / 100, np.array(backtester.pnl_history['SQUID_INK'])*50/max_pnl, color=color, marker='o',linestyle='-')
    #     ax1.tick_params(axis='y', labelcolor=color)
    #
    #     # 创建第二个Y轴
    #     ax2 = ax1.twinx()
    #     color = 'tab:red'
    #     ax2.set_ylabel('pos', color=color)
    #     ax2.plot(df_ink['timestamp'] / 100, np.array(backtester.position_history['SQUID_INK']), color=color, linestyle='-', marker='s')
    #     ax2.tick_params(axis='y', labelcolor=color)
    #
    #     # 创建第二个Y轴
    #     ax3 = ax1.twinx()
    #     color = 'tab:green'
    #     ax3.set_ylabel('price', color=color)
    #     ax3.plot(df_ink['timestamp'] / 100, np.array(df_ink['mid_price'])*50/np.nanmax(df_ink['mid_price']), color=color, linestyle='-',
    #              marker='^')
    #     ax3.tick_params(axis='y', labelcolor=color)
    #
    # # 显示网格
    #     plt.grid(True)
    #
    #     # 显示图形
    #     plt.show()

    all_round3_backtesters = []
    listings3 = { 'RAINFOREST_RESIN': Listing(symbol='RAINFOREST_RESIN', product='RAINFOREST_RESIN', denomination='SEASHELLS'),
    'SQUID_INK': Listing(symbol='SQUID_INK', product='SQUID_INK', denomination='SEASHELLS'),
     'KELP': Listing(symbol='KELP', product='KELP', denomination='SEASHELLS'),
     'CROISSANTS': Listing(symbol='CROISSANTS', product='CROISSANTS', denomination='SEASHELLS'),
    'JAMS': Listing(symbol='JAMS', product='JAMS', denomination='SEASHELLS'),
    'DJEMBES': Listing(symbol='DJEMBES', product='DJEMBES', denomination='SEASHELLS'),
     'PICNIC_BASKET1': Listing(symbol='PICNIC_BASKET1', product='PICNIC_BASKET1', denomination='SEASHELLS'),
    'PICNIC_BASKET2': Listing(symbol='PICNIC_BASKET2', product='PICNIC_BASKET2', denomination='SEASHELLS'),
    #
    # 'VOLCANIC_ROCK': Listing(symbol='VOLCANIC_ROCK', product='VOLCANIC_ROCK', denomination='SEASHELLS'),
    # 'VOLCANIC_ROCK_VOUCHER_9500': Listing(symbol='VOLCANIC_ROCK_VOUCHER_9500', product='VOLCANIC_ROCK_VOUCHER_9500', denomination='SEASHELLS'),
    # 'VOLCANIC_ROCK_VOUCHER_9750': Listing(symbol='VOLCANIC_ROCK_VOUCHER_9750', product='VOLCANIC_ROCK_VOUCHER_9750', denomination='SEASHELLS'),
    #  'VOLCANIC_ROCK_VOUCHER_10000': Listing(symbol='VOLCANIC_ROCK_VOUCHER_10000', product='VOLCANIC_ROCK_VOUCHER_10000', denomination='SEASHELLS'),
    #  'VOLCANIC_ROCK_VOUCHER_10250': Listing(symbol='VOLCANIC_ROCK_VOUCHER_10250', product='VOLCANIC_ROCK_VOUCHER_10250', denomination='SEASHELLS'),
    #  'VOLCANIC_ROCK_VOUCHER_10500': Listing(symbol='VOLCANIC_ROCK_VOUCHER_10500', product='VOLCANIC_ROCK_VOUCHER_10500', denomination='SEASHELLS'),
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

    for day in [0,1,2]:
    #day = -1
        market_data = pd.read_csv(f"./round-3-island-data-bottle/prices_round_3_day_{day}.csv", sep=";", header=0)
        trade_history = pd.read_csv(f"./round-3-island-data-bottle/trades_round_3_day_{day}.csv", sep=";", header=0)
        df_ink = market_data[market_data['product'] == 'SQUID_INK']
        #print(np.where(np.isnan(df_ink['bid_volume_1'])))
        import io
        # market_data, trade_history = _process_data_('./webruns/aggress_time.log')
        # market_data, trade_history = _process_data_('./webruns/null_strategy.log')
        trader = Trader()

        backtester = Backtester(trader, listings3, position_limit3, {}, market_data, trade_history, "trade_history_sim.log", False, None)
        backtester.run()
        print(backtester.pnl)
        all_round3_backtesters.append(copy.deepcopy(backtester))