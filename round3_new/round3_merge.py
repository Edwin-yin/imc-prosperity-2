from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
from statistics import NormalDist


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
    VOLCANIC_ROCK = 'VOLCANIC_ROCK'
    VOLCANIC_ROCK_VOUCHER_9500 = 'VOLCANIC_ROCK_VOUCHER_9500'
    VOLCANIC_ROCK_VOUCHER_9750 = 'VOLCANIC_ROCK_VOUCHER_9750'
    VOLCANIC_ROCK_VOUCHER_10000 = 'VOLCANIC_ROCK_VOUCHER_10000'
    VOLCANIC_ROCK_VOUCHER_10250 = 'VOLCANIC_ROCK_VOUCHER_10250'
    VOLCANIC_ROCK_VOUCHER_10500 = 'VOLCANIC_ROCK_VOUCHER_10500'
 

SYNTHETIC = {
    Product.PICNIC_BASKET1: Product.SYNTHETIC1,
    Product.PICNIC_BASKET2: Product.SYNTHETIC2,
    Product.BASKET1BY2: Product.SYNTHETIC12,
}

SPREAD = {
    Product.PICNIC_BASKET1: Product.SPREAD1,
    Product.PICNIC_BASKET2: Product.SPREAD2,
    Product.BASKET1BY2: Product.SPREAD12,
}

STRIKES = np.array([10500, 10250, 10000, 9750, 9500], dtype=np.int32)

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_threshold": 30,
        "clear_width": 0,
        # for making
        "disregard_edge": 0,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 1,  # joins orders within this edge
        "default_edge": 2,
        "soft_position_limit": 50,
    },
    
    Product.SQUID_INK: {
        "take_width": 2,
        "clear_width": 0,
        "clear_threshold": 0,
        "prevent_adverse": False,
        "adverse_volume": 15,
        "reversion_beta": 0.0303,
        "imb_beta": 2.843e-4,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 0,
        "zscore_threshold": 5.0,
        "zscore_threshold_for_clean": 1.0,
        "price_std_window": 50,
        "final_timestamp": 950000
    },
    
    Product.KELP: {
        "take_width": 10000,
        "clear_width": 0,
        "clear_threshold": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.257, # -0.293 for weighted
        "imb_beta": 4.47e-5,
        "disregard_edge": 1,
        "join_edge": 1,
        "default_edge": 2,
        "soft_position_limit": 50,
    },
    
    Product.SPREAD1: {
        "default_spread_mean": 0, # 57 for round 3, 49 for round 2
        # "default_spread_std": 40,
        "spread_std_window": 45,
        "zscore_threshold": 3.5,
        "target_position": 41, # max 41
        "max_taking_levels": 3,
    },
    
    Product.SPREAD12: {
        "default_spread_mean": 0, # 23 for round 3, 3 for round 2,
        # "default_spread_std": 40,
        "spread_std_window": 45,
        "zscore_threshold": 4.5,
        "target_position": 33, # max 30 for single pair; max 33 for two pairs
        "max_taking_levels": 3,
    },
    
    Product.VOLCANIC_ROCK:{
        "tte": 8 / 250,  # in unit of years
        "default_base_iv": 0.1322,
        "default_base_iv_std": 0.009201,
        "window": 100,
        "base_iv_decay": -1.089e-3, # in unit of day
        "default_parabolic_coef": 0.0642,
        "upper_threshold": 1, # zscore
        "lower_threshold": 1,   # zscore
        "option_base_iv_beta": -0.5,
        "option_parabolic_coef_beta": -0.491,
        
    },
    
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "default_std_iv": 0.0694,
        "strike": 9500,
        "zscore_threshold": 1
    },
    
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "default_std_iv": 0.0339,
        "strike": 9750,
        "zscore_threshold": 1
    },
    
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "default_std_iv": 0.00867,
        "strike": 10000,
        "zscore_threshold": 1
    },
    
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "default_std_iv": 0.00282,
        "strike": 10250,
        "zscore_threshold": 1
    },
    
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "default_std_iv": 0.00560,
        "strike": 10500,
        "zscore_threshold": 1
    },
    
    "day": 4
}

SYNTHETIC1_WEIGHTS = {
    Product.PICNIC_BASKET1: 1,
}

SYNTHETIC2_WEIGHTS = {
    Product.PICNIC_BASKET2: 1,
}

SYNTHETIC12_WEIGHTS = {
    Product.PICNIC_BASKET1: 2,
}

PICNIC_BASKET1_WEIGHTS = {
    Product.CROISSANTS: 6,
    Product.JAMS: 3,
    Product.DJEMBES: 1,
}

PICNIC_BASKET2_WEIGHTS = {
    Product.CROISSANTS: 4,
    Product.JAMS: 2,
}

BASKET1BY2_WEIGHTS = {
    Product.PICNIC_BASKET2: 3,
    Product.DJEMBES: 2,
}

class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_maturity, volatility):
        if volatility == 0:
            return spot - strike
        d1 = (
            math.log(spot) - math.log(strike) + (0.5 * volatility * volatility) * time_to_maturity
        ) / (volatility * math.sqrt(time_to_maturity))
        d2 = d1 - volatility * math.sqrt(time_to_maturity)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_maturity, volatility):
        if volatility == 0:
            return strike - spot
        d1 = (math.log(spot / strike) + (0.5 * volatility * volatility) * time_to_maturity) / (
            volatility * math.sqrt(time_to_maturity)
        )
        d2 = d1 - volatility * math.sqrt(time_to_maturity)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_maturity, volatility):
        if volatility == 0:
            return 1
        d1 = (
            math.log(spot) - math.log(strike) + (0.5 * volatility * volatility) * time_to_maturity
        ) / (volatility * math.sqrt(time_to_maturity))
        return NormalDist().cdf(d1)

    @staticmethod
    def gamma(spot, strike, time_to_maturity, volatility):
        if volatility == 0:
            return 0
        d1 = (
            math.log(spot) - math.log(strike) + (0.5 * volatility * volatility) * time_to_maturity
        ) / (volatility * math.sqrt(time_to_maturity))
        return NormalDist().pdf(d1) / (spot * volatility * math.sqrt(time_to_maturity))

    @staticmethod
    def vega(spot, strike, time_to_maturity, volatility):
        if volatility == 0:
            return 0
        d1 = (
            math.log(spot) - math.log(strike) + (0.5 * volatility * volatility) * time_to_maturity
        ) / (volatility * math.sqrt(time_to_maturity))
        # print(f"d1: {d1}")
        # print(f"vol: {volatility}")
        # print(f"spot: {spot}")
        # print(f"strike: {strike}")
        # print(f"time: {time_to_maturity}")
        return NormalDist().pdf(d1) * (spot * math.sqrt(time_to_maturity)) / 100

    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_maturity, max_iterations=200, tolerance=1e-10
    ):
        if call_price < spot - strike:
            return 0
        
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_maturity, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                flag = True
                return volatility
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return np.nan


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.SQUID_INK: 50,
            Product.KELP: 50,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
            Product.VOLCANIC_ROCK: 400, 
        }
                
    def take_best_orders_ink(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
        timestamp: int = 0,
        traderObject: Dict[str, Any] = None,
    ) -> tuple[int, int]:
        # TODO: we need add timestamp as we should do nothing in the end
        #if timestamp == 41300:
            #print('here')
        position_limit = self.LIMIT[product]
        # best_ask = min(order_depth.sell_orders.keys()) if len(order_depth.sell_orders) != 0 else np.nan
        # best_bid = max(order_depth.buy_orders.keys()) if len(order_depth.buy_orders) != 0 else np.nan
        # mid_price = (best_ask + best_bid) / 2
        # if not np.isnan(mid_price):
        #     traderObject[product]["price_history"].append(mid_price)
        #     traderObject[product]["prev_price"] = mid_price
        # else:
        #     mid_price = traderObject[product]["prev_price"]
        # if (
        #     len(traderObject[product]["price_history"])
        #     < self.params[product]["price_std_window"]
        # ):
        #     return None, None
        # elif len(traderObject[product]["price_history"]) > self.params[product]["price_std_window"]:
        #     traderObject[product]["price_history"].pop(0)
        
        ink_fair_price = self.squidink_fair_value(order_depth, traderObject)
        if(
            len(traderObject[product]["price_history"])
            < self.params[product]['price_std_window'] or
            ink_fair_price is None
        ):
            return None, None

        # Process sell orders (best ask)
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            # Check adverse condition for sell orders
            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                quantity = 0
                best_ask_zscore = (best_ask - np.mean(traderObject[product]["price_history"])
                 ) / np.std(traderObject[product]["price_history"])
                if best_ask_zscore <= -self.params[product]['zscore_threshold'] \
                        and timestamp < self.params[product]['final_timestamp']:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                elif (abs(best_ask_zscore) < self.params[product]['zscore_threshold_for_clean']
                         and position < -self.params[product]['soft_position_limit']):
                    quantity = min(
                        best_ask_amount, position_limit - position, -position
                    )  # max amt to buy
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        # Process buy orders (best bid)
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            best_bid_zscore = (best_bid - np.mean(traderObject[product]["price_history"])
                 ) / np.std(traderObject[product]["price_history"])
            # Check adverse condition for buy orders
            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                quantity = 0
                if best_bid_zscore >= self.params[product]['zscore_threshold'] \
                        and timestamp < self.params[product]['final_timestamp']:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                elif (abs(best_bid_zscore) < self.params[product]['zscore_threshold_for_clean']
                         and position > self.params[product]['soft_position_limit']):
                    quantity = min(
                        best_bid_amount, position_limit + position, position
                    )  # should be the max we can sell
                if quantity > 0:
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def take_orders_ink(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
        timestamp: int = 0,
        traderObject: Dict[str, Any] = None,
    ) -> tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders_ink(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
            timestamp,
            traderObject
        )
        return orders, buy_order_volume, sell_order_volume

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
        position_limit: int | None = None,
    ) -> tuple[int, int]:
        
        position_limit = self.LIMIT[product] if position_limit is None else position_limit

        # Process sell orders (best ask)
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            # Check adverse condition for sell orders
            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width or (best_ask <= fair_value and position < 0): # Aggressively taking when having position
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        # Process buy orders (best bid)
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            # Check adverse condition for buy orders
            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width or (best_bid >= fair_value and position > 0): # Aggressively taking when having position
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> tuple[int, int]:
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        clear_threshold: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > clear_threshold:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        elif position_after_take < -clear_threshold:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def squidink_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask is None or mm_bid is None:
                if len(traderObject[Product.SQUID_INK]["price_history"]) > 0:
                    mmmid_price = traderObject[Product.SQUID_INK]["price_history"][-1]
                else:
                    mmmid_price = (best_ask + best_bid) / 2
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
            
            imb = 0
            if mm_ask is None:
                imb = 1 if mm_bid is None else 0
            elif mm_bid is None:
                imb = -1
            else:
                imb = (order_depth.buy_orders[mm_bid] + order_depth.sell_orders[mm_ask]) /  (order_depth.buy_orders[mm_bid] - order_depth.sell_orders[mm_ask])

            if len(traderObject[Product.SQUID_INK]["price_history"]) > 0:
                last_price = traderObject[Product.SQUID_INK]["price_history"][-1]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.SQUID_INK]["reversion_beta"] + imb * self.params[Product.SQUID_INK]["imb_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject[Product.SQUID_INK]["price_history"].append(mmmid_price)
            if len(traderObject[Product.SQUID_INK]["price_history"]) > self.params[Product.SQUID_INK]["price_std_window"]:
                traderObject[Product.SQUID_INK]["price_history"].pop(0)
            return fair
        return None
    
    def kelp_weighted_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            weighted_bid = sum(price * volume for price, volume in order_depth.buy_orders.items()) / sum(order_depth.buy_orders.values())
            weighted_ask = sum(price * volume for price, volume in order_depth.sell_orders.items()) / sum(order_depth.sell_orders.values())
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            imbalance = (order_depth.buy_orders[best_bid] + order_depth.sell_orders[best_ask]) / (-order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid])
            mmmid_price = (weighted_bid + weighted_ask) / 2

            if traderObject.get("kelp_last_price", None) != None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"] + imbalance * self.params[Product.KELP]["imb_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None
    
    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None

            if mm_ask is None or mm_bid is None:
                if traderObject.get("kelp_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2    
                else:
                    mmmid_price = traderObject["kelp_last_price"]                         
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            imb = 0
            if mm_ask is None:
                imb = 1 if mm_bid is None else 0
            elif mm_bid is None:
                imb = -1
            else:
                imb = (order_depth.buy_orders[mm_bid] + order_depth.sell_orders[mm_ask]) /  (order_depth.buy_orders[mm_bid] - order_depth.sell_orders[mm_ask])

            if traderObject.get("kelp_last_price", None) != None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"] + imb * self.params[Product.KELP]["imb_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        clear_threshold: int = 0,
    ) -> tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            clear_threshold,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ) -> tuple[List[Order], int, int]:
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
                bid -= 1
            elif position < -1 * soft_position_limit:
                bid += 1
                ask += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    def get_swmid(self, order_depth) -> tuple[float, float, float]:
        best_bid = max(order_depth.buy_orders.keys()) if len(order_depth.buy_orders) != 0 else np.nan
        best_ask = min(order_depth.sell_orders.keys()) if len(order_depth.sell_orders) != 0 else np.nan
        best_bid_vol = abs(order_depth.buy_orders[best_bid]) if best_bid != np.nan else 0
        best_ask_vol = abs(order_depth.sell_orders[best_ask]) if best_ask != np.nan else 0
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol) if best_bid_vol != 0 and best_ask_vol != 0 else np.nan, best_bid, best_ask

    # def get_basket_position(self, product: str, position: Dict[str, int]) -> tuple[int, int]:
    #     # return the max and min effective basket position
    #     weights = eval(f'{SYNTHETIC[product]}_WEIGHTS')
    #     max_basket_position = 0
    #     min_basket_position = 0
    #     for basket, w in weights.items():
    #         if w == 0 or basket not in position:
    #             continue
    #         if math.ceil(position[basket] / w) > max_basket_position:
    #             max_basket_position = math.ceil(position[basket] / w)
    #         if math.floor(position[basket] / w) < min_basket_position:
    #             min_basket_position = math.floor(position[basket] / w)
                
    #     # Opposite positions for synthetic products
    #     weights = eval(f'{product}_WEIGHTS')
    #     max_opposite_position = 0
    #     min_oppositie_position = 0
    #     for basket, w in weights.items():
    #         if w == 0 or basket not in position:
    #             continue
    #         if math.ceil(position[basket] / w) > max_opposite_position:
    #             max_opposite_position = math.ceil(position[basket] / w)
    #         if math.floor(position[basket] / w) < min_oppositie_position:
    #             min_oppositie_position = math.floor(position[basket] / w)
    #     return max(max_basket_position, -max_opposite_position), min(min_basket_position, -min_oppositie_position)

    def get_synthetic_basket_order_depth(
        self, order_depths: Dict[str, OrderDepth], product: str, max_levels: int
    ) -> tuple[OrderDepth, dict, dict]:
        # Constants
        item_per_basket = eval(f'{product}_WEIGHTS')
        # Initialize the synthetic basket order depth
        synthetic_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        best_bids = {}
        best_asks = {}
        for product, weight in item_per_basket.items():
            bids = sorted(order_depths[product].buy_orders.items(), key=lambda x: x[0], reverse=True)
            volume = 0
            price = 0
            take_price = 0
            for i, bid in enumerate(bids[:max_levels]):
                volume += bid[1]
                price += bid[0] * bid[1]
                if volume >= weight:
                    if i > 0:
                        price -= bid[0] * (volume - weight)
                        volume = weight
                    else: 
                        price -= bid[0] * (volume % weight)
                        volume -= volume % weight
                    take_price = bid[0]
                    break
            best_bids[product] = (take_price, price / volume if volume >= weight else np.nan, volume // weight)
            asks = sorted(order_depths[product].sell_orders.items(), key=lambda x: x[0])
            volume = 0
            price = 0
            take_price = 0
            for i, ask in enumerate(asks[:max_levels]):
                volume += abs(ask[1])
                price += ask[0] * abs(ask[1])
                if volume >= weight:
                    if i > 0:
                        price -= ask[0] * (volume - weight)
                        volume = weight
                    else:
                        price -= ask[0] * (volume % weight)
                        volume -= volume % weight
                    take_price = ask[0]
                    break
            best_asks[product] = (take_price, price / volume if volume >= weight else np.nan, volume // weight)
        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = sum(item_per_basket[product] * price for product, (_, price, _) in best_bids.items())
        implied_ask = sum(item_per_basket[product] * price for product, (_, price, _) in best_asks.items())

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid != np.nan:
            implied_bid_volume = min(
                volume for _, (_, _, volume) in best_bids.items()
            )
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask != np.nan:
            implied_ask_volume = min(
                volume for _, (_, _, volume) in best_asks.items()
            )
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume
        return synthetic_order_price, {product: take_price for product, (take_price, _, _) in best_bids.items() if take_price != 0}, {product: take_price for product, (take_price, _, _) in best_asks.items() if take_price != 0}

    def convert_synthetic_basket_order(
        self, 
        basket_order: Order,
        synthetic_order: Order, 
        basket_bid_take_price: Dict[str, int], 
        basket_ask_take_price: Dict[str, int], 
        synthetic_bid_take_price: Dict[str, int],
        synthetic_ask_take_price: Dict[str, int],
        product: str,
        product_positions: Dict[str, int],
        order_depths: Dict[str, OrderDepth],
    ) -> None | Dict[str, Order]:
        # Initialize the dictionary to store component orders
        component_orders = dict()
        # Limit the quantity by position limits and orderbook size
        for component, quantity in eval(f'{product}_WEIGHTS').items():
            # Positions limits
            new_position = product_positions.get(component, 0) + synthetic_order.quantity * quantity
            if new_position > self.LIMIT[component]:
                synthetic_order.quantity = math.floor((self.LIMIT[component] - product_positions.get(component, 0)) / quantity)
            elif new_position < -self.LIMIT[component]:
                synthetic_order.quantity = math.ceil((-self.LIMIT[component] - product_positions.get(component, 0)) / quantity)
            # # Orderbook size limits
            # available_quantity = 0
            # if synthetic_quantity > 0:
            #     for price, size in order_depths[component].sell_orders.items():
            #         if price <= ask_take_price[component]:
            #             available_quantity -= size
            #     if available_quantity < synthetic_quantity * quantity:
            #         synthetic_quantity = math.floor(available_quantity / quantity)
            # else:
            #     for price, size in order_depths[component].buy_orders.items():
            #         if price >= bid_take_price[component]:
            #             available_quantity -= size
            #     if available_quantity > synthetic_quantity * quantity:
            #         synthetic_quantity = math.ceil(available_quantity / quantity)
            
        for component, quantity in eval(f'{SYNTHETIC[product]}_WEIGHTS').items():
            # Positions limits
            new_position = product_positions.get(component, 0) + basket_order.quantity * quantity
            if new_position > self.LIMIT[component]:
                basket_order.quantity = math.floor((self.LIMIT[component] - product_positions.get(component, 0)) / quantity)
            elif new_position < -self.LIMIT[component]:
                basket_order.quantity = math.ceil((-self.LIMIT[component] - product_positions.get(component, 0)) / quantity)

        if basket_order.quantity == 0 or synthetic_order.quantity == 0:
            return None
        if abs(basket_order.quantity) > abs(synthetic_order.quantity):
            basket_order.quantity = -synthetic_order.quantity
        elif abs(basket_order.quantity) < abs(synthetic_order.quantity):
            synthetic_order.quantity = -basket_order.quantity
                          
        # Execute orders and update order depths and product_positions as if the orders were taken
        # Synthetic orders
        for component, quantity in eval(f'{product}_WEIGHTS').items():
            if synthetic_order.quantity > 0:
                volume = quantity * synthetic_order.quantity
                component_orders[component] = Order(
                    component,
                    synthetic_ask_take_price[component],
                    volume,
                )
                # Update positions
                product_positions[component] = product_positions.get(component, 0) + volume
                # Update order_depths
                for price, size in sorted(order_depths[component].sell_orders.items(), key=lambda x: x[0]):
                    if volume >= -size:
                        volume += size
                        order_depths[component].sell_orders[price] = 0 # Don't delete level to keep max_levels
                        # del order_depths[component].sell_orders[price]
                    else:
                        order_depths[component].sell_orders[price] += volume
                        break
                
            else:
                volume = quantity * synthetic_order.quantity
                component_orders[component] = Order(
                    component,
                    synthetic_bid_take_price[component],
                    volume,
                )
                # Update positions
                product_positions[component] = product_positions.get(component, 0) + volume
                # Update order_depths
                for price, size in sorted(order_depths[component].buy_orders.items(), key=lambda x: x[0], reverse=True):
                    if -volume >= size:
                        volume += size
                        order_depths[component].buy_orders[price] = 0
                        # del order_depths[component].buy_orders[price]
                    else:
                        order_depths[component].buy_orders[price] += volume
                        break
        # Basket orders
        for component, quantity in eval(f'{SYNTHETIC[product]}_WEIGHTS').items():
            if basket_order.quantity > 0:
                volume = quantity * basket_order.quantity
                component_orders[component] = Order(
                    component,
                    basket_ask_take_price[component],
                    volume,
                )
                # Update positions
                product_positions[component] = product_positions.get(component, 0) + volume
                # Update order_depths
                for price, size in sorted(order_depths[component].sell_orders.items(), key=lambda x: x[0]):
                    if volume >= -size:
                        volume += size
                        order_depths[component].sell_orders[price] = 0 # Don't delete level to keep max_levels
                        # del order_depths[component].sell_orders[price]
                    else:
                        order_depths[component].sell_orders[price] += volume
                        break
                
            else:
                volume = quantity * basket_order.quantity
                component_orders[component] = Order(
                    component,
                    basket_bid_take_price[component],
                    volume,
                )
                # Update positions
                product_positions[component] = product_positions.get(component, 0) + volume
                # Update order_depths
                for price, size in sorted(order_depths[component].buy_orders.items(), key=lambda x: x[0], reverse=True):
                    if -volume >= size:
                        volume += size
                        order_depths[component].buy_orders[price] = 0
                        # del order_depths[component].buy_orders[price]
                    else:
                        order_depths[component].buy_orders[price] += volume
                        break

        return component_orders
    
    def execute_spread_orders(
        self,
        target_position: int,
        basket_position: int,
        basket_order_depth: OrderDepth,
        synthetic_order_depth: OrderDepth,
        basket_bid_take: Dict[str, int],
        basket_ask_take: Dict[str, int],
        synthetic_bid_take: Dict[str, int],
        synthetic_ask_take: Dict[str, int],
        product: str,
        product_positions: Dict[str, int],
        order_depths: Dict[str, OrderDepth],
    ) -> None | Dict[str, Order]:
        synthetic_product = SYNTHETIC[product]
        if target_position == basket_position:
            print(f'Target position {target_position} for {product} already reached.')
            return None

        target_quantity = abs(target_position - basket_position)
        
        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_order = Order(product, basket_ask_price, execute_volume)
            
            synthetic_order = Order(synthetic_product, synthetic_bid_price, -execute_volume)

        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_order = Order(product, basket_bid_price, -execute_volume)
            
            synthetic_order = Order(synthetic_product, synthetic_ask_price, execute_volume)
            
        aggregate_order = self.convert_synthetic_basket_order(
            basket_order, synthetic_order, basket_bid_take, basket_ask_take, synthetic_bid_take, synthetic_ask_take, product, product_positions, order_depths
        )
        return aggregate_order

    def spread_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        positions: Dict[str, int],
        product: Product,
        basket_position: int,
        TraderObject: Dict[str, Any],
        max_levels: int,
    ) -> None | Dict[str, Order]:
        if product not in SYNTHETIC.keys():
            return None
        synthetic_product = SYNTHETIC[product]
        spread_product = SPREAD[product]
        spread_data = TraderObject[spread_product]
        basket_order_depth, basket_bid_take, basket_ask_take = self.get_synthetic_basket_order_depth(order_depths, synthetic_product, max_levels)
        synthetic_order_depth, synthetic_bid_take, synthetic_ask_take = self.get_synthetic_basket_order_depth(order_depths, product, max_levels)
        basket_swmid, basket_best_bid, basket_best_ask = self.get_swmid(basket_order_depth)
        synthetic_swmid, synthetic_best_bid, synthetic_best_ask = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
       
        if spread != np.nan:
            spread_data["spread_history"].append(spread)

        if (
            len(spread_data["spread_history"])
            < self.params[spread_product]["spread_std_window"]
        ) or (spread == np.nan):
            return None
        elif len(spread_data["spread_history"]) > self.params[spread_product]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"]) if "default_spread_std" not in self.params[spread_product] else self.params[spread_product]["default_spread_std"]

        # zscore = (
        #     spread - self.params[spread_product]["default_spread_mean"]
        # ) / spread_std
        # spread_data["prev_zscore"] = zscore
        if (basket_best_bid - synthetic_best_ask - self.params[spread_product]["default_spread_mean"]) / spread_std >= self.params[spread_product]["zscore_threshold"]and basket_position > -self.params[spread_product]["target_position"]:
            return self.execute_spread_orders(
                -self.params[spread_product]["target_position"],
                basket_position,
                basket_order_depth,
                synthetic_order_depth,
                basket_bid_take,
                basket_ask_take,
                synthetic_bid_take,
                synthetic_ask_take,
                product,
                positions,
                order_depths,
            )

        if (basket_best_ask - synthetic_best_bid - self.params[spread_product]["default_spread_mean"]) / spread_std <= -self.params[spread_product]["zscore_threshold"] and basket_position < self.params[spread_product]["target_position"]:
            return self.execute_spread_orders(
                self.params[spread_product]["target_position"],
                basket_position,
                basket_order_depth,
                synthetic_order_depth,
                basket_bid_take,
                basket_ask_take,
                synthetic_bid_take,
                synthetic_ask_take,
                product,
                positions,
                order_depths,
            )
        return None

    def optimize_take_orders(self, products: List[str], result: Dict[str, List[Order]]) -> None:
        # Avoid crossed taking orders
        for product in products:
            if product not in result:
                continue
            quantity = 0
            bid_price = 0
            ask_price = np.inf
            # prod = result[product][0].quantity if len(result[product]) > 0 else 1 # debug
            for order in result[product]:
                quantity += order.quantity
                # prod *= quantity # debug
                if order.quantity > 0:
                    bid_price = max(bid_price, order.price) 
                else:
                    ask_price = min(ask_price, order.price)
            # if prod < 0:
            #     print(f'Opposite positions for {product}: {result[product]}') # debug
            if quantity > 0:
                result[product] = [Order(product, bid_price, quantity)]
            elif quantity < 0:
                result[product] = [Order(product, ask_price, quantity)]
            else:
                del result[product]
                
    def get_option_mid_price(
        self, option_order_depth: OrderDepth, traderData: Dict[str, Any]
    ) -> float:
        if (
            len(option_order_depth.buy_orders) > 0
            and len(option_order_depth.sell_orders) > 0
        ):
            best_bid = max(option_order_depth.buy_orders.keys())
            best_ask = min(option_order_depth.sell_orders.keys())
            traderData[f"option_prev_price"] = (best_bid + best_ask) / 2
            return (best_bid + best_ask) / 2
        else:
            return traderData[f"option_prev_price"]

    def delta_hedge_position(
        self,
        spot_order_depth: OrderDepth,
        option_position: int,
        spot_position: int,
        spot_buy_orders: int,
        spot_sell_orders: int,
        delta: float,
    ) -> List[Order]:
        """
        Delta hedge the overall position in VOLCANIC_ROCK_VOUCHER by creating orders in VOLCANIC_ROCK.

        Args:
            strike (int): The strike price of VOLCANIC_ROCK_VOUCHER.
            spot_order_depth (OrderDepth): The order depth for the VOLCANIC_ROCK product.
            option_position (int): The current position in VOLCANIC_ROCK_VOUCHER.
            spot_position (int): The current position in VOLCANIC_ROCK.
            spot_buy_orders (int): The total quantity of buy orders for VOLCANIC_ROCK in the current iteration.
            spot_sell_orders (int): The total quantity of sell orders for VOLCANIC_ROCK in the current iteration.
            delta (float): The current value of delta for the VOLCANIC_ROCK_VOUCHER product.
            traderData (Dict[str, Any]): The trader data for the VOLCANIC_ROCK_VOUCHER product.

        Returns:
            List[Order]: A list of orders to delta hedge the VOLCANIC_ROCK_VOUCHER position.
        """

        target_spot_position = -int(delta * option_position)
        hedge_quantity = target_spot_position - (
            spot_position + spot_buy_orders - spot_sell_orders
        )

        orders: List[Order] = []
        if hedge_quantity > 0:
            # Buy VOLCANIC_ROCK
            best_ask = min(spot_order_depth.sell_orders.keys())
            quantity = min(
                abs(hedge_quantity), -spot_order_depth.sell_orders[best_ask]
            )
            quantity = min(
                quantity,
                self.LIMIT[Product.VOLCANIC_ROCK] - (spot_position + spot_buy_orders),
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_ask, quantity))
        
        elif hedge_quantity < 0:
            # Sell VOLCANIC_ROCK
            best_bid = max(spot_order_depth.buy_orders.keys())
            quantity = min(
                abs(hedge_quantity), spot_order_depth.buy_orders[best_bid]
            )
            quantity = min(
                quantity,
                self.LIMIT[Product.VOLCANIC_ROCK] + (spot_position - spot_sell_orders),
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -quantity))

        return orders
    
    def delta_hedge_order(
        self,
        spot_order_depth: OrderDepth,
        option_orders: List[Order],
        spot_position: int,
        spot_buy_orders: int,
        spot_sell_orders: int,
        delta: float,
    ) -> List[Order]:
        """
        Delta hedge the new orders for VOLCANIC_ROCK_VOUCHER by creating orders in VOLCANIC_ROCK.

        Args:
            spot_order_depth (OrderDepth): The order depth for the VOLCANIC_ROCK product.
            option_orders (List[Order]): The new orders for VOLCANIC_ROCK_VOUCHER.
            spot_position (int): The current position in VOLCANIC_ROCK.
            spot_buy_orders (int): The total quantity of buy orders for VOLCANIC_ROCK in the current iteration.
            spot_sell_orders (int): The total quantity of sell orders for VOLCANIC_ROCK in the current iteration.
            delta (float): The current value of delta for the VOLCANIC_ROCK_VOUCHER product.

        Returns:
            List[Order]: A list of orders to delta hedge the new VOLCANIC_ROCK_VOUCHER orders.
        """
        if len(option_orders) == 0:
            return None

        net_option_quantity = sum(
            order.quantity for order in option_orders
        )
        target_spot_quantity = -int(delta * net_option_quantity)

        orders: List[Order] = []
        if target_spot_quantity > 0:
            # Buy VOLCANIC_ROCK
            best_ask = min(spot_order_depth.sell_orders.keys())
            quantity = min(
                abs(target_spot_quantity), -spot_order_depth.sell_orders[best_ask]
            )
            quantity = min(
                quantity,
                self.LIMIT[Product.VOLCANIC_ROCK] - (spot_position + spot_buy_orders),
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_ask, quantity))
        elif target_spot_quantity < 0:
            # Sell VOLCANIC_ROCK
            best_bid = max(spot_order_depth.buy_orders.keys())
            quantity = min(
                abs(target_spot_quantity), spot_order_depth.buy_orders[best_bid]
            )
            quantity = min(
                quantity,
                self.LIMIT[Product.VOLCANIC_ROCK] + (spot_position - spot_sell_orders),
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -quantity))
        return orders
    
    def hedge(
        self,
        spot_order_depth: OrderDepth,
        # option_orders: List[Order],
        spot_position: int,
        delta: float,
    ) -> List[Order]:
        
        target_spot_position = round(-delta)
        if abs(target_spot_position - spot_position) < 1.5:
            return list()

        target_spot_quantity = target_spot_position - spot_position
        if target_spot_position > self.LIMIT[Product.VOLCANIC_ROCK] or target_spot_position < -self.LIMIT[Product.VOLCANIC_ROCK]:
            print(f"Cannot fully hedge delta at position {target_spot_position}.")
        orders: List[Order] = []
        if target_spot_quantity > 0:
            # Buy VOLCANIC_ROCK
            best_ask = min(spot_order_depth.sell_orders.keys())
            quantity = min(
                abs(target_spot_quantity),
                self.LIMIT[Product.VOLCANIC_ROCK] - spot_position,
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_ask, round(quantity)))

        elif target_spot_quantity < 0:
            # Sell VOLCANIC_ROCK
            best_bid = max(spot_order_depth.buy_orders.keys())
            quantity = min(
                abs(target_spot_quantity),
                self.LIMIT[Product.VOLCANIC_ROCK] + spot_position,
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -round(quantity)))

        return orders
    
    def predict_option_vol_curve(
        self,
        tte: float,   # Current time
        spot_price: float,
        vol_curve: np.ndarray[float],
        traderData: Dict[str, Any],
        product: str = "VOLCANIC_ROCK",
        strikes: np.ndarray[int] = STRIKES,
    ) -> np.ndarray[float]:
        # Get the diff of fitted parameters and update traderData
        coef, base_iv = self.fit_iv(tte, spot_price, vol_curve, strikes)
        base_iv_diff = traderData['option_base_iv'] - base_iv
        coef_diff = traderData['option_parabolic_coef'] - coef
        traderData['option_base_iv'] = base_iv
        traderData['option_parabolic_coef'] = coef
        predicted_coef = coef + self.params[product]['option_parabolic_coef_beta'] * coef_diff
        predicted_base_iv = base_iv + self.params[product]['option_base_iv_beta'] * base_iv_diff
        # Calculate the predicted vol curve
        return predicted_coef * np.power(np.log(strikes / spot_price), 2) / (tte - 1 / 2500000) + predicted_base_iv
        
    def fit_iv(
        self, 
        tte: float, 
        spot_price: float,
        vol_curve: np.ndarray[float], 
        strikes: np.ndarray[int] = STRIKES
    ) -> tuple[float, float]:
        # Fit IVs of different strike price with a parabolic function.
        x2 = np.power(np.log(strikes /spot_price), 2) / tte
        cov = np.cov(x2, vol_curve)
        coef_ = cov[0][1] / cov[0][0]
        return coef_, np.mean(vol_curve) - coef_ * np.mean(x2)
        
    def get_vol_curve(
        self,
        tte: float,
        spot_price: float, 
        order_depths: Dict[str, OrderDepth],
        traderObject: Dict[str, Dict[str, Any]],
        product: str = "VOLCANIC_ROCK",
        strikes: np.ndarray[int] = STRIKES,
    ) -> np.ndarray[float]:
        
        vol_curve = np.zeros(len(strikes), dtype=np.float64)
        for i, strike in enumerate(strikes):
            option_order_depth = order_depths[f"{product}_VOUCHER_{strike}"]
            option_price = self.get_option_mid_price(option_order_depth, traderObject[f"{product}_VOUCHER_{strike}"])
            vol_curve[i] = BlackScholes.implied_volatility(
                option_price,
                spot_price,
                strike,
                tte
            )
        return vol_curve
    
    def predict_option_prices(
        self,
        tte: float,
        spot_price: float, 
        vol_curve: np.ndarray[float],
        strikes: np.ndarray[int] = STRIKES,
    ) -> np.ndarray[float]:
        
        predicted_prices = np.zeros(len(strikes), dtype=np.float64)
        for i, strike in enumerate(strikes):
            predicted_prices[i] = BlackScholes.black_scholes_call(
                spot_price,
                strike,
                tte - 1 / 2500000,
                vol_curve[i],
            )
        return predicted_prices
    
    def predict_option_delta(
        self,
        tte: float,
        spot_price: float,
        predicted_vol_curve: np.ndarray[float],
        strikes: np.ndarray[int] = STRIKES
    ) -> np.ndarray[float]:
        predicted_deltas = np.zeros(len(strikes), dtype=np.float64)
        for i, strike in enumerate(strikes):
            predicted_deltas[i] = BlackScholes.delta(
                spot_price,
                strike,
                tte - 1 / 2500000,
                predicted_vol_curve[i],
            )
        return predicted_deltas
    
    def predict_option_vega(
        self,
        tte: float,
        spot_price: float,
        predicted_vol_curve: np.ndarray[float],
        strikes: np.ndarray[int] = STRIKES
    ) -> np.ndarray[float]:
        predicted_vegas = np.zeros(len(strikes), dtype=np.float64)
        for i, strike in enumerate(strikes):
            predicted_vegas[i] = BlackScholes.vega(
                spot_price,
                strike,
                tte - 1 / 2500000,
                predicted_vol_curve[i],
            )
        return predicted_vegas
        
    def short_option_order(
        self,
        strike: int,
        option_order_depth: OrderDepth,
        option_position: int,
        product: str = "VOLCANIC_ROCK",
    ) -> tuple[List[Order], List[Order]]:
        target_option_position = -(self.LIMIT[f"{product}_VOUCHER_{strike}"] // 2)
        if option_position == target_option_position:
            return [], []
        if len(option_order_depth.buy_orders) > 0:
            best_bid = max(option_order_depth.buy_orders.keys())
            target_quantity = abs(
                target_option_position - option_position
            )
            quantity = min(
                target_quantity,
                abs(option_order_depth.buy_orders[best_bid]),
            )
            quote_quantity = target_quantity - quantity
            if quote_quantity == 0:
                return [Order(f"{product}_VOUCHER_{strike}", best_bid, int(-quantity))], []
            else:
                return [Order(f"{product}_VOUCHER_{strike}", best_bid, int(-quantity))], [
                    Order(f"{product}_VOUCHER_{strike}", best_bid, int(-quote_quantity))
                ]
        return [], []
    
    def long_option_order(
        self,
        strike: int,
        option_order_depth: OrderDepth,
        option_position: int,
        product: str = "VOLCANIC_ROCK",
    ) -> tuple[List[Order], List[Order]]:
        target_option_position = self.LIMIT[f"{product}_VOUCHER_{strike}"] // 2
        if option_position == target_option_position:
            return [], []
        if len(option_order_depth.sell_orders) > 0:
            best_ask = min(option_order_depth.sell_orders.keys())
            target_quantity = target_option_position - option_position
            quantity = min(
                target_quantity,
                abs(option_order_depth.sell_orders[best_ask]),
            )
            quote_quantity = target_quantity - quantity
            if quote_quantity == 0:
                return [Order(f"{product}_VOUCHER_{strike}", best_ask, int(quantity))], []
            else:
                return [Order(f"{product}_VOUCHER_{strike}", best_ask, int(quantity))], [
                    Order(f"{product}_VOUCHER_{strike}", best_ask, int(quote_quantity))
                ]
        return [], []
 
        
    def option_orders(
        self,
        order_depths: Dict[Product, OrderDepth],
        tte: float,
        spot_price: float,
        theo_base_iv: float,
        current_delta: float,
        option_positions: np.ndarray[int],
        deltas: np.ndarray[float],
        vegas: np.ndarray[float],
        traderData: Dict[str, Any],
        vol_curve: np.ndarray[float],
        product: str = "VOLCANIC_ROCK",
        strikes: np.ndarray[int] = STRIKES,
        trade_mask: None | np.ndarray[bool] = None,
    ) -> tuple[List[List[Order]], List[List[Order]], float]:
        if trade_mask is not None:
            assert(trade_mask.shape == strikes.shape)
        else:
            trade_mask = np.ones(len(strikes), dtype=bool)
        current_parabolic_coef, current_base_iv = self.fit_iv(tte, spot_price, vol_curve, strikes)
        traderData[f"option_base_iv_history"].append(current_base_iv)
        std = 0
        if (
            len(traderData[f"option_base_iv_history"])
            < self.params[product]["window"]
        ):
            std = self.params[product]["default_base_iv_std"]
        else:
            std = np.std(traderData[f"option_base_iv_history"])
            if len(traderData[f"option_base_iv_history"]) > self.params[product]["window"]:
                traderData[f"option_base_iv_history"].pop(0)

        vol_z_score = (current_base_iv - theo_base_iv) / std
        # print(f"vol_z_score: {vol_z_score}")
        take_order, make_order = [[] for _ in strikes], [[] for _ in strikes]
        if vol_z_score >= self.params[product]["upper_threshold"]:
            # Calculate fair vol for each strike
            # Method 1: Use global average of parabolic_coef
            parabolic_coef = self.params[product]["default_parabolic_coef"]
            # Method 2: Use current parabolic_coef_
            # parabolic_coef = current_parabolic_coef
            
            x = np.power(np.log(strikes / spot_price), 2) / (tte - 1 / 2500000)
            fair_vol_curve = x * parabolic_coef + theo_base_iv
            vol_z_score_per_option = (vol_curve - fair_vol_curve) / np.array([self.params[f"{product}_VOUCHER_{strike}"]["default_std_iv"] for strike in strikes], dtype=np.float64)
            strikes_idx = np.argsort(-vol_z_score_per_option)
            # print(f"Volatility diff: {vol_curve - fair_vol_curve}")
            for i in strikes_idx:
                if vol_z_score_per_option[i] < 0:
                    break
                elif not trade_mask[i] or vol_z_score_per_option[i] < self.params[f"{product}_VOUCHER_{strikes[i]}"]["zscore_threshold"]:
                    continue
                strike = strikes[i]
                option_take_order, option_make_order = self.short_option_order(
                    strike,
                    order_depths[f"{product}_VOUCHER_{strike}"],
                    option_positions[i],
                    product,
                )
                                
                current_delta += sum(order.quantity for order in option_take_order) * deltas[i]
                if current_delta < -self.LIMIT[product]:
                    current_delta -= sum(order.quantity for order in option_take_order) * deltas[i]
                    break
                take_order[i] += option_take_order
                current_delta += sum(order.quantity for order in option_make_order) * deltas[i]
                make_order[i] += option_make_order
                
                print(f"Current Vol: {vol_curve[i]} Fair Vol: {fair_vol_curve[i]}")
                print(f"Vega: {vegas[i]}")
                print(f"Expeted return {100 * vegas[i] * (vol_curve[i] - fair_vol_curve[i])}")

                if current_delta < -self.LIMIT[product]:
                    current_delta -= sum(order.quantity for order in option_make_order) * deltas[i]
                    break
                current_delta -= sum(order.quantity for order in option_make_order) * deltas[i]
                    
        elif vol_z_score <= -self.params[product]["lower_threshold"]:
            # Calculate fair vol for each strike
            # Method 1: Use global average of parabolic_coef
            parabolic_coef = self.params[product]["default_parabolic_coef"]
            # Method 2: Use current parabolic_coef_
            # parabolic_coef = current_parabolic_coef
            
            x = np.power(np.log(strikes / spot_price), 2) / (tte - 1 / 2500000)
            fair_vol_curve = x * parabolic_coef + theo_base_iv
            vol_z_score_per_option = (vol_curve - fair_vol_curve) / np.array([self.params[f"{product}_VOUCHER_{strike}"]["default_std_iv"] for strike in strikes], dtype=np.float64)

            strikes_idx = np.argsort(vol_z_score_per_option)
            # print(f"Volatility diff: {vol_curve - fair_vol_curve}")
            for i in strikes_idx:
                if vol_z_score_per_option[i] > 0:
                    break
                elif not trade_mask[i] or vol_z_score_per_option[i] > -self.params[f"{product}_VOUCHER_{strikes[i]}"]["zscore_threshold"]:
                    continue
                strike = strikes[i]
                option_take_order, option_make_order = self.long_option_order(
                    strike,
                    order_depths[f"{product}_VOUCHER_{strike}"],
                    option_positions[i],
                    product,
                )
                
                current_delta += sum(order.quantity for order in option_take_order) * deltas[i]
                if current_delta > self.LIMIT[product]:
                    current_delta -= sum(order.quantity for order in option_take_order) * deltas[i]
                    break
                take_order[i] += option_take_order
                current_delta += sum(order.quantity for order in option_make_order) * deltas[i]
                make_order[i] += option_make_order
                
                print(f"Current Vol: {vol_curve[i]} Fair Vol: {fair_vol_curve[i]}")
                print(f"Vega: {vegas[i]}")
                print(f"Expeted return {100 * vegas[i] * (fair_vol_curve[i] - vol_curve[i])}")
                
                if current_delta > self.LIMIT[product]:
                    current_delta -= sum(order.quantity for order in option_make_order) * deltas[i]
                    break
                current_delta -= sum(order.quantity for order in option_make_order) * deltas[i]

        # return take_order, make_order, current_delta
        # debug: take only
        return take_order, [[] for _ in strikes], current_delta
                
            
    
    # def option_orders(
    #     self,
    #     strike: int,
    #     option_order_depth: OrderDepth,
    #     option_position: int,
    #     option_fair_price: float,
    #     product: str = "VOLCANIC_ROCK",
    # ) -> tuple[List[Order]]:
    #     option = f"{product}_VOUCHER_{strike}"
    #     option_take_orders, buy_order_volume, sell_order_volume = (
    #         self.take_orders(
    #             option,
    #             option_order_depth,
    #             option_fair_price,
    #             self.params[product]["take_width"],
    #             option_position,
    #         )
    #     )
        
    #     option_clear_orders, buy_order_volume, sell_order_volume = (
    #         self.clear_orders(
    #             option,
    #             option_order_depth,
    #             option_fair_price,
    #             self.params[product]["clear_width"],
    #             option_position,
    #             buy_order_volume,
    #             sell_order_volume,
    #             self.params[product]["clear_threshold"],
    #         )
    #     )
    #     option_make_orders, _, _ = self.make_orders(
    #         option,
    #         option_order_depth,
    #         option_fair_price,
    #         option_position,
    #         buy_order_volume,
    #         sell_order_volume,
    #         self.params[product]["disregard_edge"],
    #         self.params[product]["join_edge"],
    #         self.params[product]["default_edge"],
    #     )
    #     return option_take_orders + option_clear_orders, option_make_orders
    
    def get_past_returns(
        self,
        traderObject: Dict[str, Any],
        order_depths: Dict[str, OrderDepth],
        timeframes: Dict[str, int],
    ) -> Dict[str, int]:
        returns_dict = {}

        for symbol, timeframe in timeframes.items():
            traderObject_key = f"{symbol}_price_history"
            if traderObject_key not in traderObject:
                traderObject[traderObject_key] = []

            price_history = traderObject[traderObject_key]

            if symbol in order_depths:
                order_depth = order_depths[symbol]
                if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                    current_price = (
                        max(order_depth.buy_orders.keys())
                        + min(order_depth.sell_orders.keys())
                    ) / 2
                else:
                    if len(price_history) > 0:
                        current_price = float(price_history[-1])
                    else:
                        returns_dict[symbol] = None
                        continue
            else:
                if len(price_history) > 0:
                    current_price = float(price_history[-1])
                else:
                    returns_dict[symbol] = None
                    continue

            price_history.append(
                f"{current_price:.1f}"
            )  # Convert float to string with 1 decimal place

            if len(price_history) > timeframe:
                price_history.pop(0)

            if len(price_history) == timeframe:
                past_price = float(price_history[0])  # Convert string back to float
                returns = (current_price - past_price) / past_price
                returns_dict[symbol] = returns
            else:
                returns_dict[symbol] = None

        return returns_dict   
            
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        # if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
        #     resin_position = (
        #         state.position[Product.RAINFOREST_RESIN]
        #         if Product.RAINFOREST_RESIN in state.position
        #         else 0
        #     )
        #     # if abs(resin_position) >= self.LIMIT[Product.RAINFOREST_RESIN]:
        #     #     print(f"Resin position limit reached at time {state.timestamp}. Current postiion: {resin_position}")
        #     resin_take_orders, buy_order_volume, sell_order_volume = (
        #         self.take_orders(
        #             Product.RAINFOREST_RESIN,
        #             state.order_depths[Product.RAINFOREST_RESIN],
        #             self.params[Product.RAINFOREST_RESIN]["fair_value"],
        #             self.params[Product.RAINFOREST_RESIN]["take_width"],
        #             resin_position,
        #         )
        #     )
        #     resin_clear_orders, buy_order_volume, sell_order_volume = (
        #         self.clear_orders(
        #             Product.RAINFOREST_RESIN,
        #             state.order_depths[Product.RAINFOREST_RESIN],
        #             self.params[Product.RAINFOREST_RESIN]["fair_value"],
        #             self.params[Product.RAINFOREST_RESIN]["clear_width"],
        #             resin_position,
        #             buy_order_volume,
        #             sell_order_volume,
        #             self.params[Product.RAINFOREST_RESIN]["clear_threshold"],
        #         )
        #     )
        #     resin_make_orders, _, _ = self.make_orders(
        #         Product.RAINFOREST_RESIN,
        #         state.order_depths[Product.RAINFOREST_RESIN],
        #         self.params[Product.RAINFOREST_RESIN]["fair_value"],
        #         resin_position,
        #         buy_order_volume,
        #         sell_order_volume,
        #         self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
        #         self.params[Product.RAINFOREST_RESIN]["join_edge"],
        #         self.params[Product.RAINFOREST_RESIN]["default_edge"],
        #         False,
        #         self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
        #     )
        #     result[Product.RAINFOREST_RESIN] = (
        #         resin_take_orders + resin_clear_orders + resin_make_orders
        #     )

        # if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
        #     if 'SQUID_INK' not in traderObject.keys():
        #         traderObject['SQUID_INK'] = {
        #             "price_history": [],
        #             "prev_price": 0,
        #             "clear_flag": False,
        #         }
        #     squidink_position = (
        #         state.position[Product.SQUID_INK]
        #         if Product.SQUID_INK in state.position
        #         else 0
        #     )
        #     # if squidink_position >= self.LIMIT[Product.SQUID_INK]:
        #     #     print(f"Squidink position limit reached at {state.timestamp}. Current position: {squidink_position}")
        #     squidink_fair_value = self.squidink_fair_value(
        #         state.order_depths[Product.SQUID_INK], traderObject
        #     )
        #     squidink_take_orders, buy_order_volume, sell_order_volume = (
        #         self.take_orders_ink(
        #             Product.SQUID_INK,
        #             state.order_depths[Product.SQUID_INK],
        #             squidink_fair_value,
        #             self.params[Product.SQUID_INK]["take_width"],
        #             squidink_position,
        #             self.params[Product.SQUID_INK]["prevent_adverse"],
        #             self.params[Product.SQUID_INK]["adverse_volume"],
        #             state.timestamp,
        #             traderObject,
        #         )
        #     )
        #     # squidink_clear_orders, buy_order_volume, sell_order_volume = (
        #     #     self.clear_orders(
        #     #         Product.SQUID_INK,
        #     #         state.order_depths[Product.SQUID_INK],
        #     #         squidink_fair_value,
        #     #         self.params[Product.SQUID_INK]["clear_width"],
        #     #         squidink_position,
        #     #         buy_order_volume,
        #     #         sell_order_volume,
        #     #         self.params[Product.SQUID_INK]["clear_threshold"],
        #     #     )
        #     # )
        #     # squidink_make_orders, _, _ = self.make_orders(
        #     #     Product.SQUID_INK,
        #     #     state.order_depths[Product.SQUID_INK],
        #     #     squidink_fair_value,
        #     #     squidink_position,
        #     #     buy_order_volume,
        #     #     sell_order_volume,
        #     #     self.params[Product.SQUID_INK]["disregard_edge"],
        #     #     self.params[Product.SQUID_INK]["join_edge"],
        #     #     self.params[Product.SQUID_INK]["default_edge"],
        #     # )
        #     result[Product.SQUID_INK] = (
        #         squidink_take_orders
        #         # + squidink_clear_orders + squidink_make_orders
        #     )
            
        # if Product.KELP in self.params and Product.KELP in state.order_depths:
        #     kelp_position = (
        #         state.position[Product.KELP]
        #         if Product.KELP in state.position
        #         else 0
        #     )
        #     # if abs(kelp_position) >= self.LIMIT[Product.KELP]:
        #     #     print(f"Kelp position limit reached at {state.timestamp}. Current position: {kelp_position}")
        #     kelp_fair_value = self.kelp_fair_value(
        #         state.order_depths[Product.KELP], traderObject
        #     )
        #     # kelp_fair_value = self.kelp_weighted_fair_value(
        #     #     state.order_depths[Product.KELP], traderObject,
        #     # )
        #     if kelp_fair_value is None:
        #         kelp_fair_value = traderObject.get("kelp_last_price", None)
            
        #     if kelp_fair_value is not None:
        #         # Take orders may have negative effect
        #         kelp_take_orders, buy_order_volume, sell_order_volume = (
        #             self.take_orders(
        #                 Product.KELP,
        #                 state.order_depths[Product.KELP],
        #                 kelp_fair_value,
        #                 self.params[Product.KELP]["take_width"],
        #                 kelp_position,
        #                 self.params[Product.KELP]["prevent_adverse"],
        #                 self.params[Product.KELP]["adverse_volume"],
        #             )
        #         )
        #         kelp_clear_orders, buy_order_volume, sell_order_volume = (
        #             self.clear_orders(
        #                 Product.KELP,
        #                 state.order_depths[Product.KELP],
        #                 kelp_fair_value,
        #                 self.params[Product.KELP]["clear_width"],
        #                 kelp_position,
        #                 buy_order_volume,
        #                 sell_order_volume,
        #                 self.params[Product.KELP]["clear_threshold"],
        #             )
        #         )
        #         kelp_make_orders, _, _ = self.make_orders(
        #             Product.KELP,
        #             state.order_depths[Product.KELP],
        #             kelp_fair_value,
        #             kelp_position,
        #             buy_order_volume,
        #             sell_order_volume,
        #             self.params[Product.KELP]["disregard_edge"],
        #             self.params[Product.KELP]["join_edge"],
        #             self.params[Product.KELP]["default_edge"],
        #             True,
        #             self.params[Product.KELP]["soft_position_limit"],
        #         )
        #         result[Product.KELP] = (
        #             kelp_take_orders + kelp_clear_orders + kelp_make_orders
        #         )
                
        # # Spread orders
        # for product in [Product.BASKET1BY2, Product.PICNIC_BASKET1]:
        #     spread_product = SPREAD[product]
        #     if spread_product not in traderObject.keys():
        #         traderObject[spread_product] = {
        #             "spread_history": [],
        #             "position": 0,
        #             # "prev_zscore": 0,
        #             # "curr_avg": 0,
        #         }
        #     order_depths = state.order_depths.copy()
        #     position = state.position.copy()
        #     spread_orders = self.spread_orders(
        #         order_depths,
        #         position,
        #         product,
        #         traderObject[spread_product]["position"],
        #         traderObject,
        #         self.params[spread_product]["max_taking_levels"],
        #     )
        
        #     if spread_orders is not None:
        #         for p, order in spread_orders.items():
        #             if p not in result.keys():
        #                 result[p] = []
        #             result[p].append(order)
                    
        #             # Track positions of trading pairs with shared component
        #             if p == Product.PICNIC_BASKET1:
        #                 traderObject[spread_product]["position"] += order.quantity / eval(f'{SYNTHETIC[product]}_WEIGHTS')[p]
        #                 # print(f"Time: {state.timestamp}")
        #                 # print(f"Current order: {product} {order.price} {order.quantity}")
        #                 # print(f"Basket1*2 position: {traderObject[SPREAD[Product.BASKET1BY2]]["position"]}")
        #                 # print(f"Basket1 position: {traderObject[SPREAD[Product.PICNIC_BASKET1]]["position"]}")
                                                                 
                
        #         # print(result)
        #         # print(state.order_depths[Product.PICNIC_BASKET1].buy_orders)
        #         # print(state.order_depths[Product.PICNIC_BASKET2].buy_orders)
        #         # print(state.order_depths[Product.DJEMBES].buy_orders)
        #         # print(state.order_depths[Product.PICNIC_BASKET1].sell_orders)
        #         # print(state.order_depths[Product.PICNIC_BASKET2].sell_orders)
        #         # print(state.order_depths[Product.DJEMBES].sell_orders)
                    
        # # Optimize spread orders
        # self.optimize_take_orders(
        #     [Product.DJEMBES, Product.PICNIC_BASKET1], result
        # )
         
        # Option trading
        product = Product.VOLCANIC_ROCK
        if product in self.params and product in state.order_depths and len(state.order_depths[product].buy_orders) > 0 and len(state.order_depths[product].sell_orders) > 0:
            if product not in traderObject:
                traderObject[product] = {
                    "option_base_iv": self.params[product]["default_base_iv"] + self.params["day"] * self.params[product]["base_iv_decay"],
                    "theoretical_option_base_iv": self.params[product]["default_base_iv"] + self.params["day"] * self.params[product]["base_iv_decay"],
                    "option_parabolic_coef": self.params[product]["default_parabolic_coef"],
                    "option_base_iv_beta": self.params[product]["option_base_iv_beta"],
                    "option_parabolic_coef_beta": self.params[product]["option_parabolic_coef_beta"],
                    "option_base_iv_history": [],
                }
            
            theoretical_base_iv = traderObject[product]["theoretical_option_base_iv"] + state.timestamp / 1000000 * self.params[product]["base_iv_decay"]        
            tte = self.params[product]["tte"] - (self.params["day"] + state.timestamp / 1000000) / 250
            spot_price = (
                        min(state.order_depths[product].buy_orders.keys())
                        + max(state.order_depths[product].sell_orders.keys())
                    ) / 2
            
            for strike in STRIKES:
                option = f"{product}_VOUCHER_{strike}"
                if option not in traderObject:
                    traderObject[option] = {
                        "option_prev_price": BlackScholes.black_scholes_call(
                            spot_price,
                            strike,
                            tte,
                            theoretical_base_iv + self.params[product]["default_parabolic_coef"] * np.power(np.log(strike / spot_price), 2) / tte,
                        ),
                    }
                    
            vol_curve = self.get_vol_curve(tte, spot_price, state.order_depths, traderObject, product) 
            predicted_delta = self.predict_option_delta(tte, spot_price, vol_curve)
            predicted_vega = self.predict_option_vega(tte, spot_price, vol_curve)
            # predicted_prices = self.predict_option_prices(tte, spot_price, vol_curve)
            option_positions = np.array([state.position[f"{product}_VOUCHER_{strike}"] if f"{product}_VOUCHER_{strike}" in state.position else 0 for strike in STRIKES])

            current_delta = np.dot(predicted_delta, option_positions)
            # Calculate zscore and set arbitrage orders
            take_orders, make_orders, current_delta = self.option_orders(
                state.order_depths,
                tte,
                spot_price,
                theoretical_base_iv,
                current_delta,
                option_positions,
                predicted_delta,
                predicted_vega,
                traderObject[product],
                vol_curve,
                product,
                STRIKES,
                trade_mask = np.array([False, False, True, False, False])
            )
            for i, (take_order, make_order) in enumerate(zip(take_orders, make_orders)):
                if len(take_order) > 0 or len(make_order) > 0:
                    option = f"{product}_VOUCHER_{STRIKES[i]}"
                    result[option] = (
                        take_order + make_order
                    )
                    # print(f"Current time: {state.timestamp}")
                    # print(f"Option at strike {STRIKES[i]}: {result[option]}")
                    # print(f"Current price: {traderObject[option]["option_prev_price"]}")
                    # print(f"Current delta: {predicted_delta[i]}")
                    # print(f"Current iv: {vol_curve[i]}")
                    # print(f"Current base iv : {traderObject[product]["option_base_iv_history"][-1]}")

            spot_orders = self.hedge(
                state.order_depths[product],
                state.position[product] if product in state.position else 0,
                current_delta,
            )
            if len(spot_orders) > 0:
                result[product] = spot_orders
                # print(f"Spot: {result[product]}")       
            
        conversions = 1
        traderData = jsonpickle.encode(traderObject)
        # traderData = None
        # if state.timestamp % 100000 == 99900:
        #     print(state.timestamp)
        #     print(state.position)
        #     print(self.get_basket_position(Product.BASKET1BY2, state))
        return result, conversions, traderData