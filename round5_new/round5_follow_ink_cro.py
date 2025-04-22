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

STRIKES = [10500, 10250, 10000, 9750, 9500]

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
        "max_taking_levels": 5,
    },
    
    Product.CROISSANTS:{
        "max_taking_levels": 1,
    },
    
    Product.KELP: {
        "take_width": 2,
        "price_zscore_threshold": 1.5,
        "price_ewma_alpha": 0.003,
        "price_std_window": 50,
        "default_price_mean": 2014,
        "default_price_std": 0.996,
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
        "spread_ewma_alpha": 0.001,
        "default_spread_mean": -125.5, # 57 for round 3, 49 for round 2
        "default_spread_std": 43,
        "spread_std_window": 100,
        "spread_zscore_threshold": 1,
        # "spread_clear_zscore_threshold": 1,
        "spread_zscore_full": 2,
        "target_position": 41, # max 41
        "max_taking_levels": 3,
    },
    
    Product.SPREAD12: {
        "spread_ewma_alpha": 0.001,
        "default_spread_mean": -419.4, # 23 for round 3, 3 for round 2,
        "default_spread_std": 118,
        "spread_std_window": 100,
        "spread_zscore_threshold": 1,
        # "spread_clear_zscore_threshold": 1,
        "spread_zscore_full": 2,
        "target_position": 33, # max 30 for single pair; max 33 for two pairs
        "max_taking_levels": 3,
    },
    
    Product.VOLCANIC_ROCK:{
        "tte": 8 / 250,  # in unit of years
        "default_base_iv": 0.126,
        # "default_base_iv_std": 0.009201,
        "option_iv_ewma_alpha": 0.002,
        "option_iv_window": 100,
        "base_iv_decay": -1.33e-3, # in unit of day
        "default_parabolic_coef": 0.264,
        # "option_base_iv_beta": -0.5,
        # "option_parabolic_coef_beta": -0.491,
        "option_vol_zscore_full": 2,
        "option_vol_zscore_threshold": 1,
        "option_price_threshold": 1,
        "option_make_edge": 1,
        "option_vega_threshold": 1, # in unit of 1%. Vega lower than this threshold will be onlhy used in hedging.
        # "option_hedge_delta_threshold": 0.95,
    },
    
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "default_iv_mean": 0.2123,
        "default_iv_std": 0.02301,
        "strike": 9500,
        "zscore_threshold": 1
    },
    
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "default_iv_mean": 0.1435,
        "default_iv_std": 0.01141,
        "strike": 9750,
        "zscore_threshold": 1
    },
    
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "default_iv_mean": 0.1235,
        "default_iv_std": 0.004798,
        "strike": 10000,
        "zscore_threshold": 1
    },
    
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "default_iv_mean": 0.1289,
        "default_iv_std": 0.005684,
        "strike": 10250,
        "zscore_threshold": 1
    },
    
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "default_iv_mean": 0.1717,
        "default_iv_std": 0.02091,
        "strike": 10500,
        "zscore_threshold": 1
    },
    
    "day": 5
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
    def black_scholes_d1(spot, strike, time_to_maturity, volatility):
        if volatility == 0:
            return np.inf
        d1 = (
            math.log(spot / strike) + (0.5 * volatility * volatility) * time_to_maturity
        ) / (volatility * math.sqrt(time_to_maturity))
        return d1
    
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
        return NormalDist().pdf(d1) * (spot * math.sqrt(time_to_maturity)) / 100

    @staticmethod
    def implied_volatility_binary(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-5
    ):
        if call_price + strike < spot:
            return 0.0  # Arbitrage condition
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility
    

    
    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_maturity, max_iterations=200, tolerance=1e-8, learning_rate=1
    ):
        if call_price < spot - strike:
            return 0

        # Initial guess for volatility
        volatility = 0.13  # Start with a reasonable guess
        for _ in range(max_iterations):
            d1 = (
            math.log(spot) - math.log(strike) + (0.5 * volatility * volatility) * time_to_maturity
        ) / (volatility * math.sqrt(time_to_maturity))
            d2 = d1 - volatility * math.sqrt(time_to_maturity)
            estimated_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
            vega = NormalDist().pdf(d1) * (spot * math.sqrt(time_to_maturity))

            # If vega is too small, the gradient descent may not converge
            if vega < 1e-10:
                return np.nan

            # Update volatility using gradient descent
            diff = estimated_price - call_price
            volatility -= learning_rate * diff / vega

            # Check for convergence
            if abs(diff) < tolerance:
                return volatility
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
    
    def market_order(
        self,
        product: Product,
        order_depth: OrderDepth,
        current_position: int,
        target_position: int,
        max_taking_levels: int = 1,
    ) -> Order | None:
        target_quantity = int(target_position - current_position)
        if target_quantity > 0:
            # Buy orders
            if len(order_depth.sell_orders) > 0:
                order_price = sorted(order_depth.sell_orders.keys())[min(max_taking_levels, len(order_depth.sell_orders)) - 1]
                return Order(product, order_price, target_quantity)
            else:
                return Order(product, max(order_depth.buy_orders.keys()) + 1, target_quantity)
        elif target_quantity < 0:
            # Sell orders
            if len(order_depth.buy_orders) > 0:
                order_price = sorted(order_depth.buy_orders.keys())[-min(max_taking_levels, len(order_depth.buy_orders))]
                return Order(product, order_price, target_quantity)
            else:
                return Order(product, min(order_depth.sell_orders.keys()) - 1, target_quantity)
        return None

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
        side: bool | None = None
    ) -> tuple[int, int]:
        
        position_limit = self.LIMIT[product] if position_limit is None else position_limit

        # Process sell orders (best ask)
        if len(order_depth.sell_orders) != 0 and side is not False:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            # Check adverse condition for sell orders
            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume or side is True:
                if best_ask <= fair_value - take_width: 
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
        if len(order_depth.buy_orders) != 0 and side is not True:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            # Check adverse condition for buy orders
            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume or side is False:
                if best_bid >= fair_value + take_width: 
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
        side: bool | None = None,
    ) -> tuple[int, int]:
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume) if side is not False else - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume) if side is not True else (position - sell_order_volume)
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
        side: bool | None = None,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > clear_threshold and side is not True:
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

        elif position_after_take < -clear_threshold and side is not False:
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

    def get_squidink_mid(self, order_depth: OrderDepth, traderData) -> float:
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
                if len(traderData["price_history"]) > 0:
                    mmmid_price = traderData["price_history"][-1]
                else:
                    mmmid_price = (best_ask + best_bid) / 2
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
            
            return mmmid_price
        return None
        
    def kelp_fair_price(self, order_depth: OrderDepth, traderData: Dict[str, Any]) -> tuple[float, float]:
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
                if traderData.get("last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2    
                else:
                    mmmid_price = traderData["last_price"]                         
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            imb = 0
            if mm_ask is None:
                imb = 1 if mm_bid is None else 0
            elif mm_bid is None:
                imb = -1
            else:
                imb = (order_depth.buy_orders[mm_bid] + order_depth.sell_orders[mm_ask]) /  (order_depth.buy_orders[mm_bid] - order_depth.sell_orders[mm_ask])

            if traderData.get("last_price", None) != None:
                last_price = traderData["last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"] + imb * self.params[Product.KELP]["imb_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderData["last_price"] = mmmid_price
            
            return fair, mmmid_price
        return None, None
    
    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
        side: bool | None = None
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
            side
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
        side: bool | None = None,
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
            side
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
        side: bool | None = None,  # True for buy, False for sell, None for both    
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

        if manage_position and side is None:
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
            side
        )

        return orders, buy_order_volume, sell_order_volume

    def get_swmid(self, order_depth) -> tuple[float, float, float]:
        best_bid = max(order_depth.buy_orders.keys()) if len(order_depth.buy_orders) != 0 else np.nan
        best_ask = min(order_depth.sell_orders.keys()) if len(order_depth.sell_orders) != 0 else np.nan
        best_bid_vol = abs(order_depth.buy_orders[best_bid]) if best_bid != np.nan else 0
        best_ask_vol = abs(order_depth.sell_orders[best_ask]) if best_ask != np.nan else 0
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol) if best_bid_vol != 0 and best_ask_vol != 0 else np.nan, best_bid, best_ask

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

        target_quantity = round(abs(target_position - basket_position))
        
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
       
        if np.isnan(spread):
            return None
        spread_data["spread_history"].append(spread)
        if (
            len(spread_data["spread_history"])
            <= self.params[spread_product]["spread_std_window"]
        ):  
            # spread_data["spread_mean"] += spread / self.params[spread_product]["spread_std_window"]
            # spread_data["spread_square_mean"] += spread ** 2 / self.params[spread_product]["spread_std_window"]
            return None
        spread_data["spread_ewma"] = spread_data["spread_ewma"] * (1 - self.params[spread_product]["spread_ewma_alpha"]) + spread * self.params[spread_product]["spread_ewma_alpha"]
        spread_data["spread_square_ewma"] = spread_data["spread_square_ewma"] * (1 - self.params[spread_product]["spread_ewma_alpha"]) + spread ** 2 * self.params[spread_product]["spread_ewma_alpha"]
        spread_data["spread_history"].pop(0)
        # spread_data["spread_mean"] += (spread - spread_data["spread_history"][0]) / self.params[spread_product]["spread_std_window"]
        # spread_data["spread_square_mean"] += (spread ** 2 - spread_data["spread_history"][0] ** 2) / self.params[spread_product]["spread_std_window"]
        
        fair = spread_data["spread_ewma"]
        spread_std = max(5e-5, math.sqrt(spread_data["spread_square_ewma"] - spread_data["spread_ewma"] ** 2)) 

        zscore = (
            spread - fair
        ) / spread_std

        # spread_data["prev_zscore"] = zscore
        if (basket_best_bid - synthetic_best_ask - fair) / spread_std >= self.params[spread_product]["spread_zscore_threshold"]:
            target_position = -round(self.params[spread_product]["target_position"] * min(1, (zscore - self.params[spread_product]["spread_zscore_threshold"]) / (self.params[spread_product]["spread_zscore_full"] - self.params[spread_product]["spread_zscore_threshold"])))
            if basket_position > target_position:
                return self.execute_spread_orders(
                    target_position,
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

        elif (basket_best_ask - synthetic_best_bid - fair) / spread_std <= -self.params[spread_product]["spread_zscore_threshold"]:
            target_position = round(self.params[spread_product]["target_position"] * min(1,  (-self.params[spread_product]["spread_zscore_threshold"] - zscore) / (self.params[spread_product]["spread_zscore_full"] - self.params[spread_product]["spread_zscore_threshold"])))
            if basket_position < target_position:
                return self.execute_spread_orders(
                    target_position,
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
            sorted_orders = sorted(result[product], key=lambda x: x.price)
            for order in sorted_orders:
                if order.quantity < 0:
                    ask_price = order.price
                    break
            for order in reversed(sorted_orders):
                if order.quantity > 0:
                    bid_price = order.price
                    break
            if bid_price >= ask_price:
                for order in sorted_orders:
                    if order.price < ask_price:
                        continue
                    if order.price > bid_price:
                        break
                    quantity += order.quantity

                if quantity > 0:
                    # print("Before optimization: ", [(order.price, order.quantity) for order in sorted_orders])
                    result[product] = [order for order in sorted_orders if order.price < ask_price or order.price > bid_price] + [Order(product, bid_price, quantity)]
                    # print("After optimization: ", [(order.price, order.quantity) for order in result[product]])
                elif quantity < 0:
                    # print("Before optimization: ", [(order.price, order.quantity) for order in sorted_orders])
                    result[product] = [order for order in sorted_orders if order.price < ask_price or order.price > bid_price] + [Order(product, ask_price, quantity)]
                    # print("After optimization: ", [(order.price, order.quantity) for order in result[product]])
                else:
                    # print("Before optimization: ", [(order.price, order.quantity) for order in sorted_orders])
                    result[product] = [order for order in sorted_orders if order.price < ask_price or order.price > bid_price] 
                    # print("After optimization: ", [(order.price, order.quantity) for order in result[product]])
                    if len(result[product]) == 0:
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
            return traderData[f"option_prev_price"] if traderData[f"option_prev_price"] != 0 else np.nan

     
    def hedge(
        self,
        order_depths: Dict[Product, OrderDepth],
        # option_orders: List[Order],
        positions: Dict[Product, int],
        delta: float,
        option_deltas: np.ndarray[float],
        product: Product = Product.VOLCANIC_ROCK,
        strikes: List[int] = STRIKES,
    ) -> Dict[Product, List[Order]]:
        spot_position = positions[product] if product in positions else 0
        if abs(spot_position + delta) < 1:
            return dict()
        target_spot_position = -delta
        target_spot_quantity = target_spot_position - spot_position
        spot_order_depth = order_depths[product]
        orders_dict = dict()
        if target_spot_quantity > 0:
            # Buy VOLCANIC_ROCK
            if len(spot_order_depth.sell_orders) > 0:
                best_ask = min(spot_order_depth.sell_orders.keys())
                quantity = min(
                    abs(target_spot_quantity),
                    self.LIMIT[product] - spot_position,
                )
                if quantity > 0:
                    orders_dict[product] = [Order(product, best_ask, round(quantity))]
                    target_spot_quantity -= quantity
                
        elif target_spot_quantity < 0:
            # Sell VOLCANIC_ROCK
            if len(spot_order_depth.buy_orders) > 0:
                best_bid = max(spot_order_depth.buy_orders.keys())
                quantity = min(
                    abs(target_spot_quantity),
                    self.LIMIT[product] + spot_position,
                )
                if quantity > 0:
                    orders_dict[product] = [Order(product, best_bid, -round(quantity))]
                    target_spot_quantity += quantity
                
        # Try hedging with deep ITM options
        if target_spot_quantity >= 1:
            strike_index = np.argsort(-option_deltas)
            sorted_deltas = option_deltas[strike_index]
            if len(sorted_deltas) == 0:
                print(f"Cannot fully hedge delta at position {target_spot_position}.")
            else:
                for i, option_delta in enumerate(sorted_deltas):
                    if target_spot_quantity < 1:
                        break
                    strike = strikes[strike_index[i]]
                    option_order_depth = order_depths[f"{product}_VOUCHER_{strike}"]
                    option_position = positions[f"{product}_VOUCHER_{strike}"] if f"{product}_VOUCHER_{strike}" in positions else 0
                    if len(option_order_depth.sell_orders) == 0:
                        best_ask = max(option_order_depth.buy_orders.keys()) + 1
                        quantity = min(
                            round(target_spot_quantity / option_delta),
                            self.LIMIT[f"{product}_VOUCHER_{strike}"] - option_position
                        )
                    else:
                        best_ask = min(option_order_depth.sell_orders.keys())
                        quantity = min(
                            round(target_spot_quantity / option_delta),
                            -option_order_depth.sell_orders[best_ask],
                            self.LIMIT[f"{product}_VOUCHER_{strike}"] - option_position
                        )
                    if quantity > 0:
                        orders_dict[f"{product}_VOUCHER_{strike}"] = [Order(f"{product}_VOUCHER_{strike}", best_ask, round(quantity))]
                        # print(f"Hedging with option at strike {strike}, delta {option_delta}, quantity {quantity}.")
                        target_spot_quantity -= quantity * option_delta
                        # print(f"Unheged delta: {hedge_quantity}.")
                if target_spot_quantity >= 1:
                    print(f"Cannot fully hedge delta with additional delta {target_spot_quantity}.")
        
        elif target_spot_quantity <= -1:
            strike_index = np.argsort(-option_deltas)
            sorted_deltas = option_deltas[strike_index]
            if len(sorted_deltas) == 0:
                print(f"Cannot fully hedge delta at position {target_spot_position}.")
            else:
                for i, option_delta in enumerate(sorted_deltas):
                    if target_spot_quantity > -1:
                        break
                    strike = strikes[strike_index[i]]
                    option_order_depth = order_depths[f"{product}_VOUCHER_{strike}"]
                    option_position = positions[f"{product}_VOUCHER_{strike}"] if f"{product}_VOUCHER_{strike}" in positions else 0
                    if len(option_order_depth.buy_orders) == 0:
                        best_bid = min(option_order_depth.sell_orders.keys()) - 1
                        quantity = min(
                            round(-target_spot_quantity / option_delta),
                            option_position + self.LIMIT[f"{product}_VOUCHER_{strike}"],
                        )
                    else:
                        best_bid = max(option_order_depth.buy_orders.keys())
                        quantity = min(
                            round(-target_spot_quantity / option_delta),
                            option_order_depth.buy_orders[best_bid],
                            option_position + self.LIMIT[f"{product}_VOUCHER_{strike}"],
                        )
                    if quantity > 0:
                        orders_dict[f"{product}_VOUCHER_{strike}"] = [Order(f"{product}_VOUCHER_{strike}", best_bid, -round(quantity))]
                        # print(f"Hedging with option at strike {strike}, delta {option_delta}, quantity {-quantity}.")
                        target_spot_quantity += quantity * option_delta
                        # print(f"Unhedged delta: {-hedge_quantity}.")
                if target_spot_quantity <= -1:
                    print(f"Cannot fully hedge delta with additional delta {target_spot_quantity}.")

        return orders_dict
    
    def predict_option_vol_curve(
        self,
        tte: float,   # Current time
        spot_price: float,
        vol_curve: np.ndarray[float],
        traderData: Dict[str, Any],
        product: str = "VOLCANIC_ROCK",
        strikes: List[int] = STRIKES,
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
        return predicted_coef * np.power(np.log(np.array(strikes, dtype=np.int32) / spot_price), 2) / (tte - 1 / 2500000) + predicted_base_iv
        
    def fit_iv(
        self, 
        tte: float, 
        spot_price: float,
        vol_curve: np.ndarray[float], 
        strikes: List[int] = STRIKES
    ) -> tuple[float, float]:
        # Fit IVs of different strike price with a parabolic function.
        x2 = np.power(np.log(np.array(strikes, dtype=np.int32) /spot_price), 2) / tte
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
        strikes: List[int] = STRIKES,
    ) -> np.ndarray[float]:
        
        vol_curve = np.zeros(len(strikes), dtype=np.float64)
        for i, strike in enumerate(strikes):
            option_order_depth = order_depths[f"{product}_VOUCHER_{strike}"]
            traderData = traderObject[f"{product}_VOUCHER_{strike}"]
            option_price = self.get_option_mid_price(option_order_depth, traderData)
            if np.isnan(option_price):
                if traderData["option_prev_vol"]  >= 0:
                    vol_curve[i] = traderData["option_prev_vol"] 
                else:
                    vol_curve[i] = np.nan
            else:
                vol_curve[i] = BlackScholes.implied_volatility(
                    option_price,
                    spot_price,
                    strike,
                    tte
                )       
                traderData["option_prev_vol"] = float(vol_curve[i])         
        return vol_curve
    
    def predict_option_prices(
        self,
        tte: float,
        spot_price: float, 
        vol_curve: np.ndarray[float],
        strikes: List[int] = STRIKES,
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
        traderObject: Dict[Product, Dict[str, Any]],
        product: Product = "VOLCANIC_ROCK",
        strikes: List[int] = STRIKES
    ) -> np.ndarray[float]:
        predicted_deltas = np.zeros(len(strikes), dtype=np.float64)
        for i, strike in enumerate(strikes):
            traderData = traderObject[f"{product}_VOUCHER_{strike}"]
            if np.isnan(predicted_vol_curve[i]):
                if len(traderData["option_iv_history"]) > 0:
                    vol = traderData["option_iv_history"][-1]
                else:
                    predicted_deltas[i] = 1.
                    continue
            else:
                vol = predicted_vol_curve[i]
            predicted_deltas[i] = BlackScholes.delta(
                spot_price,
                strike,
                tte - 1 / 2500000,
                vol,
            )
        return predicted_deltas
    
    def predict_option_vega(
        self,
        tte: float,
        spot_price: float,
        predicted_vol_curve: np.ndarray[float],
        traderObject: Dict[Product, Dict[str, Any]],
        product: Product = Product.VOLCANIC_ROCK,
        strikes: List[int] = STRIKES
    ) -> np.ndarray[float]:
        predicted_vegas = np.zeros(len(strikes), dtype=np.float64)
        for i, strike in enumerate(strikes):
            traderData = traderObject[f"{product}_VOUCHER_{strike}"]
            if np.isnan(predicted_vol_curve[i]):
                if len(traderData["option_iv_history"]) > 0:
                    vol = traderData["option_iv_history"][-1]
                else:
                    predicted_vegas[i] = 0.
                    continue
            else:
                vol = predicted_vol_curve[i]
            predicted_vegas[i] = BlackScholes.vega(
                spot_price,
                strike,
                tte - 1 / 2500000,
                vol,
            )
        return predicted_vegas
        
    def short_option_order(
        self,
        strike: int,
        option_order_depth: OrderDepth,
        option_position: int,
        option_target_position_ratio: float,
        product: Product = Product.VOLCANIC_ROCK,
    ) -> tuple[List[Order], List[Order]]:
        target_option_position = -round(self.LIMIT[f"{product}_VOUCHER_{strike}"] * min(option_target_position_ratio, 1))
        if option_position <= target_option_position:
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
                    Order(f"{product}_VOUCHER_{strike}", best_bid + self.params[product]["option_make_edge"], int(-quote_quantity))
                ]
        return [], []
    
    def long_option_order(
        self,
        strike: int,
        option_order_depth: OrderDepth,
        option_position: int,
        target_option_position_ratio: float,
        product: str = "VOLCANIC_ROCK",
    ) -> tuple[List[Order], List[Order]]:
        target_option_position = round(self.LIMIT[f"{product}_VOUCHER_{strike}"] * min(target_option_position_ratio, 1))
        if option_position >= target_option_position:
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
                    Order(f"{product}_VOUCHER_{strike}", best_ask - self.params[product]["option_make_edge"], int(quote_quantity))
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
        traderObject: Dict[str, Any],
        vol_curve: np.ndarray[float],
        product: str = "VOLCANIC_ROCK",
        strikes: List[int] = STRIKES,
    ) -> tuple[Dict[Product, List[Order]], float]:
        result = dict()
        for i, strike in enumerate(strikes):
            if np.isnan(vol_curve[i]):
                continue
            traderData = traderObject[f"{product}_VOUCHER_{strike}"]
            # Method 1: Use Base IV and parabolic curve
            # fair_vol = theo_base_iv + traderData[f"option_parabolic_coef"] * np.power(np.log(strike / spot_price), 2) / (tte - 1 / 2500000)
            # Method 2: Use sma of IVs
            current_vol = float(vol_curve[i])
            fair_vol = traderData["option_iv_ewma"]
            std = max(5e-5, math.sqrt(traderData["option_iv_square_ewma"] - traderData["option_iv_ewma"] ** 2))
            # Update history if vol is not extreme value
            if len(traderData["option_iv_history"]) < self.params[product]["option_iv_window"]:
                traderData["option_iv_history"].append(current_vol)
                # traderData["option_iv_mean"] += vol_curve[i] / self.params[product]["option_iv_window"]
                # traderData["option_iv_square_mean"] += vol_curve[i] ** 2 / self.params[product]["option_iv_window"]
                continue
            
            if current_vol >= fair_vol - std * 5 and current_vol <= fair_vol + std * 5:
                traderData["option_iv_history"].append(current_vol)
                traderData["option_iv_ewma"] = traderData["option_iv_ewma"] * (1 - self.params[product]["option_iv_ewma_alpha"]) + current_vol * self.params[product]["option_iv_ewma_alpha"]
                traderData["option_iv_square_ewma"] = traderData["option_iv_square_ewma"] * (1 - self.params[product]["option_iv_ewma_alpha"]) + current_vol ** 2 * self.params[product]["option_iv_ewma_alpha"]
                fair_vol = traderData["option_iv_ewma"]
                std = max(5e-5, math.sqrt(traderData["option_iv_square_ewma"] - traderData["option_iv_ewma"] ** 2))
                # traderData["option_iv_mean"] += (vol_curve[i] - traderData["option_iv_history"][0]) / self.params[product]["option_iv_window"]
                # traderData["option_iv_square_mean"] += (vol_curve[i] ** 2 - traderData["option_iv_history"][0] ** 2) / self.params[product]["option_iv_window"]
                traderData["option_iv_history"].pop(0)

            fair_price = BlackScholes.black_scholes_call(
                spot_price,
                strike,
                tte - 1 / 2500000,
                fair_vol,
            )
            zscore = (current_vol - fair_vol) / std
            option_bid = max(order_depths[f"{product}_VOUCHER_{strike}"].buy_orders.keys()) if len(order_depths[f"{product}_VOUCHER_{strike}"].buy_orders) > 0 else np.nan
            option_ask = min(order_depths[f"{product}_VOUCHER_{strike}"].sell_orders.keys()) if len(order_depths[f"{product}_VOUCHER_{strike}"].sell_orders) > 0 else np.nan
            if not np.isnan(option_bid) and zscore > std * self.params[product]["option_vol_zscore_threshold"] and option_bid - fair_price > self.params[product]["option_price_threshold"]:
                # print(f"Option strike: {strike}")
                # print(f"Current Price: {option_bid} Fair Price: {fair_price}")
                # print(f"Zscore: {(vol_curve[i] - fair_vol) / std}")
                
                option_take_order, option_make_order = self.short_option_order(
                    strike,
                    order_depths[f"{product}_VOUCHER_{strike}"],
                    option_positions[i],
                    (zscore - self.params[product]["option_vol_zscore_threshold"]) / (self.params[product]["option_vol_zscore_full"] - self.params[product]["option_vol_zscore_threshold"]),
                    product,
                )
            
                current_delta += sum(order.quantity for order in option_take_order) * deltas[i]
                if current_delta >= -self.LIMIT[product]:
                    result[f"{product}_VOUCHER_{strike}"] = option_take_order + option_make_order
                else:
                    current_delta -= sum(order.quantity for order in option_take_order) * deltas[i]
                
                # print(f"Current Vol: {bid_vol_curve[i]} Fair Vol: {mean}")
                # print(f"Vega: {vegas[i]}")
                # print(f"Expeted return {100 * vegas[i] * (vol_curve[i] - mean)}")

                # if current_delta < -self.LIMIT[product]:
                #     current_delta -= sum(order.quantity for order in option_make_order) * deltas[i]
                #     break
                # current_delta -= sum(order.quantity for order in option_make_order) * deltas[i]
         
            elif not np.isnan(option_ask) and zscore < -self.params[product]["option_vol_zscore_threshold"] and fair_price - option_ask > self.params[product]["option_price_threshold"]: 
                # print(f"Option strike: {strike}")
                # print(f"Current Price: {option_ask} Fair Price: {fair_price}")
                # print(f"Zscore: {(fair_vol - vol_curve[i]) / std}")
                option_take_order, option_make_order = self.long_option_order(
                    strike,
                    order_depths[f"{product}_VOUCHER_{strike}"],
                    option_positions[i],
                    (zscore + self.params[product]["option_vol_zscore_threshold"]) / (self.params[product]["option_vol_zscore_threshold"] - self.params[product]["option_vol_zscore_full"]),
                    product,
                )
                
                current_delta += sum(order.quantity for order in option_take_order) * deltas[i]
                if current_delta <= self.LIMIT[product]:
                    result[f"{product}_VOUCHER_{strike}"] = option_take_order + option_make_order
                else:
                    current_delta -= sum(order.quantity for order in option_take_order) * deltas[i]

                
                
                # if current_delta > self.LIMIT[product]:
                #     current_delta -= sum(order.quantity for order in option_make_order) * deltas[i]
                #     break
                # current_delta -= sum(order.quantity for order in option_make_order) * deltas[i]
        return result, current_delta
 
         
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

        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            # if abs(resin_position) >= self.LIMIT[Product.RAINFOREST_RESIN]:
            #     print(f"Resin position limit reached at time {state.timestamp}. Current postiion: {resin_position}")
            resin_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    resin_position,
                )
            )
            resin_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    resin_position,
                    buy_order_volume,
                    sell_order_volume,
                    self.params[Product.RAINFOREST_RESIN]["clear_threshold"],
                )
            )
            resin_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                False,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                resin_take_orders + resin_clear_orders + resin_make_orders
            )
        
        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            if Product.SQUID_INK not in traderObject.keys():
                traderObject[Product.SQUID_INK] = {
                    "high_price": None,
                    "low_price": None,
                    "long_flag": False,
                    "short_flag": False,
                }
            traderData = traderObject[Product.SQUID_INK]
            # Olivia knows the high and low price of SQUID_INK
            if Product.SQUID_INK in state.market_trades.keys(): 
                for trade in state.market_trades[Product.SQUID_INK]:
                    if trade.buyer == "Olivia":
                        traderData["low_price"] = min(trade.price, traderData["low_price"]) if traderData["low_price"] != None else trade.price
                        traderData["long_flag"] = True
                    elif trade.seller == "Olivia":
                        traderData["high_price"] = max(trade.price, traderData["high_price"]) if traderData["high_price"] != None else trade.price
                        traderData["short_flag"] = True
            if Product.SQUID_INK in state.own_trades.keys():
                for trade in state.own_trades[Product.SQUID_INK]:
                    if trade.buyer == "Olivia":
                        traderData["low_price"] = min(trade.price, traderData["low_price"]) if traderData["low_price"] != None else trade.price
                        traderData["long_flag"] = True
                    elif trade.seller == "Olivia":
                        traderData["high_price"] = max(trade.price, traderData["high_price"]) if traderData["high_price"] != None else trade.price
                        traderData["short_flag"] = True

            squidink_position = (
                state.position[Product.SQUID_INK]
                if Product.SQUID_INK in state.position
                else 0
            )
            # Stop trading when holding full position
            if squidink_position == self.LIMIT[Product.SQUID_INK]:
                traderData["long_flag"] = False
            elif squidink_position == -self.LIMIT[Product.SQUID_INK]:
                traderData["short_flag"] = False
            
            
            order = None
            if traderData["long_flag"]:
                order = self.market_order(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    squidink_position,
                    self.LIMIT[Product.SQUID_INK],
                    self.params[Product.SQUID_INK]["max_taking_levels"],
                )  
            elif traderData["short_flag"]:
                order = self.market_order(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    squidink_position,
                    -self.LIMIT[Product.SQUID_INK],
                    self.params[Product.SQUID_INK]["max_taking_levels"],
                )   
            if order is not None:
                result[Product.SQUID_INK] = [order]       
            # if squidink_position >= self.LIMIT[Product.SQUID_INK]:
            #     print(f"Squidink position limit reached at {state.timestamp}. Current position: {squidink_position}")
            # squidink_take_orders, buy_order_volume, sell_order_volume = (
            #     self.take_orders_ink(
            #         Product.SQUID_INK,
            #         state.order_depths[Product.SQUID_INK],
            #         squidink_position,
            #         self.params[Product.SQUID_INK]["prevent_adverse"],
            #         self.params[Product.SQUID_INK]["adverse_volume"],
            #         state.timestamp,
            #         traderObject[Product.SQUID_INK],
            #     )
            # )
            # result[Product.SQUID_INK] = (
            #     squidink_take_orders
            #     # + squidink_clear_orders + squidink_make_orders
            # )
            
        if Product.CROISSANTS in self.params and Product.CROISSANTS in state.order_depths:
            if Product.CROISSANTS not in traderObject.keys():
                traderObject[Product.CROISSANTS] = {
                    "high_price": None,
                    "low_price": None,
                    "long_flag": False,
                    "short_flag": False,
                }
            traderData = traderObject[Product.CROISSANTS]
            # Olivia knows the high and low price of CROISSANTS
            if Product.CROISSANTS in state.market_trades.keys(): 
                for trade in state.market_trades[Product.CROISSANTS]:
                    if trade.buyer == "Olivia":
                        traderData["low_price"] = min(trade.price, traderData["low_price"]) if traderData["low_price"] != None else trade.price
                        traderData["long_flag"] = True
                    elif trade.seller == "Olivia":
                        traderData["high_price"] = max(trade.price, traderData["high_price"]) if traderData["high_price"] != None else trade.price
                        traderData["short_flag"] = True
            if Product.CROISSANTS in state.own_trades.keys():
                for trade in state.own_trades[Product.CROISSANTS]:
                    if trade.buyer == "Olivia":
                        traderData["low_price"] = min(trade.price, traderData["low_price"]) if traderData["low_price"] != None else trade.price
                        traderData["long_flag"] = True
                    elif trade.seller == "Olivia":
                        traderData["high_price"] = max(trade.price, traderData["high_price"]) if traderData["high_price"] != None else trade.price
                        traderData["short_flag"] = True

            croissants_position = (
                state.position[Product.CROISSANTS]
                if Product.CROISSANTS in state.position
                else 0
            )
            # Stop trading when holding full position
            if croissants_position == self.LIMIT[Product.CROISSANTS]:
                traderData["long_flag"] = False
            elif croissants_position == -self.LIMIT[Product.CROISSANTS]:
                traderData["short_flag"] = False
            
            
            order = None
            if traderData["long_flag"]:
                order = self.market_order(
                    Product.CROISSANTS,
                    state.order_depths[Product.CROISSANTS],
                    croissants_position,
                    self.LIMIT[Product.CROISSANTS],
                    self.params[Product.CROISSANTS]["max_taking_levels"],
                )  
            elif traderData["short_flag"]:
                order = self.market_order(
                    Product.CROISSANTS,
                    state.order_depths[Product.CROISSANTS],
                    croissants_position,
                    -self.LIMIT[Product.CROISSANTS],
                    self.params[Product.CROISSANTS]["max_taking_levels"],
                )   
            if order is not None:
                result[Product.CROISSANTS] = [order]
            
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            if Product.KELP not in traderObject.keys():
                traderObject[Product.KELP] = {
                    "price_history": [],
                    "last_price": self.params[Product.KELP]["default_price_mean"],
                    # "price_mean": 0,
                    # "price_square_mean": 0,
                    # "price_ewma": self.params[Product.KELP]["default_price_mean"],
                    # "price_square_ewma": self.params[Product.KELP]["default_price_std"] ** 2 + self.params[Product.KELP]["default_price_mean"] ** 2,
                    "high_price": None,
                    "low_price": None,
                    "long_flag": False,
                    "short_flag": False,
                }
            traderData = traderObject[Product.KELP]
            # Olivia knows the high and low price of KELP
            if Product.KELP in state.market_trades.keys(): 
                for trade in state.market_trades[Product.KELP]:
                    if trade.buyer == "Olivia":
                        traderData["low_price"] = min(trade.price, traderData["low_price"]) if traderData["low_price"] != None else trade.price
                        traderData["long_flag"] = True
                    elif trade.seller == "Olivia":
                        traderData["high_price"] = max(trade.price, traderData["high_price"]) if traderData["high_price"] != None else trade.price
                        traderData["short_flag"] = True
            if Product.KELP in state.own_trades.keys():
                for trade in state.own_trades[Product.KELP]:
                    if trade.buyer == "Olivia":
                        traderData["low_price"] = min(trade.price, traderData["low_price"]) if traderData["low_price"] != None else trade.price
                        traderData["long_flag"] = True
                    elif trade.seller == "Olivia":
                        traderData["high_price"] = max(trade.price, traderData["high_price"]) if traderData["high_price"] != None else trade.price
                        traderData["short_flag"] = True

            
            kelp_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            # if abs(kelp_position) >= self.LIMIT[Product.KELP]:
            #     print(f"Kelp position limit reached at {state.timestamp}. Current position: {kelp_position}")
            kelp_fair_value, kelp_mmmid = self.kelp_fair_price(
                state.order_depths[Product.KELP], traderObject[Product.KELP]
            )
            
            if kelp_mmmid is not None:
                side = True if traderData["long_flag"] and not traderData["short_flag"] else False if traderData["short_flag"] and not traderData["long_flag"] else None
                kelp_take_orders, buy_order_volume, sell_order_volume = (
                    self.take_orders(
                        Product.KELP,
                        state.order_depths[Product.KELP],
                        kelp_fair_value,
                        self.params[Product.KELP]["take_width"],
                        kelp_position,
                        self.params[Product.KELP]["prevent_adverse"],
                        self.params[Product.KELP]["adverse_volume"],
                        side
                    )
                )
                
                kelp_clear_orders, buy_order_volume, sell_order_volume = (
                    self.clear_orders(
                        Product.KELP,
                        state.order_depths[Product.KELP],
                        kelp_fair_value,
                        self.params[Product.KELP]["clear_width"],
                        kelp_position,
                        buy_order_volume,
                        sell_order_volume,
                        self.params[Product.KELP]["clear_threshold"],
                        side
                    )
                )
                kelp_make_orders, _, _ = self.make_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume,
                    self.params[Product.KELP]["disregard_edge"],
                    self.params[Product.KELP]["join_edge"],
                    self.params[Product.KELP]["default_edge"],
                    True,
                    self.params[Product.KELP]["soft_position_limit"],
                    side
                )
                result[Product.KELP] = (
                    kelp_take_orders + kelp_clear_orders + kelp_make_orders
                )

        # Spread orders
        for product in [Product.BASKET1BY2]:
            spread_product = SPREAD[product]
            if spread_product not in traderObject.keys():
                traderObject[spread_product] = {
                    "spread_history": [],
                    "position": 0,
                    # "spread_mean": 0,
                    # "spread_square_mean": 0,
                    "spread_ewma": self.params[spread_product]["default_spread_mean"],
                    "spread_square_ewma": self.params[spread_product]["default_spread_std"] ** 2 + self.params[spread_product]["default_spread_mean"] ** 2,
                    # "prev_zscore": 0,
                    # "curr_avg": 0,
                }
            flag = True
            for element in eval(f"{product}_WEIGHTS"):
                if element not in state.order_depths.keys():
                    flag = False
                    break
            for element in eval(f"{SYNTHETIC[product]}_WEIGHTS"):
                if element not in state.order_depths.keys():
                    flag = False
                    break
            if not flag:
                continue

            spread_orders = self.spread_orders(
                state.order_depths,
                state.position,
                product,
                traderObject[spread_product]["position"],
                traderObject,
                self.params[spread_product]["max_taking_levels"],
            )
        
            if spread_orders is not None:
                for p, order in spread_orders.items():
                    if p not in result.keys():
                        result[p] = []
                    result[p].append(order)
                    
                    # Track positions of trading pairs with shared component
                    if p == Product.PICNIC_BASKET1:
                        traderObject[spread_product]["position"] += int(order.quantity / eval(f'{SYNTHETIC[product]}_WEIGHTS')[p])
                                                                                         
                                    
        # Optimize spread orders
        # self.optimize_take_orders(
        #     [Product.DJEMBES, Product.PICNIC_BASKET1], result
        # )
         
        # # Option trading
        # product = Product.VOLCANIC_ROCK
        # if product in self.params and product in state.order_depths and len(state.order_depths[product].buy_orders) > 0 and len(state.order_depths[product].sell_orders) > 0:
        #     if product not in traderObject:
        #         traderObject[product] = {
        #             "option_base_iv": self.params[product]["default_base_iv"] + self.params["day"] * self.params[product]["base_iv_decay"],
        #             "theoretical_option_base_iv": self.params[product]["default_base_iv"] + self.params["day"] * self.params[product]["base_iv_decay"],
        #             "option_parabolic_coef": self.params[product]["default_parabolic_coef"],
        #             # "option_base_iv_beta": self.params[product]["option_base_iv_beta"],
        #             # "option_parabolic_coef_beta": self.params[product]["option_parabolic_coef_beta"],
        #             # "option_base_iv_history": [],
        #         }
            
        #     theoretical_base_iv = traderObject[product]["theoretical_option_base_iv"] + state.timestamp / 1000000 * self.params[product]["base_iv_decay"]        
        #     tte = self.params[product]["tte"] - (self.params["day"] + state.timestamp / 1000000) / 250
        #     spot_price = (
        #                 min(state.order_depths[product].buy_orders.keys())
        #                 + max(state.order_depths[product].sell_orders.keys())
        #             ) / 2
            
        #     for strike in STRIKES:
        #         option = f"{product}_VOUCHER_{strike}"
        #         if option not in traderObject:
        #             traderObject[option] = {
        #                 "option_prev_price": 0,
        #                 "option_prev_vol": -1,
        #                 "option_iv_history": [],
        #                 "option_iv_ewma": self.params[option]["default_iv_mean"],
        #                 "option_iv_square_ewma": self.params[option]["default_iv_mean"] ** 2 + self.params[option]["default_iv_std"] ** 2,
        #             }
                    
        #     vol_curve = self.get_vol_curve(tte, spot_price, state.order_depths, traderObject, product) 
        #     # ask_vol_curve = self.get_ask_vol_curve(tte, spot_price, state.order_depths, product)
        #     # bid_vol_curve = self.get_bid_vol_curve(tte, spot_price, state.order_depths, product)
        #     predicted_delta = self.predict_option_delta(tte, spot_price, vol_curve, traderObject)
        #     predicted_vega = self.predict_option_vega(tte, spot_price, vol_curve, traderObject)
        #     # predicted_prices = self.predict_option_prices(tte, spot_price, vol_curve)
        #     option_positions = np.array([state.position[f"{product}_VOUCHER_{strike}"] if f"{product}_VOUCHER_{strike}" in state.position else 0 for strike in STRIKES])

        #     current_delta = np.dot(predicted_delta, option_positions)
        #     # Don't trade options with too low vega
        #     trade_mask = (predicted_vega > self.params[product]["option_vega_threshold"])
        #     # Calculate zscore and set arbitrage orders
        #     trading_options_strikes = [strike for i, strike in enumerate(STRIKES) if trade_mask[i]]
        #     option_orders_dict, current_delta = self.option_orders(
        #         state.order_depths,
        #         tte,
        #         spot_price,
        #         theoretical_base_iv,
        #         current_delta,
        #         option_positions[trade_mask],
        #         predicted_delta[trade_mask],
        #         traderObject,
        #         vol_curve[trade_mask],
        #         product,
        #         trading_options_strikes,
        #     )
        #     for p in option_orders_dict:
        #         if p in result.keys():
        #             result[p] += option_orders_dict[p]
        #         else:
        #             result[p] = option_orders_dict[p]
                    
                            
        #     # Use spot and options not being traded to hedge
        #     hedge_orders_dict = self.hedge(
        #         state.order_depths,
        #         state.position,
        #         current_delta,
        #         predicted_delta[~trade_mask],
        #         product,
        #         [strike for i, strike in enumerate(STRIKES) if not trade_mask[i]],
        #     )
        #     for p in hedge_orders_dict:
        #         if p in result.keys():
        #             result[p] += hedge_orders_dict[p]
        #         else:
        #             result[p] = hedge_orders_dict[p]
        #         # print(f"Spot: {result[product]}")
        #         # print(f"Current Delta after hedging: {current_delta + sum(order.quantity for order in spot_orders) + state.position.get(product, 0)}")
                       
            
        conversions = 1
        traderData = jsonpickle.encode(traderObject)
        return result, conversions, traderData