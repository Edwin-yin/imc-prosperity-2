from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math


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
    
    # Product.SQUID_INK: {
    #     "take_width": 2,
    #     "clear_width": 0,
    #     "clear_threshold": 0,
    #     "prevent_adverse": True,
    #     "adverse_volume": 15,
    #     "reversion_beta": -0.2,
    #     "disregard_edge": 1,
    #     "join_edge": 2,
    #     "default_edge": 4,
    #     "soft_position_limit": 15,
    # },
    
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
        "default_spread_mean": 49,
        # "default_spread_std": 40,
        "spread_std_window": 45,
        "zscore_threshold": 3,
        "target_position": 41, # max 41
        "max_taking_levels": 3,
    },
    
    Product.SPREAD12: {
        "default_spread_mean": 0,
        # "default_spread_std": 40,
        "spread_std_window": 45,
        "zscore_threshold": 4,
        "target_position": 33, # max 30 for single pair; max 33 for two pairs
        "max_taking_levels": 3,
    },
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
            Product.PICNIC_BASKET2: 100
        }

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
    ) -> tuple[int, int]:
        position_limit = self.LIMIT[product]

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

        if position_after_take < -clear_threshold:
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
            imb = (order_depth.buy_orders[best_bid] + order_depth.sell_orders[best_ask]) / (
                order_depth.buy_orders[best_bid] + abs(order_depth.sell_orders[best_ask])
            )
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
            if mm_ask == None or mm_bid == None:
                if traderObject.get("squidink_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["squidink_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("squidink_last_price", None) != None:
                last_price = traderObject["squidink_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.SQUID_INK]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["squidink_last_price"] = mmmid_price
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
            volume_ask = abs(order_depth.sell_orders[mm_ask]) if mm_ask != None else 0
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            volume_bid = abs(order_depth.buy_orders[mm_bid]) if mm_bid != None else 0
            imb = 0
            if mm_ask == None or mm_bid == None:
                if traderObject.get("kelp_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                    imb = (order_depth.buy_orders[best_bid] + order_depth.sell_orders[best_ask]) / (order_depth.buy_orders[best_bid] + abs(order_depth.sell_orders[best_ask]))
                else:
                    mmmid_price = traderObject["kelp_last_price"]
                    imb = (order_depth.buy_orders[best_bid] + order_depth.sell_orders[best_ask]) / (order_depth.buy_orders[best_bid] + abs(order_depth.sell_orders[best_ask]))
                    
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
                imb = (volume_bid - volume_ask) / (volume_bid + volume_ask)

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
            squidink_position = (
                state.position[Product.SQUID_INK]
                if Product.SQUID_INK in state.position
                else 0
            )
            # if abs(squidink_position) >= self.LIMIT[Product.SQUID_INK]:
            #     print(f"Squidink position limit reached at {state.timestamp}. Current position: {squidink_position}")
            squidink_fair_value = self.squidink_fair_value(
                state.order_depths[Product.SQUID_INK], traderObject
            )
            squidink_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    squidink_fair_value,
                    self.params[Product.SQUID_INK]["take_width"],
                    squidink_position,
                    self.params[Product.SQUID_INK]["prevent_adverse"],
                    self.params[Product.SQUID_INK]["adverse_volume"],
                )
            )
            squidink_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    squidink_fair_value,
                    self.params[Product.SQUID_INK]["clear_width"],
                    squidink_position,
                    buy_order_volume,
                    sell_order_volume,
                    self.params[Product.SQUID_INK]["clear_threshold"],
                )
            )
            squidink_make_orders, _, _ = self.make_orders(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                squidink_fair_value,
                squidink_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.SQUID_INK]["disregard_edge"],
                self.params[Product.SQUID_INK]["join_edge"],
                self.params[Product.SQUID_INK]["default_edge"],
            )
            result[Product.SQUID_INK] = (
                squidink_take_orders + squidink_clear_orders + squidink_make_orders
            )
            
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            # if abs(kelp_position) >= self.LIMIT[Product.KELP]:
            #     print(f"Kelp position limit reached at {state.timestamp}. Current position: {kelp_position}")
            kelp_fair_value = self.kelp_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            # kelp_fair_value = self.kelp_weighted_fair_value(
            #     state.order_depths[Product.KELP], traderObject,
            # )
            if kelp_fair_value is None:
                kelp_fair_value = traderObject.get("kelp_last_price", None)
            
            if kelp_fair_value is not None:
                # Take orders may have negative effect
                kelp_take_orders, buy_order_volume, sell_order_volume = (
                    self.take_orders(
                        Product.KELP,
                        state.order_depths[Product.KELP],
                        kelp_fair_value,
                        self.params[Product.KELP]["take_width"],
                        kelp_position,
                        self.params[Product.KELP]["prevent_adverse"],
                        self.params[Product.KELP]["adverse_volume"],
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
                )
                result[Product.KELP] = (
                    kelp_take_orders + kelp_clear_orders + kelp_make_orders
                )
        # Spread orders
        for product in [Product.BASKET1BY2, Product.PICNIC_BASKET1]:
            spread_product = SPREAD[product]
            if spread_product not in traderObject.keys():
                traderObject[spread_product] = {
                    "spread_history": [],
                    "position": 0,
                    # "prev_zscore": 0,
                    # "curr_avg": 0,
                }
            order_depths = state.order_depths.copy()
            position = state.position.copy()
            spread_orders = self.spread_orders(
                order_depths,
                position,
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
                        traderObject[spread_product]["position"] += order.quantity / eval(f'{SYNTHETIC[product]}_WEIGHTS')[p]
                        # print(f"Time: {state.timestamp}")
                        # print(f"Current order: {product} {order.price} {order.quantity}")
                        # print(f"Basket1*2 position: {traderObject[SPREAD[Product.BASKET1BY2]]["position"]}")
                        # print(f"Basket1 position: {traderObject[SPREAD[Product.PICNIC_BASKET1]]["position"]}")
                                                                 
                
                # print(result)
                # print(state.order_depths[Product.PICNIC_BASKET1].buy_orders)
                # print(state.order_depths[Product.PICNIC_BASKET2].buy_orders)
                # print(state.order_depths[Product.DJEMBES].buy_orders)
                # print(state.order_depths[Product.PICNIC_BASKET1].sell_orders)
                # print(state.order_depths[Product.PICNIC_BASKET2].sell_orders)
                # print(state.order_depths[Product.DJEMBES].sell_orders)
                    
        # Optimize spread orders
        self.optimize_take_orders(
            [Product.DJEMBES, Product.PICNIC_BASKET1], result
        )
        conversions = 1
        traderData = jsonpickle.encode(traderObject)
        # traderData = None
        # if state.timestamp % 100000 == 99900:
        #     print(state.timestamp)
        #     print(state.position)
        #     print(self.get_basket_position(Product.BASKET1BY2, state))
        return result, conversions, traderData