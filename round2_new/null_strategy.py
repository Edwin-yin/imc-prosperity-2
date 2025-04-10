from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math


class Trader:

    def run(self, state: TradingState):
        conversions = 1
        result = dict()
        traderData = None
        return result, conversions, traderData
