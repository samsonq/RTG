# Copyright 2021 Optiver Asia Pacific Pty. Ltd.
#
# This file is part of Ready Trader Go.
#
#     Ready Trader Go is free software: you can redistribute it and/or
#     modify it under the terms of the GNU Affero General Public License
#     as published by the Free Software Foundation, either version 3 of
#     the License, or (at your option) any later version.
#
#     Ready Trader Go is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public
#     License along with Ready Trader Go.  If not, see
#     <https://www.gnu.org/licenses/>.
import asyncio
import itertools
import time
import math
import numpy as np
import pandas as pd
import scipy

from typing import List

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side


LOT_SIZE = 50
POSITION_LIMIT = 100
TICK_SIZE_IN_CENTS = 100
MIN_BID_NEAREST_TICK = (MINIMUM_BID + TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
T = 900
WAIT_TIME = 0.25
UPDATE_TIME = 0.25
IMBAL_THRESHOLD = 0.2
INV_SCALAR = -0.05
ROLLING_WNDW = 4
g = 0.001
K = 2
VOL1 = 50
VOL2 = 100

def roundup(x):
    return int(math.ceil(x / 100.)) * 100

def rounddown(x):
    return int(math.floor(x / 100.)) * 100

def weighted_mid(bid_volumes, bid_prices, ask_volumes, ask_prices):
    V_bid = 0
    V_ask = 0
    for i in range(min(len(bid_volumes), len(ask_volumes))):
        V_bid += math.exp(-0.5*i) * bid_volumes[i]
        V_ask += math.exp(-0.5*i) * ask_volumes[i]
    I = V_bid/(V_bid + V_ask) if (V_bid + V_ask) != 0 else 0
    theo = I * ask_prices[0] + (1-I) * bid_prices[0]
    return theo

def weighted_price(bid_volumes, bid_prices, ask_volumes, ask_prices):
        P_w = 0
        V_cum = 0
        for i in range(min(len(bid_volumes), len(ask_volumes))):
            # calculate wpr as the guide price
            P_w += (bid_prices[i] * ask_volumes[i] + ask_prices[i] * bid_volumes[i])
            V_cum += (ask_volumes[i] + bid_volumes[i])
        wp = P_w / V_cum if V_cum != 0 else 0
        return wp
# get rid of the influence of the nan
def ts_append(value, ts):
    if value == 0 and ts:
        ts.append(ts[-1])
    if value != 0:
        ts.append(value)
    if not ts and value == 0:
        ts.append(0)
        
class AutoTrader(BaseAutoTrader):
    """Example Auto-trader.

    When it starts this auto-trader places ten-lot bid and ask orders at the
    current best-bid and best-ask prices respectively. Thereafter, if it has
    a long position (it has bought more lots than it has sold) it reduces its
    bid and ask prices. Conversely, if it has a short position (it has sold
    more lots than it has bought) then it increases its bid and ask prices.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        """Initialise a new instance of the AutoTrader class."""
        super().__init__(loop, team_name, secret)
        self.order_ids = itertools.count(1)
        self.bids = set()
        self.asks = set()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = self.position_cost = 0
        self.time = self.bid_qtime = self.ask_qtime = self.bid_etime = self.ask_etime = time.time()
        self.last_trade = self.hedge_price = 0
        self.trade_type = ''
        self.theo_F = self.theo_E = 0
        self.bid_volume = self.ask_volume = 0
        self.EBid = self.EAsk = self.FBid = self.FAsk = 0
        self.BA_F = self.BA_E = 0
        self.mid_F = self.mid_E = 0
        self.wpr_E = []  # volume weighted price ETF
        self.wpr_F = []  # Volume Weighted Price Fut
        self.vol_E = self.vol_F = self.N_E = self.N_F = self.signal = 0
        self.bid_ask_spreads = []
        self.diff_spread = []
        self.history = []
        self.cancel_signal_bid = 0
        self.cancel_signal_ask = 0

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s", client_order_id, error_message.decode())
        if client_order_id != 0 and (client_order_id in self.bids or client_order_id in self.asks):
            self.on_order_status_message(client_order_id, 0, 0, 0)

    def on_hedge_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your hedge orders is filled.

        The price is the average price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received hedge filled for order %d with average price %d and volume %d", client_order_id,
                         price, volume)

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically to report the status of an order book.

        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.
        """
        self.logger.info("received order book for instrument %d with sequence number %d", instrument,
                         sequence_number)
        # self.orderbook.append([instrument,sequence_number,ask_prices,ask_volumes,bid_prices,bid_volumes])
        # Futures Market
        if instrument == Instrument.FUTURE:
            # Calculate Weighted Price
            wp = weighted_mid(bid_volumes, bid_prices, ask_volumes, ask_prices)
            V_bid = 0
            V_ask = 0
            for i in range(min(len(bid_volumes), len(ask_volumes))):
                V_bid += math.exp(-0.5*i) * bid_volumes[i]
                V_ask += math.exp(-0.5*i) * ask_volumes[i]
            self.imbal = (V_bid - V_ask) / (V_bid + V_ask) if (V_bid + V_ask) != 0 else 0
            self.FBid = bid_prices[0]
            self.FAsk = ask_prices[0]
            self.mid_F = (self.FBid + self.FAsk)/2
            self.BA_F = self.FAsk - self.FBid
            self.theo_F = wp
            if self.signal == 0:
                if wp != 0:
                    self.signal = 1
            if self.signal == 1:
                self.N_F += 1
                ts_append(wp, self.wpr_F)
                # Calculate Volatility
                if self.N_F > ROLLING_WNDW:
                    self.wpr_F.pop(0)
                    self.vol_F = np.std(self.wpr_F)
            
        if instrument == Instrument.ETF:
            # Calculate Weighted Price
            wp = weighted_mid(bid_volumes, bid_prices, ask_volumes, ask_prices)
            self.EBid = bid_prices[0]
            self.EAsk = ask_prices[0]
            self.mid_E = (self.EBid + self.EAsk)/2
            self.BA_E = self.EAsk - self.EBid
            self.theo_E = wp
            if self.signal == 0:
                if wp != 0:
                    self.signal = 1
            if self.signal == 1:
                self.N_E += 1
                ts_append(wp, self.wpr_E)
                # Calculate Volatility
                if self.N_E > ROLLING_WNDW:
                    self.wpr_E.pop(0)
                    self.vol_F = np.std(self.wpr_E)
        
            if self.theo_F != 0:
                # Spread Calculation
                if self.vol_E < VOL1:
                    K1 = 4
                    K2 = 4
                elif self.vol_E < VOL2:
                    K1 = 6
                    K2 = 6
                else:
                    K1 = 8
                    K2 = 8
                if self.imbal > IMBAL_THRESHOLD:
                    new_ask_price = roundup(self.theo_F + K2 * TICK_SIZE_IN_CENTS)
                    new_bid_price = rounddown(self.theo_F - K1/2 * TICK_SIZE_IN_CENTS)
                    tt = "Bid Imbal"
                elif self.imbal < -IMBAL_THRESHOLD:
                    new_ask_price = roundup(self.theo_F + K1/2 * TICK_SIZE_IN_CENTS)
                    new_bid_price = rounddown(self.theo_F - K2 * TICK_SIZE_IN_CENTS)
                    tt = "Ask Imbal"
                else:
                    new_ask_price = roundup(self.theo_F + K1/2 * TICK_SIZE_IN_CENTS)
                    new_bid_price = rounddown(self.theo_F - K1/2 * TICK_SIZE_IN_CENTS)
                    tt = "Normal"
                if self.position >= 0:
                    new_bid_volume = round(LOT_SIZE * math.exp(INV_SCALAR * self.position))
                    new_ask_volume = round(LOT_SIZE)
                if self.position < 0:
                    new_ask_volume = round(LOT_SIZE * math.exp(-INV_SCALAR * self.position))
                    new_bid_volume = round(LOT_SIZE)
                # Orders
                # If no orders in book
                if self.bid_id == 0 and self.ask_id == 0:
                    # Check if new_price is not same as current price
                    if new_bid_price not in (self.bid_price, 0):
                        # Send orders
                        self.bid_id = next(self.order_ids)
                        self.bid_price = new_bid_price
                        self.send_insert_order(self.bid_id, Side.BUY, new_bid_price, new_bid_volume, Lifespan.GOOD_FOR_DAY)
                        self.logger.info("Order %d: Bid %d @ %d, Imbal: %.2f, Theo: %d, Type: " + tt, self.bid_id, new_bid_volume, new_bid_price, self.imbal, self.theo_F)
                        self.bids.add(self.bid_id)
                        self.bid_qtime = time.time()
                        self.bid_volume = new_bid_volume
                        self.trade_type = tt
                        
                    if new_ask_price not in (self.ask_price, 0):
                        self.ask_id = next(self.order_ids)
                        self.ask_price = new_ask_price
                        self.send_insert_order(self.ask_id, Side.SELL, new_ask_price, new_ask_volume, Lifespan.GOOD_FOR_DAY)
                        self.logger.info("Order %d: Ask %d @ %d, Imbal: %.2f, Theo: %d, Type: " + tt, self.ask_id, new_ask_volume, new_ask_price, self.imbal, self.theo_F)
                        self.asks.add(self.ask_id)
                        self.ask_qtime = time.time()
                        self.ask_volume = new_ask_volume
                        self.trade_type = tt
                
                # Else if only 1 order
                elif self.bid_id == 0 and self.ask_id != 0:
                    # If didn't wait long enough:
                    if (time.time() - self.bid_etime) > WAIT_TIME:
                        self.send_cancel_order(self.ask_id)
                        self.logger.info("Ask %d Canceled TradeType: " + self.trade_type, self.ask_id)
                        self.ask_id = 0
                        # Check if new_price is not same as current price and position limit
                        if new_bid_price not in (self.bid_price, 0):
                            # Send orders
                            self.bid_id = next(self.order_ids)
                            self.bid_price = new_bid_price
                            self.send_insert_order(self.bid_id, Side.BUY, new_bid_price, new_bid_volume, Lifespan.GOOD_FOR_DAY)
                            self.logger.info("Order %d: Bid %d @ %d, Imbal: %.2f, Theo: %d, Type: " + tt, self.bid_id, new_bid_volume, new_bid_price, self.imbal, self.theo_F)
                            self.bids.add(self.bid_id)
                            self.bid_qtime = time.time()
                            self.bid_volume = new_bid_volume
                            self.trade_type = tt
                            
                        if new_ask_price not in (self.ask_price, 0):
                            self.ask_id = next(self.order_ids)
                            self.ask_price = new_ask_price
                            self.send_insert_order(self.ask_id, Side.SELL, new_ask_price, new_ask_volume, Lifespan.GOOD_FOR_DAY)
                            self.logger.info("Order %d: Ask %d @ %d, Imbal: %.2f, Theo: %d, Type: " + tt, self.ask_id, new_ask_volume, new_ask_price, self.imbal, self.theo_F)
                            self.asks.add(self.ask_id)
                            self.ask_qtime = time.time()
                            self.ask_volume = new_ask_volume
                            self.trade_type = tt
                    # Else
                    else:
                        pass
                    
                elif self.bid_id != 0 and self.ask_id == 0:
                    # If didn't wait long enough:
                    if (time.time() - self.ask_etime) > WAIT_TIME:
                        self.send_cancel_order(self.bid_id)
                        self.logger.info("Bid %d Canceled TradeType: " + self.trade_type, self.bid_id)
                        self.bid_id = 0
                        # Check if new_price is not same as current price and position limit
                        if new_bid_price not in (self.bid_price, 0):
                            # Send orders
                            self.bid_id = next(self.order_ids)
                            self.bid_price = new_bid_price
                            self.send_insert_order(self.bid_id, Side.BUY, new_bid_price, new_bid_volume, Lifespan.GOOD_FOR_DAY)
                            self.logger.info("Order %d: Bid %d @ %d, Imbal: %.2f, Theo: %d, Type: " + tt, self.bid_id, new_bid_volume, new_bid_price, self.imbal, self.theo_F)
                            self.bids.add(self.bid_id)
                            self.bid_qtime = time.time()
                            self.bid_volume = new_bid_volume
                            self.trade_type = tt
                            
                        if new_ask_price not in (self.ask_price, 0):
                            self.ask_id = next(self.order_ids)
                            self.ask_price = new_ask_price
                            self.send_insert_order(self.ask_id, Side.SELL, new_ask_price, new_ask_volume, Lifespan.GOOD_FOR_DAY)
                            self.logger.info("Order %d: Ask %d @ %d, Imbal: %.2f, Theo: %d, Type: " + tt, self.ask_id, new_ask_volume, new_ask_price, self.imbal, self.theo_F)
                            self.asks.add(self.ask_id)
                            self.ask_qtime = time.time()
                            self.ask_volume = new_ask_volume
                            self.trade_type = tt
                    # Else pass
                    else:
                        pass
                    
                # If there are 2 orders
                elif self.bid_id != 0 and self.ask_id != 0:
                    # If over update threshold
                    if (time.time() - min(self.ask_qtime, self.bid_qtime)) > UPDATE_TIME:
                        # Cancel all orders
                        self.send_cancel_order(self.bid_id)
                        self.send_cancel_order(self.ask_id)
                        self.logger.info("Both %d, %d Canceled TradeType: " + self.trade_type, self.bid_id, self.ask_id)
                        self.bid_id = 0
                        self.ask_id = 0
                        # Check if new_price is not same as current price and position limit
                        if new_bid_price not in (self.bid_price, 0):
                            # Send orders
                            self.bid_id = next(self.order_ids)
                            self.bid_price = new_bid_price
                            self.send_insert_order(self.bid_id, Side.BUY, new_bid_price, new_bid_volume, Lifespan.GOOD_FOR_DAY)
                            self.logger.info("Order %d: Bid %d @ %d, Imbal: %.2f, Theo: %d, Type: " + tt, self.bid_id, new_bid_volume, new_bid_price, self.imbal, self.theo_F)
                            self.bids.add(self.bid_id)
                            self.bid_qtime = time.time()
                            self.bid_volume = new_bid_volume
                            self.trade_type = tt
                            
                        if new_ask_price not in (self.ask_price, 0):
                            self.ask_id = next(self.order_ids)
                            self.ask_price = new_ask_price
                            self.send_insert_order(self.ask_id, Side.SELL, new_ask_price, new_ask_volume, Lifespan.GOOD_FOR_DAY)
                            self.logger.info("Order %d: Ask %d @ %d, Imbal: %.2f, Theo: %d, Type: " + tt, self.ask_id, new_ask_volume, new_ask_price, self.imbal, self.theo_F)
                            self.asks.add(self.ask_id)
                            self.ask_qtime = time.time()
                            self.ask_volume = new_ask_volume
                            self.trade_type = tt
                    else:
                        pass
                        

                    
    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received order filled for order %d with price %d and volume %d TradeType: " + self.trade_type, client_order_id,
                         price, volume)
        # Hedge
        if client_order_id in self.bids:
            self.position += volume
            self.send_hedge_order(next(self.order_ids), Side.ASK, MIN_BID_NEAREST_TICK, volume)
        elif client_order_id in self.asks:
            self.position -= volume
            self.send_hedge_order(next(self.order_ids), Side.BID, MAX_ASK_NEAREST_TICK, volume)
            
    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int,
                                fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """
        self.logger.info("received order status for order %d with fill volume %d remaining %d and fees %d",
                         client_order_id, fill_volume, remaining_volume, fees)
        
        if remaining_volume == 0:
            if client_order_id == self.bid_id:
                self.bid_id = 0
                self.bid_etime = time.time()
                self.logger.info("Bid Execution Time: %.4f TradeType: " + self.trade_type, self.bid_etime - self.bid_qtime)
            
            elif client_order_id == self.ask_id:
                self.ask_id = 0
                self.ask_etime = time.time()
                self.logger.info("Ask Execution Time: %.4f TradeType: " + self.trade_type, self.ask_etime - self.ask_qtime)
            # It could be either a bid or an ask
            self.bids.discard(client_order_id)
            self.asks.discard(client_order_id)
            
    def on_trade_ticks_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                               ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically when there is trading activity on the market.

        The five best ask (i.e. sell) and bid (i.e. buy) prices at which there
        has been trading activity are reported along with the aggregated volume
        traded at each of those price levels.

        If there are less than five prices on a side, then zeros will appear at
        the end of both the prices and volumes arrays.
        """
        self.logger.info("received trade ticks for instrument %d with sequence number %d", instrument,
                         sequence_number)