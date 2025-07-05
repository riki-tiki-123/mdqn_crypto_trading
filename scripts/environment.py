import numpy as np
import pandas as pd
import csv

def compute_rsi(prices, window=14):
    if len(prices) < window + 1:
        return 0.5
    deltas = np.diff(prices[-(window + 1):])
    ups = deltas[deltas > 0].sum() / window
    downs = -deltas[deltas < 0].sum() / window
    rs = ups / (downs + 1e-6)
    return 1 - 1 / (1 + rs)

def compute_ema(prices, window=14):
    if len(prices) < window:
        return prices[-1]
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    return np.convolve(prices[-window:], weights, mode='valid')[0]

def compute_macd(prices, slow=26, fast=12):
    if len(prices) < slow:
        return 0
    ema_fast = pd.Series(prices).ewm(span=fast).mean().values[-1]
    ema_slow = pd.Series(prices).ewm(span=slow).mean().values[-1]
    return ema_fast - ema_slow

def compute_bollinger(prices, window=20, num_std=2):
    if len(prices) < window:
        return 0, 0
    series = pd.Series(prices[-window:])
    middle = series.mean()
    std = series.std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return (upper - prices[-1]) / prices[-1], (prices[-1] - lower) / prices[-1]

class BitcoinTradingEnv:
    def __init__(
        self, 
        data_path, 
        data_indices=None, 
        window_size=10, 
        fee_rate=0.001, 
        slippage=0.0005,
        max_daily_trades=10,
        inactivity_penalty=0.1,
        hold_time_penalty=0.0,
        bust_threshold=0.10,
    ):
        df = pd.read_csv(data_path)
        self.prices = df['Close'].values
        self.volumes = df['Volume'].values
        if data_indices is not None:
            self.prices = self.prices[data_indices]
            self.volumes = self.volumes[data_indices]
        self.window_size = window_size
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.max_daily_trades = max_daily_trades
        self.inactivity_penalty = inactivity_penalty
        self.hold_time_penalty = hold_time_penalty
        self.bust_threshold = bust_threshold  # Fraction of initial_cash, episode ends below this
        self.trade_history = []

        self.initial_cash = 600  # Start cash

        self.reset()

    def reset(self):
        self.timestep = self.window_size
        self.last_buy_price, self.hold_time, self.completed_trades = 0, 0, 0
        self.total_profit, self.trades_today, self.current_day = 0, 0, self.timestep // 24
        self.trade_history.clear()
        self.inactivity_steps = 0
        self.balance = self.initial_cash
        self.coins = 0
        return self._get_state()

    def _get_state(self):
        window = self.prices[self.timestep - self.window_size:self.timestep]
        volumes = self.volumes[self.timestep - self.window_size:self.timestep]
        price_now = window[-1]
        price_mean = np.mean(window)
        price_std = np.std(window) + 1e-6

        # Normalized price and volume
        norm_window = (window - price_mean) / price_std
        norm_volumes = (volumes - np.mean(volumes)) / (np.std(volumes) + 1e-6)
        # Technical indicators
        sma = price_mean / price_now
        ema = compute_ema(window, window=8) / price_now
        macd = compute_macd(window) / price_now
        rsi = compute_rsi(window)
        boll_hi, boll_lo = compute_bollinger(window)
        momentum = (price_now - window[0]) / window[0]
        pct_change = (price_now - window[-2]) / window[-2]
        vol_change = (volumes[-1] - volumes[-2]) / (volumes[-2] + 1e-6)
        volatility = price_std / price_now
        position_flag = 1 if self.coins > 0 else 0
        normalized_buy = self.last_buy_price / price_now if self.coins > 0 else 0
        hold_time_norm = self.hold_time / self.window_size if self.coins > 0 else 0
        time_of_day = np.sin(2 * np.pi * ((self.timestep % 24) / 24))

        state = np.concatenate([
            norm_window, norm_volumes, [
                sma, ema, macd, rsi, boll_hi, boll_lo, 
                momentum, pct_change, vol_change, volatility,
                position_flag, normalized_buy, hold_time_norm, time_of_day
            ]
        ])
        return state.astype(np.float32)

    def step(self, action):
        reward, done = 0, False
        price, timestamp = self.prices[self.timestep], self.timestep
        day = self.timestep // 24
        if day != self.current_day:
            self.trades_today, self.current_day = 0, day

        self.inactivity_steps += 1

        # Actions: 0=Buy, 1=Hold, 2=Sell
        if action == 0:  # BUY (all-in, no loans, only if flat)
            if self.coins == 0 and self.trades_today < self.max_daily_trades and self.balance > 0:
                buy_price = price * (1 + self.slippage)
                coins_bought = self.balance / (buy_price * (1 + self.fee_rate))
                total_cost = coins_bought * buy_price * (1 + self.fee_rate)
                self.coins = coins_bought
                self.balance -= total_cost
                self.last_buy_price = buy_price
                self.hold_time = 0
                self.trade_history.append(['buy', timestamp, buy_price, 0])
                reward = 0  # No reward on open
                self.trades_today += 1
                self.inactivity_steps = 0
            else:
                reward = -2  # Heavy penalty for invalid buy

        elif action == 2:  # SELL (all-in, only if holding)
            if self.coins > 0:
                sell_price = price * (1 - self.slippage)
                proceeds = self.coins * sell_price * (1 - self.fee_rate)
                cost_basis = self.coins * self.last_buy_price * (1 + self.fee_rate)
                profit = proceeds - cost_basis
                reward = profit  # Reward is *actual net profit/loss* in USD
                self.total_profit += profit
                self.completed_trades += 1
                self.trade_history.append(['sell', timestamp, sell_price, profit])
                self.balance += proceeds
                self.coins = 0
                self.last_buy_price = 0
                self.hold_time = 0
                self.inactivity_steps = 0
                self.trades_today += 1
            else:
                reward = -2  # Heavy penalty for invalid sell

        else:  # HOLD
            if self.coins > 0:
                self.hold_time += 1
                # Only penalize if hold time is too long
                if self.hold_time > 48:
                    reward -= self.hold_time_penalty * (self.hold_time - 48)

        # Penalty for inactivity
        if self.inactivity_steps > 72:
            reward -= self.inactivity_penalty

        # Penalty for too many trades in a day
        if self.trades_today > self.max_daily_trades:
            reward -= 5

        # Bankruptcy check: cash+coin < X% of starting
        net_worth = self.balance + self.coins * price
        if net_worth < self.initial_cash * self.bust_threshold:
            reward -= 20
            done = True

        self.timestep += 1
        if self.timestep >= len(self.prices) - 1:
            done = True

        next_state = self._get_state()
        return next_state, reward, done

    def save_trade_history(self, filename='trade_history.csv'):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Action', 'Timestamp', 'Price', 'Profit'])
            writer.writerows(self.trade_history)
