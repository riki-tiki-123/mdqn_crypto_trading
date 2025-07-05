import numpy as np

def calculate_sharpe(trade_history, risk_free_rate=0.0):
    """
    Calculates Sharpe Ratio from trade history.
    Uses profit/loss from each trade (sells only).
    """
    # Extract profits from sell trades
    returns = [t['profit'] for t in trade_history if t['action'] == 'sell']
    if len(returns) < 2:
        return 0.0  # Not enough trades to measure risk

    returns = np.array(returns)
    # Excess returns (subtract risk-free rate per trade, if desired)
    excess_returns = returns - risk_free_rate

    avg_return = np.mean(excess_returns)
    std_return = np.std(excess_returns)
    if std_return < 1e-6:
        return 0.0
    sharpe = avg_return / std_return
    return sharpe

def calculate_winrate(trade_history):
    """
    Calculates the winrate: % of sell trades that are profitable.
    """
    sells = [t for t in trade_history if t['action'] == 'sell']
    if len(sells) == 0:
        return 0.0
    wins = sum(1 for t in sells if t['profit'] > 0)
    return wins / len(sells)

def count_trades(trade_history, action_type='sell'):
    """
    Count the number of trades of a certain type ('buy' or 'sell').
    """
    return sum(1 for t in trade_history if t['action'] == action_type)

def set_random_seed(seed=42):
    """
    Sets random seed for reproducibility.
    """
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_trades_vs_price(price_series, trade_history, plot_path=None):
    """
    Optional: plot price with buy/sell markers from trade history.
    """
    import matplotlib.pyplot as plt

    buys = [t for t in trade_history if t['action'] == 'buy']
    sells = [t for t in trade_history if t['action'] == 'sell']
    buy_x = [t['timestamp'] for t in buys]
    buy_y = [t['price'] for t in buys]
    sell_x = [t['timestamp'] for t in sells]
    sell_y = [t['price'] for t in sells]

    plt.figure(figsize=(15, 6))
    plt.plot(price_series, label='Price', color='blue')
    plt.scatter(buy_x, buy_y, marker='^', color='green', label='Buy', s=60)
    plt.scatter(sell_x, sell_y, marker='v', color='red', label='Sell', s=60)
    plt.legend()
    plt.xlabel('Time step')
    plt.ylabel('Price')
    plt.title('Trading Actions vs. Price')
    plt.grid(True)
    if plot_path:
        plt.savefig(plot_path)
        plt.close()
    else:
        plt.show()
