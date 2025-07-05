import numpy as np
import pandas as pd
import os
import torch
from train import train_dqn
from test import test_dqn
from utils import calculate_sharpe, calculate_winrate, set_random_seed

# ==== User config ====
DATA_PATH = 'data/BTCUSDT_1h_ohlcv_2y.csv'
WINDOW_SIZE = 10
TRAIN_SIZE = 16000
TEST_SIZE = 1000
MAX_EPISODES = 1000
MODEL_DIR = 'models_1h'
DUELING = True
USE_BATCHNORM = True
DROPOUT = 0.2
SEED = 42
os.makedirs(MODEL_DIR, exist_ok=True)

def get_walk_forward_folds(n_samples, train_size, test_size):
    folds = []
    start_train = 0
    while (start_train + train_size + test_size) <= n_samples:
        train_idx = list(range(start_train, start_train + train_size))
        test_idx = list(range(start_train + train_size, start_train + train_size + test_size))
        folds.append((train_idx, test_idx))
        start_train += test_size  # Slide window forward by test_size for next fold
    return folds

def main():
    # ---- Print configuration summary ----
    print("\n========= CONFIG SUMMARY =========")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Max episodes per fold: {MAX_EPISODES}")
    print("==================================\n")

    set_random_seed(SEED)
    df = pd.read_csv(DATA_PATH)
    n_samples = len(df)
    folds = get_walk_forward_folds(n_samples, TRAIN_SIZE, TEST_SIZE)
    results = []

    print(f"\nTotal samples: {n_samples}, Folds: {len(folds)}")
    for fold_num, (train_idx, test_idx) in enumerate(folds):
        print(f"\n==== Fold {fold_num+1}/{len(folds)} ====")
        print(f"Train: {train_idx[0]}–{train_idx[-1]}, len={len(train_idx)}")
        print(f"Test: {test_idx[0]}–{test_idx[-1]}, len={len(test_idx)}")

        model_path = os.path.join(MODEL_DIR, f"dqn_fold{fold_num+1}.pth")
        plot_path = os.path.join(MODEL_DIR, f"profit_fold{fold_num+1}.png")

        train_profit, train_trades, train_history = train_dqn(
            data_path=DATA_PATH,
            train_indices=train_idx,
            window_size=WINDOW_SIZE,
            episodes=MAX_EPISODES,
            model_save_path=model_path,
            plot_path=plot_path,
            return_trade_history=True,
            dueling=DUELING,
            use_batchnorm=USE_BATCHNORM,
            dropout=DROPOUT
        )

        train_sharpe = calculate_sharpe(train_history)
        train_winrate = calculate_winrate(train_history)

        test_profit, test_trades, test_history = test_dqn(
            data_path=DATA_PATH,
            test_indices=test_idx,
            window_size=WINDOW_SIZE,
            model_path=model_path,
            return_trade_history=True,
            dueling=DUELING,
            use_batchnorm=USE_BATCHNORM,
            dropout=DROPOUT
        )
        test_sharpe = calculate_sharpe(test_history)
        test_winrate = calculate_winrate(test_history)

        print(f"Train Profit: {train_profit:.2f}, Test Profit: {test_profit:.2f}")
        print(f"Train Sharpe: {train_sharpe:.3f}, Test Sharpe: {test_sharpe:.3f}")
        print(f"Train Winrate: {train_winrate:.2%}, Test Winrate: {test_winrate:.2%}")
        print(f"Train Trades: {train_trades}, Test Trades: {test_trades}")

        results.append({
            'fold': fold_num+1,
            'train_profit': train_profit,
            'test_profit': test_profit,
            'train_sharpe': train_sharpe,
            'test_sharpe': test_sharpe,
            'train_winrate': train_winrate,
            'test_winrate': test_winrate,
            'train_trades': train_trades,
            'test_trades': test_trades,
            'train_size': len(train_idx),
            'test_size': len(test_idx)
        })

    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(MODEL_DIR, "walk_forward_results_1h.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n===== ALL FOLD RESULTS =====")
    print(results_df)
    print(f"\nResults saved to {results_csv_path}")

if __name__ == "__main__":
    main()
