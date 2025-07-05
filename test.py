import torch
import numpy as np
from environment import BitcoinTradingEnv
from dqn_model import TradeDQN

def test_dqn(
    data_path, 
    test_indices, 
    window_size=10, 
    input_dim=None, 
    model_path=None, 
    dueling=True, 
    use_batchnorm=True, 
    dropout=0.2,
    return_trade_history=True,
    verbose=False
):
    # --- CUDA SUPPORT ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Testing on device:", device)
    # --------------------

    if input_dim is None:
        # Should match the input dim from your env
        input_dim = window_size * 2 + 14

    env = BitcoinTradingEnv(data_path, data_indices=test_indices, window_size=window_size)
    model = TradeDQN(input_dim, dueling=dueling, use_batchnorm=use_batchnorm, dropout=dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    state_np = np.array(env.reset(), dtype=np.float32)
    state = torch.from_numpy(state_np).unsqueeze(0).to(device)
    total_reward = 0
    done = False
    trade_history = []

    while not done:
        # Fully greedy policy (no exploration in test)
        with torch.no_grad():
            q_values = model(state)
            action = torch.argmax(q_values).item()

        next_state_np, reward, done = env.step(action)
        next_state = torch.from_numpy(np.array(next_state_np, dtype=np.float32)).unsqueeze(0).to(device)
        total_reward += reward
        state = next_state

        if verbose and len(env.trade_history) > 0 and env.trade_history[-1][0] in ['buy', 'sell']:
            print(f"Step: {env.timestep}, Action: {env.trade_history[-1][0]}, Price: {env.trade_history[-1][2]:.2f}, Reward: {reward:.2f}")

    # Build trade history dictionary for compatibility
    for t in env.trade_history:
        trade_history.append({
            'action': t[0],
            'timestamp': int(t[1]),
            'price': float(t[2]),
            'profit': float(t[3]) if len(t) > 3 else 0.0
        })

    final_profit = env.total_profit
    num_trades = sum(1 for t in trade_history if t['action'] == 'sell')

    if return_trade_history:
        return final_profit, num_trades, trade_history
    else:
        return final_profit
