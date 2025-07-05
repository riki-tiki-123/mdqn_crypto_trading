import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt
from environment import BitcoinTradingEnv
from dqn_model import TradeDQN

def train_dqn(
    data_path, 
    train_indices, 
    window_size=10, 
    input_dim=None, 
    batch_size=64, 
    gamma=0.98, 
    epsilon_start=1.0, 
    epsilon_decay=0.995,
    epsilon_min=0.01, 
    lr=0.0003, 
    episodes=1000, 
    replay_memory_size=5000, 
    target_update_freq=500,
    model_save_path='trade_dqn_final.pth',
    plot_path=None,
    return_trade_history=True,
    dueling=True, 
    use_batchnorm=True, 
    dropout=0.2,
    init_model_path=None   # <--- NEW: option to load weights
):
    # ----------- CUDA SETUP -----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(f"\nStarting DQN training for {episodes} episodes...\n")
    # -----------------------------------

    if input_dim is None:
        input_dim = window_size * 2 + 14  # adjust if your feature count changes

    env = BitcoinTradingEnv(data_path, data_indices=train_indices, window_size=window_size)
    model = TradeDQN(input_dim, dueling=dueling, use_batchnorm=use_batchnorm, dropout=dropout).to(device)
    if init_model_path is not None:
        model.load_state_dict(torch.load(init_model_path, map_location=device))
        print(f"Loaded model weights from {init_model_path}")

    target_model = TradeDQN(input_dim, dueling=dueling, use_batchnorm=use_batchnorm, dropout=dropout).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    memory = deque(maxlen=replay_memory_size)
    profits = []
    balances = []
    equity_curves = []
    epsilon = epsilon_start
    trade_history = []
    global_step = 0

    for episode in range(episodes):
        state_np = np.array(env.reset(), dtype=np.float32)
        state = torch.from_numpy(state_np).unsqueeze(0).to(device)
        total_reward = 0
        done = False

        while not done:
            global_step += 1

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                model.eval()
                with torch.no_grad():
                    action = torch.argmax(model(state)).item()
                model.train()

            next_state_np, reward, done = env.step(action)
            next_state = torch.from_numpy(np.array(next_state_np, dtype=np.float32)).unsqueeze(0).to(device)
            memory.append((state, action, reward, next_state, done))
            total_reward += reward
            state = next_state

            # Double DQN update
            if len(memory) >= batch_size:
                minibatch = random.sample(memory, batch_size)
                batch_states = torch.cat([x[0] for x in minibatch]).to(device)
                batch_actions = torch.tensor([x[1] for x in minibatch], dtype=torch.long, device=device)
                batch_rewards = torch.tensor([x[2] for x in minibatch], dtype=torch.float32, device=device)
                batch_next_states = torch.cat([x[3] for x in minibatch]).to(device)
                batch_dones = torch.tensor([x[4] for x in minibatch], dtype=torch.float32, device=device)

                with torch.no_grad():
                    # Use main model for action selection, target for evaluation
                    next_actions = model(batch_next_states).argmax(dim=1, keepdim=True)
                    next_q_target = target_model(batch_next_states).gather(1, next_actions).squeeze()
                    targets = batch_rewards + (gamma * next_q_target * (1 - batch_dones))

                q_values = model(batch_states)
                action_q = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze()
                loss = F.mse_loss(action_q, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network by steps (not episode)
            if global_step % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())

        # Final metrics for this episode
        profits.append(env.total_profit)
        balances.append(env.balance)
        # Compute final account value = cash + current coin value
        final_price = env.prices[env.timestep-1] if env.timestep > 0 else env.prices[0]
        account_equity = env.balance + env.coins * final_price

        # Epsilon decay (linear or exp)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            epsilon = max(epsilon, epsilon_min)

        # Save trade history for this episode
        for t in env.trade_history:
            trade_history.append({
                'action': t[0],
                'timestamp': int(t[1]),
                'price': float(t[2]),
                'profit': float(t[3]) if len(t) > 3 else 0.0
            })

        if (episode+1) % 10 == 0:
            print(f"Ep:{episode+1}, Trades:{env.completed_trades}, Profit:${env.total_profit:.2f}, "
                  f"Reward:{total_reward:.2f}, Eps:{epsilon:.3f}, "
                  f"Balance:${env.balance:.2f}, Coins:{env.coins:.6f}, "
                  f"AccountValue:${account_equity:.2f}")
            env.save_trade_history(f'trades_ep_{episode+1}.csv')

    # Plot and save profit per episode if requested
    if plot_path is not None:
        plt.figure(figsize=(12,6))
        plt.plot(profits, label='Episode Gross Profit')
        plt.plot(balances, label='Ending Cash')
        plt.plot(equity_curves, label='Account Value (Equity)')
        plt.xlabel('Episode')
        plt.ylabel('Dollars')
        plt.title('Profit and Account Value per Episode')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
    else:
        plt.plot(profits, label='Gross Profit')
        plt.plot(balances, label='Ending Cash')
        plt.plot(equity_curves, label='Account Value')
        plt.xlabel('Episode')
        plt.ylabel('Dollars')
        plt.title('Profit and Account Value per Episode')
        plt.grid(True)
        plt.legend()
        plt.show()

    torch.save(model.state_dict(), model_save_path)
    print("Training completed and model saved!")

    final_profit = profits[-1] if len(profits) > 0 else 0
    num_trades = sum(1 for t in trade_history if t['action'] == 'sell')

    if return_trade_history:
        return final_profit, num_trades, trade_history
    else:
        return final_profit

# ---- Run training on previous 2 years of data using previous weights ----

if __name__ == "__main__":
    data_path = "BTCUSDT_1h_ohlcv_2y_prior.csv"           # <--- Use previous 2y dataset
    model_save_path = "models_1h/dqn_fold2.pth"
    prev_weights = "models_1h/dqn_fold1.pth"              # <--- Use previous model weights

    # Adapt as needed for your data length
    train_indices = list(range(0, 16000))                 # Use up to 16000 if your data has at least that many

    train_dqn(
        data_path=data_path,
        train_indices=train_indices,
        window_size=10,
        episodes=1000,
        model_save_path=model_save_path,
        plot_path="models_1h/profit_fold2.png",
        return_trade_history=True,
        dueling=True,
        use_batchnorm=True,
        dropout=0.2,
        init_model_path=prev_weights      # <--- Loads previous weights!
    )
