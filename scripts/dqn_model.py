import torch
import torch.nn as nn
import torch.nn.functional as F

class TradeDQN(nn.Module):
    def __init__(
        self, 
        input_dim, 
        output_dim=3, 
        dueling=True, 
        use_batchnorm=True, 
        dropout=0.2
    ):
        super().__init__()
        self.dueling = dueling
        self.use_batchnorm = use_batchnorm

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256) if use_batchnorm else nn.Identity()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128) if use_batchnorm else nn.Identity()
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64) if use_batchnorm else nn.Identity()

        if dueling:
            # Dueling: value and advantage streams
            self.value = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            self.advantage = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_dim)
            )
        else:
            self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = F.relu(self.bn3(self.fc3(x)))

        if self.dueling:
            value = self.value(x)
            advantage = self.advantage(x)
            qvals = value + advantage - advantage.mean(dim=1, keepdim=True)
            return qvals
        else:
            return self.out(x)

# ----------- USAGE EXAMPLE BELOW ------------

# Set device globally (put this at the top of your training/inference script)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Example instantiation:
# input_dim = 42  # change this to your actual input dimension
# model = TradeDQN(input_dim=input_dim).to(device)

# Example: sending a batch of states to the model on GPU
# state_batch = torch.randn(batch_size, input_dim).to(device)
# q_values = model(state_batch)

# When loading a saved model:
# model.load_state_dict(torch.load('trade_dqn.pth', map_location=device))

# Make sure any input to the model is also on the same device:
# state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
# output = model(state_tensor)

# In your training loop, ensure all tensors you use with the model are .to(device)







