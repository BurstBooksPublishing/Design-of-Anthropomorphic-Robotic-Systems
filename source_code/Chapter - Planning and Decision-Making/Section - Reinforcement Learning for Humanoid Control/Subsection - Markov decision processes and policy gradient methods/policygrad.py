import torch
import torch.nn.functional as F

def compute_gae_advantages(rewards, values, next_values, dones, gamma, lam):
    """
    Compute Generalized Advantage Estimation (GAE) advantages.
    """
    # Compute TD errors
    deltas = rewards + gamma * next_values * (1 - dones) - values
    
    # Compute GAE advantages in reverse order
    gae = 0
    advantages = torch.zeros_like(deltas)
    for t in reversed(range(len(deltas))):
        gae = deltas[t] + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    
    return advantages

# Convert lists to tensors
states_tensor = torch.stack(states)
actions_tensor = torch.stack(actions)
rewards_tensor = torch.stack(rewards)
next_states_tensor = torch.stack(next_states)
dones_tensor = torch.stack(dones)

# Compute value predictions
values = critic(states_tensor)                    # V_w(s_t)
next_values = critic(next_states_tensor)         # V_w(s_{t+1})

# Compute GAE advantages
advantages = compute_gae_advantages(
    rewards_tensor, values, next_values, dones_tensor, gamma, lam
).detach()

# Compute value targets for critic loss
returns = advantages + values.detach()

# Policy loss (maximize expected advantage)
log_probs = actor.get_log_prob(actions_tensor, states_tensor)
policy_loss = -(log_probs * advantages).mean()

# Critic loss (MSE between predicted values and returns)
value_loss = F.mse_loss(values, returns)

# Optimize actor
optimizer_actor.zero_grad()
policy_loss.backward()
optimizer_actor.step()

# Optimize critic
optimizer_critic.zero_grad()
value_loss.backward()
optimizer_critic.step()