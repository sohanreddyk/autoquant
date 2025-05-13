import pandas as pd
from finrl.env.env_stock_trading import StockTradingEnv
from agents.ppo_agent import train_agent, evaluate_agent

# Load and preprocess data
df = pd.read_csv("data/stock_data.csv")
df = df.reset_index()
df.columns = [col.replace(" ", "_") for col in df.columns]  # Remove any spaces

# Subset and reformat for one stock (e.g., AAPL)
aapl_df = df[df['Ticker'] == 'AAPL'].copy()
aapl_df = aapl_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
aapl_df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
aapl_df['tic'] = 'AAPL'
aapl_df['day'] = pd.to_datetime(aapl_df['date']).dt.dayofyear
aapl_df['date'] = pd.to_datetime(aapl_df['date'])
aapl_df = aapl_df.sort_values('date')

# Define environment
stock_dim = 1
env_kwargs = {
    "df": aapl_df,
    "stock_dim": stock_dim,
    "hmax": 100,
    "initial_amount": 100000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "reward_scaling": 1e-4,
}
env = StockTradingEnv(**env_kwargs)

# Train the PPO agent
model = train_agent(env, timesteps=10000)

# Evaluate the agent
final_reward = evaluate_agent(model, env)
print(f"🏁 Final Total Reward: {final_reward}")
