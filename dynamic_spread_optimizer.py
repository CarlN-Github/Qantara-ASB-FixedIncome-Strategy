import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DynamicSpreadOptimizer:
    """
    Institutional-grade backtesting engine for a Dynamic Spread Overlay Strategy.
    Targets spread compression/widening between High-Yield Sovereigns and Supranationals.
    """
    
    def __init__(self, data: pd.DataFrame, sovereign_col: str, supra_col: str, tc_bps: float = 20.0):
        """
        Initializes the backtesting engine.
        
        :param data: DataFrame containing price histories
        :param sovereign_col: Column name for the Sovereign asset
        :param supra_col: Column name for the Supranational asset
        :param tc_bps: Transaction cost in basis points applied per allocation switch
        """
        self.df = data.copy()
        self.sov_col = sovereign_col
        self.sup_col = supra_col
        self.tc = tc_bps / 10000.0  # Convert bps to decimal
        
    def generate_signals(self, rolling_window: int = 60, z_threshold: float = 1.0):
        """
        Generates allocation signals based on spread Z-scores.
        - If spread is unusually tight (Z < -threshold), switch to Supranationals (De-risk).
        - If spread is unusually wide (Z > threshold), switch to Sovereigns (Yield capture).
        - Otherwise, hold the current position.
        """
        # 1. Calculate the Yield Spread Proxy (simplification using inverse prices for demonstration)
        # In a live environment, you would ingest actual Z-spreads or G-spreads here.
        self.df['Spread_Proxy'] = self.df[self.sup_col] / self.df[self.sov_col] 
        
        # 2. Calculate Rolling Z-Score of the spread
        rolling_mean = self.df['Spread_Proxy'].rolling(window=rolling_window).mean()
        rolling_std = self.df['Spread_Proxy'].rolling(window=rolling_window).std()
        self.df['Spread_Z_Score'] = (self.df['Spread_Proxy'] - rolling_mean) / rolling_std
        
        # 3. Generate Trading Signal (1 = 100% Sovereign, 0 = 100% Supra)
        # Default starting allocation is 100% Sovereign (Yield seeking)
        conditions = [
            self.df['Spread_Z_Score'] > z_threshold,   # Spread Wide -> Buy Sovereign
            self.df['Spread_Z_Score'] < -z_threshold   # Spread Tight -> Buy Supra (Defensive)
        ]
        choices = [1.0, 0.0]
        
        self.df['Target_Allocation'] = np.select(conditions, choices, default=np.nan)
        # Forward fill to maintain the position until a new signal is triggered
        self.df['Target_Allocation'] = self.df['Target_Allocation'].ffill().fillna(1.0)
        
    def run_backtest(self):
        """
        Executes the backtest, factoring in transaction costs and preventing look-ahead bias.
        """
        # Calculate daily returns of the underlying assets
        self.df["Sovereign_Return"] = self.df[self.sov_col].pct_change()
        self.df["Supra_Return"] = self.df[self.sup_col].pct_change()
        
        # Shift the allocation signal by 1 day to prevent look-ahead bias
        # We trade at the close based on the signal, earning the next day's return
        self.df["Allocation_Signal"] = self.df["Target_Allocation"].shift(1)
        
        # Calculate gross strategy returns
        self.df["Gross_Strategy_Return"] = (
            self.df["Allocation_Signal"] * self.df["Sovereign_Return"]
        ) + ((1 - self.df["Allocation_Signal"]) * self.df["Supra_Return"])
        
        # Calculate transaction costs based on portfolio turnover
        # Any change in allocation triggers a trade execution penalty
        self.df["Trade_Execution"] = self.df["Allocation_Signal"].diff().abs().fillna(0)
        
        # Net Strategy Return (Gross - Execution Costs)
        self.df["Strategy_Return"] = self.df["Gross_Strategy_Return"] - (self.df["Trade_Execution"] * self.tc)
        
        # Calculate Cumulative Returns for plotting
        self.df['Cum_Sovereign'] = (1 + self.df["Sovereign_Return"].fillna(0)).cumprod()
        self.df['Cum_Supra'] = (1 + self.df["Supra_Return"].fillna(0)).cumprod()
        self.df['Cum_Strategy'] = (1 + self.df["Strategy_Return"].fillna(0)).cumprod()
        
    def plot_results(self):
        """
        Visualizes the performance comparison and the dynamic allocation.
        """
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Top Panel: Cumulative Returns
        ax1.plot(self.df.index, self.df['Cum_Sovereign'], label='Buy & Hold: Sovereigns', color='#d62728', alpha=0.7)
        ax1.plot(self.df.index, self.df['Cum_Supra'], label='Buy & Hold: Supranationals', color='#1f77b4', alpha=0.7)
        ax1.plot(self.df.index, self.df['Cum_Strategy'], label='Dynamic Spread Overlay', color='#2ca02c', linewidth=2.5)
        
        ax1.set_title('Qantara ASB Strategy Pitch: Dynamic Spread Overlay vs Benchmarks', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend(loc='upper left')
        
        # Bottom Panel: Capital Allocation
        ax2.fill_between(self.df.index, 0, self.df['Allocation_Signal'], color='#d62728', alpha=0.3, label='Sovereign Exposure')
        ax2.fill_between(self.df.index, self.df['Allocation_Signal'], 1, color='#1f77b4', alpha=0.3, label='Supranational Exposure')
        ax2.set_ylabel('Allocation')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['100% Supra', '100% Sov'])
        ax2.legend(loc='upper left')
        
        plt.tight_layout()
        plt.show()

    def print_statistics(self):
        """Prints standard institutional performance metrics."""
        stats = {}
        for col, name in zip(['Sovereign_Return', 'Supra_Return', 'Strategy_Return'], 
                             ['Sovereign (B&H)', 'Supranational (B&H)', 'Dynamic Strategy']):
            
            ann_ret = self.df[col].mean() * 252
            ann_vol = self.df[col].std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            
            stats[name] = {
                'Ann. Return': f"{ann_ret:.2%}",
                'Ann. Volatility': f"{ann_vol:.2%}",
                'Sharpe Ratio': f"{sharpe:.2f}"
            }
            
        print("\n=== Institutional Performance Summary (Accounting for 20bps Slippage) ===")
        print(pd.DataFrame(stats).T)
        print("=========================================================================\n")


# ==========================================
# SIMULATION MODULE (For demonstration)
# ==========================================
def generate_mock_market_data(days=1000):
    """Generates realistic synthetic data for Sovereigns and Supranationals."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=days, freq="B")
    
    # Sovereign: Higher yield/drift, higher volatility, occasional severe drawdowns
    sov_returns = np.random.normal(loc=0.0004, scale=0.008, size=days)
    # Inject some emerging market distress (widening spreads)
    sov_returns[200:250] -= 0.005 
    sov_returns[600:630] -= 0.008
    
    # Supranational: Lower yield, highly stable (AAA/AA- proxy)
    supra_returns = np.random.normal(loc=0.00015, scale=0.002, size=days)
    
    df = pd.DataFrame({
        'Sovereign_Index': (1 + sov_returns).cumprod() * 100,
        'Supra_Index': (1 + supra_returns).cumprod() * 100
    }, index=dates)
    
    return df

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Generate realistic market data
    market_data = generate_mock_market_data(days=1000)
    
    # 2. Initialize the Qantara Strategy Engine
    # Factoring in 20 bps transaction costs per switch (Frontier Market Liquidity Penalty)
    engine = DynamicSpreadOptimizer(
        data=market_data, 
        sovereign_col='Sovereign_Index', 
        supra_col='Supra_Index', 
        tc_bps=20.0 
    )
    
    # 3. Generate Signals (Z-score approach) and Backtest
    engine.generate_signals(rolling_window=60, z_threshold=1.2)
    engine.run_backtest()
    
    # 4. Display Results
    engine.print_statistics()
    engine.plot_results()