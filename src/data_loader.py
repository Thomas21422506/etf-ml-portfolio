"""
Data loading and preprocessing functions
"""

import os
import pandas as pd


def find_data_file():
    """Automatically finds the data file in data/ or current directory."""
    # Try data/ folder first
    if os.path.exists('data'):
        data_files = os.listdir('data')
        csv_files = [f for f in data_files if 'DATA' in f and 'PRICE' in f and 'FINAL' in f and f.endswith('.csv')]
        if csv_files:
            full_path = os.path.join('data', csv_files[0])
            print(f"[INIT] Data file found: {full_path}")
            return full_path
    
    # Fallback to current directory
    files = os.listdir('.')
    csv_files = [f for f in files if 'DATA' in f and 'PRICE' in f and 'FINAL' in f and f.endswith('.csv')]
    
    if csv_files:
        print(f"[INIT] Data file found: {csv_files[0]}")
        return csv_files[0]
    else:
        print("[INIT] No data file matching pattern 'DATA*PRICE*FINAL*.csv' was found.")
        print("[INIT] Available CSV files:", [f for f in files if f.endswith('.csv')])
        return None


def load_data_properly(file_path):
    """Loads raw ETF price data with the correct structure."""
    df = pd.read_csv(
        file_path,
        sep=';',
        skiprows=1,
        header=0,
        dtype=str
    )

    df.columns = [col.strip() for col in df.columns]
    df['Dates'] = pd.to_datetime(df['Dates'], format='%b-%d-%Y')

    numeric_columns = ['QQQ', 'SPY', 'TLT', 'GLD', 'EEM']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].str.replace(',', '.').astype(float)

    df.set_index('Dates', inplace=True)
    return df


def create_weekly_data(daily_prices):
    """Converts daily prices into weekly returns (Friday close)."""
    weekly_prices = daily_prices.resample('W-FRI').last()
    weekly_returns = weekly_prices.pct_change().dropna()
    return weekly_returns