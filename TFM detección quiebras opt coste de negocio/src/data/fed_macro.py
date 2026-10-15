import os
import requests
import pandas as pd
import numpy as np

FED_API_KEY = os.getenv("FED_API_KEY", "3db6890dafdd5256345433bc73dfebe1")
START_DATE = "1999-01-01"
END_DATE = "2018-12-31"

SERIES = {
    'DFF': 'interest_rate',        # Daily
    'T10Y2Y': 'yield_curve',       # Daily
    'GDPC1': 'gdp',                # Quarterly
    'PAYEMS': 'employment',        # Monthly
    'CPIAUCSL': 'inflation'        # Monthly
}

def fetch_fred_series(series_id, series_name):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FED_API_KEY}&file_type=json&observation_start={START_DATE}&observation_end={END_DATE}"
    res = requests.get(url)
    if res.status_code == 200:
        data = res.json().get('observations', [])
        df = pd.DataFrame(data)
        if df.empty:
            return None
        df['date'] = pd.to_datetime(df['date'])
        
        # Handle '.' values which mean NaN in FRED
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['value'])
        df['year'] = df['date'].dt.year
        
        # Calculate annual mean and volatility
        annual_stats = df.groupby('year')['value'].agg(
            mean='mean',
            volatility='std'
        ).reset_index()
        
        # Rename columns
        annual_stats = annual_stats.rename(columns={
            'mean': f'{series_name}_mean',
            'volatility': f'{series_name}_volatility'
        })
        
        return annual_stats
    else:
        print(f"Error fetching {series_id}: {res.status_code} - {res.text}")
        return None

def get_fed_macro_data():
    dfs = []
    for s_id, s_name in SERIES.items():
        print(f"Fetching {s_id} ({s_name}) from FRED...")
        df_stat = fetch_fred_series(s_id, s_name)
        if df_stat is not None:
            dfs.append(df_stat)
            
    # Merge all
    if not dfs:
        return pd.DataFrame()
        
    final_df = dfs[0]
    for i in range(1, len(dfs)):
        final_df = pd.merge(final_df, dfs[i], on='year', how='outer')
        
    # Fill NaN volatility with 0
    final_df = final_df.fillna(0)
    return final_df

def merge_fed_macro_data(df):
    """Combines dataset with FRED macroeconomic variables"""
    fed_df = get_fed_macro_data()
    if not fed_df.empty:
        df_merged = pd.merge(df, fed_df, on='year', how='left')
        return df_merged
    return df

if __name__ == "__main__":
    df_macro = get_fed_macro_data()
    print("=== FRED MACROECONOMIC DATA (1999-2018) ===")
    print(df_macro.head())
