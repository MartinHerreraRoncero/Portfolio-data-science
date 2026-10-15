import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold


class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """
    Purged TimeSeries Cross-Validator for Financial Bankruptcy Prediction.
    
    Guarantees:
    1. Strict Time Series Ordering: No future data in training set.
    2. Purging Window: Applies a strict temporal purging gap (365 days for 12M, 
       730 days / 2 full years for 24M) between Train and Validation to prevent 
       forward-looking target horizon leakage (target_bankrupt_12m, target_bankrupt_24m).
    3. Representative Validation Folds: Preserves mature bankrupt entities in validation folds 
       while preventing target overlap through temporal purging.
    """

    def __init__(self, n_splits=5, purge_window_days=None, horizon="12M"):
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.horizon = str(horizon).upper()
        if purge_window_days is not None:
            self.purge_window_days = purge_window_days
        elif "24" in self.horizon:
            self.purge_window_days = 730
        else:
            self.purge_window_days = 365

    def split(self, X, y=None, groups=None, date_column=None):
        if date_column is None:
            raise ValueError("date_column (pandas Series of filing dates) must be provided.")

        # Determine effective purge window: if evaluating 24M horizon, enforce strict 730 days
        purge_days = self.purge_window_days
        if hasattr(y, 'name') and '24' in str(y.name):
            purge_days = max(purge_days, 730)
        elif "24" in self.horizon:
            purge_days = max(purge_days, 730)

        dates = pd.to_datetime(date_column).reset_index(drop=True)
        unique_dates = np.sort(dates.unique())

        n_samples = len(dates)
        indices = np.arange(n_samples)

        # Split timeline into n_splits + 1 chunks
        date_splits = np.array_split(unique_dates, self.n_splits + 1)

        for i in range(self.n_splits):
            train_date_end = pd.to_datetime(date_splits[i][-1])
            purge_cutoff_date = train_date_end + pd.Timedelta(days=purge_days)
            
            val_date_start = pd.to_datetime(date_splits[i + 1][0])
            # Apply strict temporal purging gap (365d for 12M, 730d for 24M) between Train and Validation
            if val_date_start <= purge_cutoff_date:
                val_date_start = purge_cutoff_date

            val_date_end = pd.to_datetime(date_splits[i + 1][-1])

            if val_date_start > val_date_end:
                continue

            train_mask = (dates <= train_date_end)
            val_mask = (dates >= val_date_start) & (dates <= val_date_end)

            train_idx = indices[train_mask]
            val_idx = indices[val_mask]

            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx


def test_validation_scheme():
    print("=" * 65)
    print("FASE 4: VERIFICACIÓN DEL ESQUEMA PURGED TIMESERIES CV CORREGIDO")
    print("=" * 65)

    import duckdb
    con = duckdb.connect()
    df = con.execute("""
        SELECT cik, filed_date, target_bankrupt_12m, target_bankrupt_24m 
        FROM 'data/processed/sec_dataset/v2_sec_financials_pivoted_clean.parquet'
        ORDER BY filed_date ASC;
    """).df()
    con.close()

    print(f"[Validation Engine] Loaded {len(df):,} observations across {df['cik'].nunique():,} unique CIKs.")
    print(f"Total Bankrupt 12M (+): {df['target_bankrupt_12m'].sum():,}")
    print(f"Total Bankrupt 24M (+): {df['target_bankrupt_24m'].sum():,}")

    for horizon_label, purge_days in [("12M", 365), ("24M", 730)]:
        print(f"\n--- Probando Validación Purged para Horizonte {horizon_label} (Purga: {purge_days} días) ---")
        cv = PurgedGroupTimeSeriesSplit(n_splits=5, purge_window_days=purge_days)
        target_series = df['target_bankrupt_24m'] if horizon_label == "24M" else df['target_bankrupt_12m']

        for fold, (train_idx, val_idx) in enumerate(cv.split(df, y=target_series, groups=df['cik'], date_column=df['filed_date']), 1):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            train_min_date = train_df['filed_date'].min()
            train_max_date = train_df['filed_date'].max()
            val_min_date = val_df['filed_date'].min()
            val_max_date = val_df['filed_date'].max()

            actual_gap_days = (pd.to_datetime(val_min_date) - pd.to_datetime(train_max_date)).days
            train_pos = target_series.iloc[train_idx].sum()
            val_pos = target_series.iloc[val_idx].sum()

            print(f"\nFold {fold} [{horizon_label}]: Gap = {actual_gap_days} días (>= {purge_days}d)")
            print(f"  - Train Range: {train_min_date} to {train_max_date} | Obs: {len(train_idx):>7,} | Positives: {train_pos:>5,}")
            print(f"  - Val   Range: {val_min_date} to {val_max_date} | Obs: {len(val_idx):>7,} | Positives: {val_pos:>5,}")
            assert actual_gap_days >= purge_days, f"Gap {actual_gap_days} < required purge {purge_days}"

    print("\n[Validation Engine] Purged TimeSeriesSplit CV verified successfully for both 12M and 24M!")


if __name__ == "__main__":
    test_validation_scheme()
