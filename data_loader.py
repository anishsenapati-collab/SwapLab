"""
data_loader.py
ES418 Group 13 — Interest Rate Swap Valuation
Loads, cleans, and prepares the DTB3 rate series for Vasicek calibration.

YOUR CSV STRUCTURE (auto-detected):
  Column 1 : observation_date
  Column 2 : DTB3
  Date range: 2006-03-27 → 2026-03-26
  Missing   : 215 rows (blank, forward-filled)

HOW TO USE:
  1. Place DTB3.csv in the same folder as this file
  2. Run standalone test:  python data_loader.py
  3. Or import into main.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats


# ── LOAD & CLEAN ──────────────────────────────────────────────────────────────

def load_dtb3(filepath='DTB3.csv', cache_path='rates_clean.npy'):
    """
    Load and clean the FRED DTB3 CSV file.

    Handles your specific CSV format:
      - Date column named 'observation_date'
      - Rate column named 'DTB3'
      - Missing values are blank (not '.' like older FRED exports)
      - Date range 2006-03-27 to 2026-03-26

    Parameters
    ----------
    filepath   : str  — path to DTB3.csv
    cache_path : str  — cleaned array cached here for fast future reloads

    Returns
    -------
    rates : np.ndarray        — daily rates in DECIMAL form (0.05 = 5%)
    dates : pd.DatetimeIndex  — corresponding business dates
    """

    # reload cache if it is newer than the CSV
    if os.path.exists(cache_path) and os.path.exists(filepath):
        if os.path.getmtime(cache_path) > os.path.getmtime(filepath):
            print(f"[data_loader] Loading cached rates from '{cache_path}'")
            rates = np.load(cache_path)
            df    = _read_and_clean(filepath)
            return rates, df.index

    print(f"[data_loader] Reading '{filepath}' ...")
    df    = _read_and_clean(filepath)
    rates = df['rate'].values / 100.0   # percent → decimal

    np.save(cache_path, rates)
    print(f"[data_loader] Saved clean cache → '{cache_path}'")

    return rates, df.index


def _read_and_clean(filepath):
    """Read CSV, rename columns, forward-fill missing, return clean DataFrame."""

    df = pd.read_csv(
        filepath,
        parse_dates=['observation_date'],
        index_col='observation_date'
    )

    # rename DTB3 column to 'rate' for consistency
    df.columns = ['rate']

    # blank cells come in as NaN already — no '.' replacement needed
    df['rate'] = pd.to_numeric(df['rate'], errors='coerce')

    n_missing_before = df['rate'].isna().sum()

    # forward-fill weekends / public holidays (max 3 consecutive days)
    df['rate'] = df['rate'].ffill(limit=3)
    df = df.dropna()

    n_filled = n_missing_before - df['rate'].isna().sum()

    print(f"[data_loader] Rows (after clean) : {len(df):,}")
    print(f"[data_loader] Missing filled      : {n_filled}")
    print(f"[data_loader] Date range          : "
          f"{df.index[0].date()} → {df.index[-1].date()}")
    print(f"[data_loader] Rate range          : "
          f"{df['rate'].min():.4f}% → {df['rate'].max():.4f}%")
    print(f"[data_loader] Mean rate           : {df['rate'].mean():.4f}%")

    return df


# ── DESCRIPTIVE STATS ─────────────────────────────────────────────────────────

def describe_data(rates):
    """
    Print a full statistical summary.
    Copy the output into your report's Data section.
    """
    daily_chg = np.diff(rates) * 10_000   # basis points

    print("\n" + "=" * 48)
    print("  Data Summary  —  DTB3  2006–2026")
    print("=" * 48)
    print(f"  Observations       : {len(rates):,}")
    print(f"  Mean rate          : {rates.mean()*100:.4f}%")
    print(f"  Std of level       : {rates.std()*100:.4f}%")
    print(f"  Min rate           : {rates.min()*100:.4f}%")
    print(f"  Max rate           : {rates.max()*100:.4f}%")
    print(f"  Mean daily Δr      : {daily_chg.mean():.4f} bp")
    print(f"  Std of daily Δr    : {daily_chg.std():.4f} bp")
    print(f"  Days rate rose     : {(daily_chg>0).sum():,} "
          f"({(daily_chg>0).mean()*100:.1f}%)")
    print(f"  Days rate fell     : {(daily_chg<0).sum():,} "
          f"({(daily_chg<0).mean()*100:.1f}%)")
    print(f"  Days unchanged     : {(daily_chg==0).sum():,} "
          f"({(daily_chg==0).mean()*100:.1f}%)")
    print(f"  Negative rate obs  : {(rates<0).sum():,} "
          f"({(rates<0).mean()*100:.3f}%)")
    print("=" * 48 + "\n")


# ── PLOTS ─────────────────────────────────────────────────────────────────────

def plot_raw_data(rates, dates, save=True):
    """
    Two-panel figure:
      Top    — full rate time series with annotated key events
      Bottom — histogram of daily changes with normal distribution fit

    Use as Figure 1 in your report.
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    # ── top: rate history ─────────────────────────────────────────────
    ax = axes[0]
    ax.plot(dates, rates * 100, color='steelblue', linewidth=0.8, zorder=3)
    ax.fill_between(dates, 0, rates * 100,
                    alpha=0.12, color='steelblue', zorder=2)

    events = {
        '2008-09-15': ('GFC',             'crimson'),
        '2015-12-16': ('First hike',      '#887700'),
        '2020-03-15': ('COVID cuts',      'darkorange'),
        '2022-03-17': ('Hike cycle',      'darkgreen'),
    }
    for date_str, (label, color) in events.items():
        ts = pd.Timestamp(date_str)
        if dates[0] <= ts <= dates[-1]:
            ax.axvline(ts, color=color, lw=1.1, ls='--', alpha=0.8, zorder=4)
            ax.text(ts, rates.max() * 100 * 0.82,
                    f' {label}', color=color, fontsize=8.5, zorder=5)

    ax.set_ylabel('3-Month T-Bill Rate (%)', fontsize=11)
    ax.set_title(
        f'US 3-Month Treasury Bill Rate — DTB3 (FRED)  '
        f'| {dates[0].date()} to {dates[-1].date()}',
        fontsize=12)
    ax.grid(alpha=0.25)

    # ── bottom: histogram of Δr ───────────────────────────────────────
    ax2     = axes[1]
    delta_r = np.diff(rates) * 10_000   # basis points
    ax2.hist(delta_r, bins=120, color='steelblue',
             alpha=0.65, edgecolor='white', linewidth=0.3)
    ax2.axvline(0, color='crimson', linewidth=1.2, linestyle='--')

    mu, std = sp_stats.norm.fit(delta_r)
    x_fit   = np.linspace(delta_r.min(), delta_r.max(), 300)
    bw      = (delta_r.max() - delta_r.min()) / 120
    ax2.plot(x_fit,
             sp_stats.norm.pdf(x_fit, mu, std) * len(delta_r) * bw,
             'r-', lw=1.8,
             label=f'Normal fit  μ={mu:.3f} bp  σ={std:.3f} bp')

    ax2.set_xlabel('Daily change Δr (basis points)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of daily rate changes', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    plt.tight_layout()

    if save:
        fig.savefig('data_overview.png', dpi=150, bbox_inches='tight')
        fig.savefig('data_overview.pdf',           bbox_inches='tight')
        print("[data_loader] Saved data_overview.png + .pdf")

    plt.close()
    return fig


# ── STANDALONE TEST ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    rates, dates = load_dtb3()
    describe_data(rates)
    plot_raw_data(rates, dates)
    print("data_loader.py — standalone test complete.")
