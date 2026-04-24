
import pandas as pd
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL MARKET CYCLES — 20-Year FRED Data with Cycle Labels
# Fetches: MORTGAGE30US, HOUST, CPIAUCSL, CSUSHPISA (Case-Shiller HPI)
# Labels: 5 known housing/macro market cycles
# ─────────────────────────────────────────────────────────────────────────────

FRED_API_KEY = 'your-fred-api-key-here'  # Replace with your real FRED API key

CYCLE_SERIES = {
    'MORTGAGE30US': 'Mortgage Rate (30Y Fixed, %)',
    'HOUST':        'Housing Starts (Thousands)',
    'CPIAUCSL':     'CPI (All Urban Consumers, Index)',
    'CSUSHPISA':    'Case-Shiller HPI (National, SA)',
}

end_date   = datetime(2024, 12, 1)
start_date = datetime(2004, 1, 1)  # ~20 years of history

# ─────────────────────────────────────────────
# DEFINED MARKET CYCLES  (inclusive date ranges)
# ─────────────────────────────────────────────
CYCLES = [
    {
        'cycle_name':        '2008 Financial Crisis',
        'period_start':      '2007-01-01',
        'period_end':        '2009-06-30',
        'description':       'Subprime mortgage collapse, credit freeze, mass foreclosures',
        'cycle_type':        'Crash',
        'color_code':        '#f04438',  # danger red
    },
    {
        'cycle_name':        '2012 Recovery',
        'period_start':      '2011-01-01',
        'period_end':        '2014-12-31',
        'description':       'Fed QE-driven stabilization, prices bottomed and began climbing',
        'cycle_type':        'Recovery',
        'color_code':        '#17b26a',  # success green
    },
    {
        'cycle_name':        '2020 COVID Crash',
        'period_start':      '2020-01-01',
        'period_end':        '2020-06-30',
        'description':       'Pandemic shock; housing starts collapsed briefly before rebounding',
        'cycle_type':        'Crash',
        'color_code':        '#FFB482',  # orange
    },
    {
        'cycle_name':        '2021-22 Boom',
        'period_start':      '2020-07-01',
        'period_end':        '2022-06-30',
        'description':       'Ultra-low rates + remote work surge drove historic price appreciation',
        'cycle_type':        'Boom',
        'color_code':        '#A1C9F4',  # light blue
    },
    {
        'cycle_name':        '2023 Rate Hike Correction',
        'period_start':      '2022-07-01',
        'period_end':        '2024-12-31',
        'description':       'Fed tightening cycle pushed mortgage rates to 8%; affordability crisis',
        'cycle_type':        'Correction',
        'color_code':        '#D0BBFF',  # lavender
    },
]

# ─────────────────────────────────────────────
# SYNTHETIC REALISTIC 20-YEAR MACRO DATA
# Hand-calibrated to match known historical patterns
# ─────────────────────────────────────────────
def generate_historical_mock_data(start: datetime, end: datetime) -> pd.DataFrame:
    """
    Generate realistic monthly macro data that closely mirrors known historical patterns:
    - MORTGAGE30US: peaked ~6.7% pre-2008, bottomed ~3% post-2009, spiked ~8% in 2023
    - HOUST:        crashed from ~1800K to ~400K in 2009, COVID dip, recovered
    - CPIAUCSL:     slow drift 2004–2020, then sharp acceleration 2021–2023
    - CSUSHPISA:    boom pre-2007, bust 2007–2012, recovery, COVID boom, plateau
    """
    dates = pd.date_range(start=start, end=end, freq='MS')
    n = len(dates)
    t_yrs = np.array([(d - datetime(2004, 1, 1)).days / 365.25 for d in dates])

    # — Mortgage Rate (30Y, %) ——————————————————————————————
    # Starts ~5.8%, dips post-crisis to ~3.3%, climbs to 8% by 2023
    mortgage = np.interp(t_yrs, [0, 3, 6, 10, 16, 17, 18, 19, 20],
                                 [5.8, 6.6, 5.0, 3.4, 3.1, 3.0, 5.5, 7.8, 6.9])
    mortgage += np.random.default_rng(42).normal(0, 0.12, n)

    # — Housing Starts (Thousands) —————————————————————————
    # 2004: ~2000K, crash to ~400K by 2009, slow recovery, COVID dip, boom ~1700K, settle ~1400K
    houst = np.interp(t_yrs, [0, 1, 3, 5, 6, 8, 11, 14, 16, 16.4, 17, 18, 19, 20],
                              [1900, 2050, 2100, 1200, 550, 450, 800, 1100, 1350, 1100, 1500, 1700, 1400, 1380])
    houst += np.random.default_rng(7).normal(0, 50, n)

    # — CPI (All Urban Consumers, Index 1982-84=100) ————————
    # ~188 in 2004, gradual to ~260 by 2020, then surge to ~315 by 2023
    cpi_base = np.interp(t_yrs, [0, 4, 8, 12, 16, 17, 18, 19, 20],
                                  [188, 210, 218, 232, 260, 271, 292, 304, 314])
    cpi_base += np.random.default_rng(13).normal(0, 0.5, n)

    # — Case-Shiller HPI (National, SA, Index 2000=100) ————
    # 2004: ~170, peak ~205 in 2006, trough ~145 in 2012, then COVID boom to ~320, slight correction
    hpi = np.interp(t_yrs, [0, 2, 3, 5, 7, 8, 11, 14, 16, 16.5, 17, 18, 19, 20],
                            [170, 195, 206, 175, 150, 145, 165, 195, 230, 215, 275, 315, 305, 310])
    hpi += np.random.default_rng(99).normal(0, 1.5, n)

    return pd.DataFrame({
        'date':         dates,
        'MORTGAGE30US': np.round(mortgage, 2),
        'HOUST':        np.round(houst, 0),
        'CPIAUCSL':     np.round(cpi_base, 1),
        'CSUSHPISA':    np.round(hpi, 1),
    })


# ─────────────────────────────────────────────
# FETCH DATA — FRED API with mock fallback
# ─────────────────────────────────────────────
_use_mock = False
try:
    from fredapi import Fred
    _fred   = Fred(api_key=FRED_API_KEY)
    _frames = []
    for _sid in CYCLE_SERIES:
        _s = _fred.get_series(
            _sid,
            observation_start=start_date.strftime('%Y-%m-%d'),
            observation_end=end_date.strftime('%Y-%m-%d'),
        )
        _s.name = _sid
        _frames.append(_s)
    _raw = pd.concat(_frames, axis=1).resample('MS').mean().reset_index()
    _raw.rename(columns={'index': 'date'}, inplace=True)
    _raw.dropna(inplace=True)
    _raw['date'] = pd.to_datetime(_raw['date'])
    macro_cycles_raw = _raw.sort_values('date').reset_index(drop=True)
    print("✅ Live FRED data fetched for 4 series over 20 years.")
except Exception as _e:
    print(f"⚠️  FRED API unavailable ({type(_e).__name__}: {_e}).")
    print("   Using realistic hand-calibrated mock data matching known historical patterns.")
    macro_cycles_raw = generate_historical_mock_data(start_date, end_date)
    _use_mock = True

print(f"   Period : {macro_cycles_raw['date'].min().date()} → {macro_cycles_raw['date'].max().date()}")
print(f"   Shape  : {macro_cycles_raw.shape}")

# ─────────────────────────────────────────────
# DERIVED INDICATORS
# ─────────────────────────────────────────────
macro_cycles_raw = macro_cycles_raw.sort_values('date').reset_index(drop=True)
macro_cycles_raw['CPI_YOY_PCT']   = macro_cycles_raw['CPIAUCSL'].pct_change(12).mul(100).round(2)
macro_cycles_raw['HPI_YOY_PCT']   = macro_cycles_raw['CSUSHPISA'].pct_change(12).mul(100).round(2)
macro_cycles_raw['HOUST_MA6']     = macro_cycles_raw['HOUST'].rolling(6, min_periods=1).mean().round(0)

# ─────────────────────────────────────────────
# ASSIGN CYCLE LABELS — join on date range
# ─────────────────────────────────────────────
def _assign_cycle(date: pd.Timestamp) -> tuple[str, str, str]:
    for cyc in CYCLES:
        if pd.Timestamp(cyc['period_start']) <= date <= pd.Timestamp(cyc['period_end']):
            return cyc['cycle_name'], cyc['cycle_type'], cyc['description']
    return 'Pre-Crisis Era', 'Baseline', 'Pre-2008 housing expansion'

_assignments = macro_cycles_raw['date'].apply(_assign_cycle)
macro_cycles_raw['cycle_name']  = [a[0] for a in _assignments]
macro_cycles_raw['cycle_type']  = [a[1] for a in _assignments]
macro_cycles_raw['cycle_desc']  = [a[2] for a in _assignments]

# ─────────────────────────────────────────────
# BUILD historical_cycles_df
# Final output: one row per month with cycle labels + macro values
# ─────────────────────────────────────────────
historical_cycles_df = macro_cycles_raw[[
    'date', 'cycle_name', 'cycle_type',
    'MORTGAGE30US', 'HOUST', 'CPIAUCSL', 'CSUSHPISA',
    'CPI_YOY_PCT', 'HPI_YOY_PCT', 'HOUST_MA6',
    'cycle_desc',
]].copy()

historical_cycles_df.rename(columns={
    'MORTGAGE30US': 'mortgage_rate_pct',
    'HOUST':        'housing_starts_k',
    'CPIAUCSL':     'cpi_index',
    'CSUSHPISA':    'case_shiller_hpi',
    'CPI_YOY_PCT':  'cpi_yoy_pct',
    'HPI_YOY_PCT':  'hpi_yoy_pct',
    'HOUST_MA6':    'housing_starts_ma6',
}, inplace=True)

historical_cycles_df.reset_index(drop=True, inplace=True)

# ─────────────────────────────────────────────
# CYCLE SUMMARY TABLE — one row per named cycle
# ─────────────────────────────────────────────
_labeled = historical_cycles_df[historical_cycles_df['cycle_name'] != 'Pre-Crisis Era']

cycle_summary = (
    _labeled.groupby(['cycle_name', 'cycle_type', 'cycle_desc'])
    .agg(
        period_start   = ('date', 'min'),
        period_end     = ('date', 'max'),
        n_months       = ('date', 'count'),
        avg_mortgage   = ('mortgage_rate_pct',  'mean'),
        avg_hpi        = ('case_shiller_hpi',   'mean'),
        avg_cpi_yoy    = ('cpi_yoy_pct',        'mean'),
        avg_housing_k  = ('housing_starts_k',   'mean'),
        peak_hpi       = ('case_shiller_hpi',   'max'),
        trough_hpi     = ('case_shiller_hpi',   'min'),
    )
    .reset_index()
    .sort_values('period_start')
    .reset_index(drop=True)
)

cycle_summary['avg_mortgage']  = cycle_summary['avg_mortgage'].round(2)
cycle_summary['avg_hpi']       = cycle_summary['avg_hpi'].round(1)
cycle_summary['avg_cpi_yoy']   = cycle_summary['avg_cpi_yoy'].round(2)
cycle_summary['avg_housing_k'] = cycle_summary['avg_housing_k'].round(0)
cycle_summary['peak_hpi']      = cycle_summary['peak_hpi'].round(1)
cycle_summary['trough_hpi']    = cycle_summary['trough_hpi'].round(1)

# ─────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────
_DIV  = '=' * 95
_DIV2 = '-' * 95

print(f"\n{_DIV}")
print("📅  HISTORICAL MARKET CYCLES — 20-YEAR MACRO SNAPSHOT  (2004–2024)")
print(_DIV)
print(f"  Total monthly observations : {len(historical_cycles_df)}")
print(f"  Labeled cycle observations : {len(_labeled)}")
print(f"  Data source                : {'🌐 Live FRED API' if not _use_mock else '🧪 Calibrated Mock Data'}")
print(f"  Series fetched             : MORTGAGE30US, HOUST, CPIAUCSL, CSUSHPISA")
print(_DIV)

print("\n\n📊  CYCLE SUMMARY (aggregated per named cycle):\n")
print(f"{'#':<3} {'Cycle Name':<30} {'Type':<12} {'Period':<24} {'Mo':<5} "
      f"{'Avg Mortgage':>13} {'Avg HPI':>9} {'Avg CPI YoY':>12} {'Avg Starts(K)':>14}")
print(_DIV2)

_cycle_order = [
    '2008 Financial Crisis',
    '2012 Recovery',
    '2020 COVID Crash',
    '2021-22 Boom',
    '2023 Rate Hike Correction',
]

for _i, _row in cycle_summary.iterrows():
    _period = f"{_row['period_start'].strftime('%Y-%m')} → {_row['period_end'].strftime('%Y-%m')}"
    _icons  = {'Crash': '📉', 'Recovery': '📈', 'Boom': '🚀', 'Correction': '⚠️', 'Baseline': '📋'}
    _icon   = _icons.get(_row['cycle_type'], '•')
    print(f"{_i+1:<3} {_icon} {_row['cycle_name']:<27} {_row['cycle_type']:<12} {_period:<24} "
          f"{int(_row['n_months']):<5} {_row['avg_mortgage']:>12.2f}% {_row['avg_hpi']:>9.1f} "
          f"{_row['avg_cpi_yoy']:>11.2f}% {_row['avg_housing_k']:>13.0f}K")
    print(f"    └─ {_row['cycle_desc']}")
    print(f"       HPI range: {_row['trough_hpi']:.1f} → {_row['peak_hpi']:.1f}")
    print()

print(_DIV)
print("\n\n📋  SAMPLE: historical_cycles_df (first 3 rows per cycle):\n")
_sample = (
    historical_cycles_df[historical_cycles_df['cycle_name'].isin(_cycle_order)]
    .groupby('cycle_name', sort=False)
    .head(3)
    .sort_values('date')
    .reset_index(drop=True)
)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.2f}'.format)
print(_sample[[
    'date', 'cycle_name', 'cycle_type',
    'mortgage_rate_pct', 'housing_starts_k', 'cpi_index',
    'case_shiller_hpi', 'cpi_yoy_pct', 'hpi_yoy_pct',
]].to_string(index=False))

print(f"\n{_DIV}")
print(f"✅  historical_cycles_df ready — shape: {historical_cycles_df.shape}")
print(f"    Columns: {list(historical_cycles_df.columns)}")
print(f"    All 5 cycles labeled: {sorted(historical_cycles_df['cycle_name'].unique().tolist())}")
print(_DIV)
