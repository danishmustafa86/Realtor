
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE: Economist Agent CSV Validator
# Validates all outputs from the economist_agent block
# ─────────────────────────────────────────────────────────────────────────────

_test_results = []  # (test_name, passed, actual, expected, detail)

def _record(name: str, passed: bool, actual, expected: str, detail: str = ""):
    _test_results.append((name, passed, actual, expected, detail))

DIVIDER = "=" * 65

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — DATA INTEGRITY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("TEST 1 · Data Integrity")
print(DIVIDER)

# 1a. Expected columns present (ticket requests: date, mortgage_rate, housing_starts, cpi)
# Actual column names from the block: date, MORTGAGE30US, HOUST, CPIAUCSL
_REQUIRED_COLS = {'date', 'MORTGAGE30US', 'HOUST', 'CPIAUCSL'}
_actual_cols   = set(macro_data.columns)
_missing_cols  = _REQUIRED_COLS - _actual_cols
_cols_ok       = len(_missing_cols) == 0
_record(
    "1a · Required columns present",
    _cols_ok,
    actual=list(_actual_cols),
    expected=str(_REQUIRED_COLS),
    detail=f"Missing: {_missing_cols}" if not _cols_ok else "All required columns found",
)
print(f"  Shape  : {macro_data.shape}")
print(f"  Dtypes :\n{macro_data.dtypes.to_string()}")
print(f"  Cols OK: {'✅' if _cols_ok else '❌'} — {_REQUIRED_COLS}")

# 1b. No unexpected nulls in core columns
_null_counts = macro_data[list(_REQUIRED_COLS)].isnull().sum()
_nulls_ok    = _null_counts.sum() == 0
_record(
    "1b · No nulls in core columns",
    _nulls_ok,
    actual=_null_counts.to_dict(),
    expected="0 nulls in each column",
    detail=f"Null counts: {_null_counts.to_dict()}",
)
print(f"\n  Null counts (core columns):\n{_null_counts.to_string()}")
print(f"  Nulls OK: {'✅' if _nulls_ok else '❌'}")

# 1c. Date range spans at least 3 years
_date_min  = macro_data['date'].min()
_date_max  = macro_data['date'].max()
_span_days = (_date_max - _date_min).days
_span_ok   = _span_days >= 365 * 3
_record(
    "1c · Date range ≥ 3 years",
    _span_ok,
    actual=f"{_date_min.date()} → {_date_max.date()} ({_span_days} days)",
    expected="≥ 1095 days (~3 years)",
)
print(f"\n  Date range : {_date_min.date()} → {_date_max.date()}  ({_span_days} days)")
print(f"  Span ≥ 3yr : {'✅' if _span_ok else '❌'}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — MARKET HEAT INDEX
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("TEST 2 · Market Heat Index")
print(DIVIDER)

_mhi_is_float = isinstance(market_heat_index, float)
_mhi_in_range = 1.0 <= market_heat_index <= 10.0
_mhi_ok       = _mhi_is_float and _mhi_in_range

_record(
    "2a · market_heat_index is float in [1, 10]",
    _mhi_ok,
    actual=f"{market_heat_index} (type={type(market_heat_index).__name__})",
    expected="float between 1.0 and 10.0",
)

_status_label = (
    "🟢 Cool – Buyer-Friendly"  if market_heat_index < 4 else
    "🟡 Warm – Balanced Market" if market_heat_index < 6 else
    "🟠 Hot – Seller's Market"  if market_heat_index < 8 else
    "🔴 Very Hot – High Stress"
)

print(f"  market_heat_index : {market_heat_index}")
print(f"  Type              : {type(market_heat_index).__name__}")
print(f"  In range [1,10]   : {'✅' if _mhi_in_range else '❌'}")
print(f"  Market Status     : {_status_label}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — TREND VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("TEST 3 · Trend Validation")
print(DIVIDER)

_df = macro_data.copy().sort_values('date').reset_index(drop=True)

# Month-over-month changes
_mortgage_mom = _df['MORTGAGE30US'].diff()   # +ve = rates rising
_cpi_mom      = _df['CPIAUCSL'].diff()       # +ve = inflation rising

# Recent (last 6 months)
_recent_mort_mom = _mortgage_mom.iloc[-6:]
_recent_cpi_mom  = _cpi_mom.iloc[-6:]

_mort_trend_dir = "Rising 📈"  if _recent_mort_mom.mean() > 0 else "Falling 📉"
_cpi_trend_dir  = "Rising 📈"  if _recent_cpi_mom.mean() > 0  else "Falling 📉"

# Trend consistency check:
# If mortgage rates are rising, inflation pressure should be non-trivial (CPI above 2%)
_current_cpi_yoy = (
    (_df['CPIAUCSL'].iloc[-1] - _df['CPIAUCSL'].iloc[-13]) / _df['CPIAUCSL'].iloc[-13] * 100
    if len(_df) >= 14 else float('nan')
)

# Mortgage MoM should be numeric and finite (sanity check)
_mort_finite = bool(np.isfinite(_recent_mort_mom.dropna()).all())
_cpi_finite  = bool(np.isfinite(_recent_cpi_mom.dropna()).all())
_trends_ok   = _mort_finite and _cpi_finite

_record(
    "3a · Mortgage MoM changes are finite",
    _mort_finite,
    actual=f"mean={_recent_mort_mom.mean():.4f}%",
    expected="All finite numeric values",
)
_record(
    "3b · CPI MoM changes are finite",
    _cpi_finite,
    actual=f"mean={_recent_cpi_mom.mean():.4f} pts",
    expected="All finite numeric values",
)

# Check trend is directionally consistent: if CPI YoY > 3%, heat index should be > 5
_yoy_ok = True
if not np.isnan(_current_cpi_yoy):
    if _current_cpi_yoy > 3.0:
        _yoy_ok = market_heat_index > 5.0
_record(
    "3c · Heat index consistent with CPI YoY > 3%",
    _yoy_ok,
    actual=f"CPI YoY={_current_cpi_yoy:.2f}%, Heat={market_heat_index:.2f}",
    expected="Heat > 5.0 when CPI YoY > 3%",
)

print(f"  Mortgage MoM (last 6m)  mean : {_recent_mort_mom.mean():+.4f}%  → {_mort_trend_dir}")
print(f"  CPI MoM (last 6m) mean       : {_recent_cpi_mom.mean():+.4f} pts → {_cpi_trend_dir}")
print(f"  CPI Year-over-Year           : {_current_cpi_yoy:.2f}%")
print(f"  Mortgage MoM finite    : {'✅' if _mort_finite else '❌'}")
print(f"  CPI MoM finite         : {'✅' if _cpi_finite else '❌'}")
print(f"  Heat vs CPI YoY align  : {'✅' if _yoy_ok else '❌'}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — OUTPUT SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("TEST 4 · Output Summary Report")
print(DIVIDER)

_all_test_names = [t[0] for t in _test_results]
_all_passed     = [t[1] for t in _test_results]

_col_w = 45
print(f"\n  {'Check':<{_col_w}} {'Status':<8}  Actual vs Expected")
print(f"  {'-'*(_col_w)} {'-------':<8}  {'-------------------'}")

for _name, _passed, _actual, _expected, *_detail in _test_results:
    _status = "✅ PASS" if _passed else "❌ FAIL"
    _label  = f"actual={str(_actual)[:35]!r}  expected={_expected[:35]!r}"
    print(f"  {_name:<{_col_w}} {_status:<8}  {_label}")

_n_passed = sum(_all_passed)
_n_total  = len(_all_tests := _test_results)

print(f"\n{DIVIDER}")
_final_icon = "🎉" if _n_passed == _n_total else "⚠️ "
print(f"  {_final_icon}  FINAL RESULT : {_n_passed}/{_n_total} tests passed")
print(DIVIDER)

# Trigger assertion failure if any test failed (surfaces issues clearly)
_failures = [t[0] for t in _test_results if not t[1]]
if _failures:
    raise AssertionError(f"❌ {len(_failures)} test(s) FAILED: {_failures}")

print("\n✅ All tests passed — economist_agent outputs are valid.")
