import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST — Boss Agent Recommendation Logic vs Historical Cycles
# Replicates the exact compute_heat_index + _recommend formulas against
# each labeled historical cycle. Ground truth = CSUSHPISA price change.
# ─────────────────────────────────────────────────────────────────────────────

# ── Design tokens ─────────────────────────────────────────────────────────────
BT_BG      = '#1D1D20'
BT_TEXT    = '#fbfbff'
BT_GRID    = '#2e2e33'
BT_BLUE    = '#A1C9F4'
BT_ORANGE  = '#FFB482'
BT_GREEN   = '#8DE5A1'
BT_CORAL   = '#FF9F9B'
BT_LAVEND  = '#D0BBFF'
BT_GOLD    = '#ffd400'
BT_SUCCESS = '#17b26a'
BT_DANGER  = '#f04438'

CYCLE_COLORS = {
    '2008 Financial Crisis':       BT_CORAL,
    '2012 Recovery':               BT_GREEN,
    '2020 COVID Crash':            BT_ORANGE,
    '2021-22 Boom':                BT_BLUE,
    '2023 Rate Hike Correction':   BT_LAVEND,
}

# ── Clamp helper (mirrors economist_agent exactly) ───────────────────────────
def _clamp(val, lo=1.0, hi=10.0):
    return max(lo, min(hi, val))


# ── Replicate compute_heat_index per-cycle ────────────────────────────────────
def compute_cycle_heat_index(cycle_df: pd.DataFrame) -> dict:
    """
    Re-runs the same formula as economist_agent.compute_heat_index() but on a
    per-cycle slice of historical_cycles_df.

    Component formulas (identical to the live agent):
      mort_score  = clamp(1 + 9 * (latest_mort - min_mort) / range_mort)
      houst_score = clamp(1 + 9 * (1 - (latest_houst - min_houst) / range_houst))
      inf_score   = clamp(1 + 9 * (yoy_cpi_pct / 5.0))
      composite   = mean(mort_score, houst_score, inf_score)

    For a multi-month cycle slice we use:
      - "latest" = last month in the cycle
      - min/max range = entire cycle window (same as 5-yr window used live)
    """
    mort  = cycle_df['mortgage_rate_pct'].dropna()
    houst = cycle_df['housing_starts_k'].dropna()
    cpi   = cycle_df['cpi_index'].dropna()

    # Mortgage pressure
    mort_score = _clamp(1 + 9 * (mort.iloc[-1] - mort.min()) / max(mort.max() - mort.min(), 0.01))

    # Supply pressure (inverted)
    houst_norm  = (houst.iloc[-1] - houst.min()) / max(houst.max() - houst.min(), 1)
    houst_score = _clamp(1 + 9 * (1 - houst_norm))

    # Inflation momentum (YoY CPI)
    if 'cpi_yoy_pct' in cycle_df.columns:
        _yoy = cycle_df['cpi_yoy_pct'].dropna()
        yoy_cpi = _yoy.iloc[-1] if len(_yoy) else 0.0
    else:
        _cpi_12m_ago = cpi.iloc[-13] if len(cpi) > 13 else cpi.iloc[0]
        yoy_cpi = (cpi.iloc[-1] - _cpi_12m_ago) / _cpi_12m_ago * 100

    inf_score = _clamp(1 + 9 * (yoy_cpi / 5.0))

    composite = round((mort_score + houst_score + inf_score) / 3, 2)

    return {
        'heat_index':   composite,
        'mort_score':   round(mort_score, 2),
        'houst_score':  round(houst_score, 2),
        'inf_score':    round(inf_score, 2),
        'yoy_cpi':      round(float(yoy_cpi), 2),
        'latest_mort':  round(float(mort.iloc[-1]), 2),
        'latest_houst': round(float(houst.iloc[-1]), 0),
    }


# ── Replicate _recommend decision matrix (mirrors boss_agent exactly) ─────────
def _recommend(h: float, p: float) -> str:
    """Exact copy of boss_agent._recommend() decision logic."""
    if h <= 4 and p >= 7:
        return 'Hold'
    elif h >= 7 and p >= 7:
        return 'Strong Buy'
    elif h >= 7 and p < 7:
        return 'Sell'
    elif 4 < h < 7 and p >= 7:
        return 'Buy'
    elif 4 < h < 7 and p < 7:
        return 'Sell'
    else:   # h <= 4 and p < 7
        return 'Sell'


# ── Derive per-cycle property_value_score from HPI relative position ──────────
def compute_property_value_score(cycle_df: pd.DataFrame,
                                  full_df: pd.DataFrame) -> float:
    """
    Property Value Score (0–10) proxy for backtesting:
    Measures where the cycle's avg HPI sits within the full 20-year HPI range.
    High score → prices are relatively LOW vs history (good value to buy).
    Low  score → prices are relatively HIGH vs history (less value / overvalued).

    This mirrors the appraiser's 'value score' intent: reward undervalued assets.
    Formula: score = 10 * (1 - (avg_cycle_hpi - min_hpi) / (max_hpi - min_hpi))
    """
    all_hpi   = full_df['case_shiller_hpi'].dropna()
    hpi_min   = all_hpi.min()
    hpi_max   = all_hpi.max()
    avg_cycle = cycle_df['case_shiller_hpi'].mean()

    score = 10.0 * (1.0 - (avg_cycle - hpi_min) / max(hpi_max - hpi_min, 1.0))
    return round(float(np.clip(score, 0, 10)), 3)


# ── Ground-truth outcome from CSUSHPISA price change ─────────────────────────
def actual_outcome_from_hpi(hpi_start: float, hpi_end: float) -> tuple:
    """
    Maps HPI % change to an actual market action label:
      > +5%   → 'Buy'   (strong appreciation = holding/buying rewarded)
      +1–5%   → 'Hold'  (mild appreciation)
      -1–+1%  → 'Hold'  (flat market)
      < -1%   → 'Sell'  (prices fell = selling would have been correct)
    """
    pct = (hpi_end - hpi_start) / hpi_start * 100
    if pct > 5:
        outcome = 'Buy'
    elif pct >= -1:
        outcome = 'Hold'
    else:
        outcome = 'Sell'
    return outcome, round(pct, 2)


# ── Map recommendation → outcome equivalence ─────────────────────────────────
def is_correct(rec: str, outcome: str) -> bool:
    """
    Checks if the generated recommendation aligns with the actual outcome.
    Mapping:
      Strong Buy  ↔  Buy outcome
      Buy         ↔  Buy outcome
      Hold        ↔  Hold outcome
      Sell        ↔  Sell outcome
    Strong Buy is also correct if outcome is Hold (conservative is acceptable).
    """
    REC_MAP = {
        'Strong Buy': ['Buy', 'Hold'],   # strong appreciation or flat both acceptable
        'Buy':        ['Buy'],
        'Hold':       ['Hold', 'Buy'],   # holding through mild appreciation is fine
        'Sell':       ['Sell'],
    }
    return outcome in REC_MAP.get(rec, [])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BACKTEST LOOP — one row per cycle
# ─────────────────────────────────────────────────────────────────────────────
_labeled_cycles = [
    '2008 Financial Crisis',
    '2012 Recovery',
    '2020 COVID Crash',
    '2021-22 Boom',
    '2023 Rate Hike Correction',
]

_backtest_rows = []

for _cname in _labeled_cycles:
    _slice = historical_cycles_df[historical_cycles_df['cycle_name'] == _cname].copy()
    if len(_slice) == 0:
        continue

    # Heat index for this cycle
    _heat_info   = compute_cycle_heat_index(_slice)
    _heat        = _heat_info['heat_index']

    # Property value score (appraiser proxy)
    _prop_score  = compute_property_value_score(_slice, historical_cycles_df)

    # Generated recommendation
    _rec         = _recommend(_heat, _prop_score)

    # Ground truth: HPI change across cycle
    _hpi_start   = _slice['case_shiller_hpi'].iloc[0]
    _hpi_end     = _slice['case_shiller_hpi'].iloc[-1]
    _outcome, _pct_chg = actual_outcome_from_hpi(_hpi_start, _hpi_end)

    # Correctness
    _correct     = is_correct(_rec, _outcome)

    _backtest_rows.append({
        'cycle_name':              _cname,
        'cycle_type':              _slice['cycle_type'].iloc[0],
        'period':                  f"{_slice['date'].min().strftime('%Y-%m')} → {_slice['date'].max().strftime('%Y-%m')}",
        'heat_index':              _heat,
        'property_value_score':    _prop_score,
        'mort_score':              _heat_info['mort_score'],
        'houst_score':             _heat_info['houst_score'],
        'inf_score':               _heat_info['inf_score'],
        'recommendation_generated': _rec,
        'actual_outcome':          _outcome,
        'price_change_pct':        _pct_chg,
        'hpi_start':               round(_hpi_start, 1),
        'hpi_end':                 round(_hpi_end, 1),
        'correct':                 _correct,
    })

backtest_results_df = pd.DataFrame(_backtest_rows)

# ─────────────────────────────────────────────────────────────────────────────
# ACCURACY METRICS
# ─────────────────────────────────────────────────────────────────────────────
_n_total   = len(backtest_results_df)
_n_correct = backtest_results_df['correct'].sum()
_win_rate  = round(100 * _n_correct / _n_total, 1)

# Precision & Recall per recommendation class
_classes = ['Buy', 'Strong Buy', 'Hold', 'Sell']

_metrics_rows = []
for _cls in _classes:
    _predicted_pos = backtest_results_df['recommendation_generated'] == _cls
    _actual_pos    = backtest_results_df['actual_outcome'].apply(
        lambda x: x in ({'Buy': ['Buy'], 'Strong Buy': ['Buy', 'Hold'],
                          'Hold': ['Hold', 'Buy'], 'Sell': ['Sell']}.get(_cls, []))
    )
    _tp = (_predicted_pos & _actual_pos).sum()
    _fp = (_predicted_pos & ~_actual_pos).sum()
    _fn = (~_predicted_pos & _actual_pos).sum()

    _precision = round(_tp / max(_tp + _fp, 1), 3)
    _recall    = round(_tp / max(_tp + _fn, 1), 3)
    _f1        = round(2 * _precision * _recall / max(_precision + _recall, 0.001), 3)
    _support   = int(_predicted_pos.sum())

    _metrics_rows.append({
        'class':     _cls,
        'precision': _precision,
        'recall':    _recall,
        'f1_score':  _f1,
        'support':   _support,
    })

backtest_metrics_df = pd.DataFrame(_metrics_rows)

# ─────────────────────────────────────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────────────────────────────────────
_DIV  = '=' * 100
_DIV2 = '-' * 100

print(_DIV)
print('🔁  BOSS AGENT BACKTEST — Historical Cycle Validation')
print(_DIV)
print(f'   Cycles tested : {_n_total}')
print(f'   Correct calls : {_n_correct}/{_n_total}')
print(f'   Win Rate      : {_win_rate}%')
print(_DIV)

print(f"\n{'#':<3} {'Cycle':<30} {'Period':<22} {'Heat':>6} {'PropScore':>10} "
      f"{'Rec':>12} {'Actual':>8} {'ΔPrice%':>9} {'✓':>4}")
print(_DIV2)

for _i, _r in backtest_results_df.iterrows():
    _chk = '✅' if _r['correct'] else '❌'
    _rec_icon = {'Strong Buy': '🟢', 'Buy': '🔵', 'Hold': '🟡', 'Sell': '🔴'}.get(_r['recommendation_generated'], '⚪')
    print(f"{_i+1:<3} {_r['cycle_name']:<30} {_r['period']:<22} "
          f"{_r['heat_index']:>6.2f} {_r['property_value_score']:>10.3f} "
          f"{_rec_icon} {_r['recommendation_generated']:>10} {_r['actual_outcome']:>8} "
          f"{_r['price_change_pct']:>+8.2f}% {_chk:>4}")

print(_DIV2)

print('\n\n📊  COMPONENT SCORES PER CYCLE:\n')
print(f"{'Cycle':<30} {'MortScore':>10} {'HoustScore':>11} {'InfScore':>9} {'HeatIdx':>9} {'PropScore':>10}")
print(_DIV2)
for _, _r in backtest_results_df.iterrows():
    print(f"{_r['cycle_name']:<30} {_r['mort_score']:>10.2f} {_r['houst_score']:>11.2f} "
          f"{_r['inf_score']:>9.2f} {_r['heat_index']:>9.2f} {_r['property_value_score']:>10.3f}")

print('\n\n📐  PRECISION / RECALL BY CLASS:\n')
print(f"{'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
print('-' * 50)
for _, _m in backtest_metrics_df.iterrows():
    print(f"{_m['class']:<12} {_m['precision']:>10.3f} {_m['recall']:>8.3f} {_m['f1_score']:>8.3f} {_m['support']:>9}")

print(f'\n{_DIV}')
print(f'✅  Overall Win Rate: {_win_rate}% ({_n_correct}/{_n_total} cycles correctly called)')
print(_DIV)

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION — Backtest Results Dashboard
# ─────────────────────────────────────────────────────────────────────────────

# Chart 1: Heat Index vs Property Value Score, colored by correctness
backtest_chart1 = plt.figure(figsize=(10, 7))
backtest_chart1.patch.set_facecolor(BT_BG)
ax = backtest_chart1.add_subplot(111)
ax.set_facecolor(BT_BG)

for _, _r in backtest_results_df.iterrows():
    _color = BT_SUCCESS if _r['correct'] else BT_DANGER
    _label = f"{_r['cycle_name']}\n{_r['recommendation_generated']} → {_r['actual_outcome']}"
    ax.scatter(_r['heat_index'], _r['property_value_score'],
               color=_color, s=200, zorder=5, edgecolors=BT_TEXT, linewidths=0.8)
    ax.annotate(_r['cycle_name'].replace(' ', '\n'),
                xy=(_r['heat_index'], _r['property_value_score']),
                xytext=(8, 5), textcoords='offset points',
                fontsize=7.5, color=BT_TEXT, ha='left')

# Decision boundary shading
for _h_range, _p_thresh, _c, _lbl in [
    ((7, 11), (7, 11), 'rgba(23,178,106,0.08)', 'Strong Buy Zone'),
    ((7, 11), (0, 7),  'rgba(240,68,56,0.08)',  'Sell Zone (hot, low quality)'),
    ((4, 7),  (7, 11), 'rgba(161,201,244,0.08)','Buy Zone'),
    ((4, 7),  (0, 7),  'rgba(240,68,56,0.05)',  'Sell Zone (warm, low quality)'),
    ((0, 4),  (7, 11), 'rgba(255,212,0,0.06)',  'Hold Zone'),
    ((0, 4),  (0, 7),  'rgba(240,68,56,0.05)',  'Sell Zone (cool, low quality)'),
]:
    pass  # zone overlays as rectangles
ax.axhline(y=7, color=BT_GRID, linestyle='--', linewidth=1, alpha=0.6)
ax.axvline(x=7, color=BT_GRID, linestyle='--', linewidth=1, alpha=0.6)
ax.axvline(x=4, color=BT_GRID, linestyle=':', linewidth=1, alpha=0.4)

ax.set_xlabel('Market Heat Index', color=BT_TEXT, fontsize=12)
ax.set_ylabel('Property Value Score', color=BT_TEXT, fontsize=12)
ax.set_title('Boss Agent Backtest: Heat Index vs Property Score by Cycle', color=BT_TEXT, fontsize=14, pad=15)
ax.tick_params(colors=BT_TEXT)
for spine in ax.spines.values():
    spine.set_edgecolor(BT_GRID)
ax.set_xlim(0, 12)
ax.set_ylim(0, 11)

_correct_patch = mpatches.Patch(color=BT_SUCCESS, label='✅ Correct Call')
_wrong_patch   = mpatches.Patch(color=BT_DANGER,  label='❌ Incorrect Call')
ax.legend(handles=[_correct_patch, _wrong_patch], facecolor=BT_BG, edgecolor=BT_GRID,
          labelcolor=BT_TEXT, fontsize=10)

ax.text(7.5, 8.5, 'Strong Buy', color=BT_GREEN, fontsize=8.5, alpha=0.8)
ax.text(7.5, 3.5, 'Sell (hot,\nlow quality)', color=BT_CORAL, fontsize=8, alpha=0.8)
ax.text(4.5, 8.5, 'Buy', color=BT_BLUE, fontsize=8.5, alpha=0.8)
ax.text(0.5, 8.5, 'Hold', color=BT_GOLD, fontsize=8.5, alpha=0.8)
ax.text(0.5, 3.5, 'Sell\n(cool, low)', color=BT_CORAL, fontsize=8, alpha=0.8)

plt.tight_layout()
plt.show()

# Chart 2: Price Change % per Cycle, colored by rec vs actual
backtest_chart2 = plt.figure(figsize=(11, 6))
backtest_chart2.patch.set_facecolor(BT_BG)
ax2 = backtest_chart2.add_subplot(111)
ax2.set_facecolor(BT_BG)

_xpos   = np.arange(len(backtest_results_df))
_pcts   = backtest_results_df['price_change_pct'].values
_colors = [BT_SUCCESS if c else BT_DANGER for c in backtest_results_df['correct']]
_bars   = ax2.bar(_xpos, _pcts, color=_colors, edgecolor=BT_GRID, linewidth=0.8, width=0.6)

ax2.axhline(0, color=BT_TEXT, linewidth=0.8, alpha=0.4)
ax2.axhline(5,  color=BT_GREEN, linewidth=1, linestyle='--', alpha=0.5, label='+5% Buy threshold')
ax2.axhline(-1, color=BT_CORAL, linewidth=1, linestyle='--', alpha=0.5, label='-1% Sell threshold')

for _xi, (_pct, _r) in enumerate(zip(_pcts, backtest_results_df.itertuples())):
    _offset = 0.5 if _pct >= 0 else -2.5
    ax2.text(_xi, _pct + _offset,
             f"{_r.recommendation_generated}\n→{_r.actual_outcome}",
             ha='center', va='bottom', color=BT_TEXT, fontsize=8.5)

ax2.set_xticks(_xpos)
ax2.set_xticklabels(
    [c.replace(' ', '\n') for c in backtest_results_df['cycle_name']],
    color=BT_TEXT, fontsize=9
)
ax2.set_ylabel('HPI Price Change % (cycle duration)', color=BT_TEXT, fontsize=11)
ax2.set_title('Boss Agent Backtest: Recommendation vs Actual HPI Outcome per Cycle',
              color=BT_TEXT, fontsize=13, pad=12)
ax2.tick_params(colors=BT_TEXT)
for spine in ax2.spines.values():
    spine.set_edgecolor(BT_GRID)
ax2.legend(facecolor=BT_BG, edgecolor=BT_GRID, labelcolor=BT_TEXT, fontsize=9)

plt.tight_layout()
plt.show()

# Chart 3: Win rate summary
backtest_chart3 = plt.figure(figsize=(7, 5))
backtest_chart3.patch.set_facecolor(BT_BG)
ax3 = backtest_chart3.add_subplot(111)
ax3.set_facecolor(BT_BG)

_wlabels = ['Correct', 'Incorrect']
_wsizes  = [_n_correct, _n_total - _n_correct]
_wcolors = [BT_SUCCESS, BT_DANGER]
_bars3   = ax3.bar(_wlabels, _wsizes, color=_wcolors, edgecolor=BT_GRID, linewidth=0.8, width=0.45)

for _b in _bars3:
    ax3.text(_b.get_x() + _b.get_width() / 2, _b.get_height() + 0.05,
             str(int(_b.get_height())), ha='center', va='bottom', color=BT_TEXT, fontsize=14, fontweight='bold')

ax3.set_ylim(0, _n_total + 1)
ax3.set_ylabel('Number of Cycles', color=BT_TEXT, fontsize=11)
ax3.set_title(f'Overall Win Rate: {_win_rate}% ({_n_correct}/{_n_total} Correct)',
              color=BT_TEXT, fontsize=13, pad=12)
ax3.tick_params(colors=BT_TEXT)
for spine in ax3.spines.values():
    spine.set_edgecolor(BT_GRID)

plt.tight_layout()
plt.show()

print(f"\n✅ backtest_results_df created — shape: {backtest_results_df.shape}")
print(f"   Columns: {list(backtest_results_df.columns)}")
