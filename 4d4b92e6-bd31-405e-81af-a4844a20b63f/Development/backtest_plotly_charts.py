
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# ── Design tokens (mirroring backtest_boss_agent) ─────────────────────────────
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
    'Pre-Crisis Era':            '#909094',
    '2008 Financial Crisis':     BT_CORAL,
    '2012 Recovery':             BT_GREEN,
    '2020 COVID Crash':          BT_ORANGE,
    '2021-22 Boom':              BT_BLUE,
    '2023 Rate Hike Correction': BT_LAVEND,
}

# ─────────────────────────────────────────────────────────────────────────────
# ACCURACY SUMMARY (print before charts)
# ─────────────────────────────────────────────────────────────────────────────
_DIV = '=' * 80

_n_total   = len(backtest_results_df)
_n_correct = int(backtest_results_df['correct'].sum())
_win_rate  = round(100 * _n_correct / _n_total, 1)

_best  = backtest_results_df[backtest_results_df['correct']]['cycle_name'].tolist()
_worst = backtest_results_df[~backtest_results_df['correct']]['cycle_name'].tolist()

print(_DIV)
print('🔁  BACKTEST VALIDATION — PLOTLY DASHBOARD SUMMARY')
print(_DIV)
print(f'  Overall System Accuracy : {_win_rate}%  ({_n_correct}/{_n_total} cycles correctly predicted)')
print()
print('  ✅ PASS — Model predicted correctly:')
for c in _best:
    _row = backtest_results_df[backtest_results_df['cycle_name'] == c].iloc[0]
    print(f'     • {c}  [{_row["recommendation_generated"]} → actual: {_row["actual_outcome"]}, ΔHPI: {_row["price_change_pct"]:+.1f}%]')
print()
print('  ❌ FAIL — Model predicted incorrectly:')
for c in _worst:
    _row = backtest_results_df[backtest_results_df['cycle_name'] == c].iloc[0]
    print(f'     • {c}  [{_row["recommendation_generated"]} → actual: {_row["actual_outcome"]}, ΔHPI: {_row["price_change_pct"]:+.1f}%]')
print()
print(f'  🏆 Best predicted cycle : {_best[0] if _best else "None"}')
print(f'  ⚠️  Worst predicted cycle : {_worst[-1] if _worst else "None"}')
print(_DIV)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1: 20-Year Timeline — Market Heat Index overlaid with HPI % Change
# ─────────────────────────────────────────────────────────────────────────────

# Build per-month heat index from historical_cycles_df using same formula
_df = historical_cycles_df.copy().sort_values('date').reset_index(drop=True)

# Compute rolling market heat index using full 60-month window
_WINDOW = 60

_heat_list = []
for _i in range(len(_df)):
    _win = _df.iloc[max(0, _i - _WINDOW + 1): _i + 1]

    _mort  = _win['mortgage_rate_pct'].dropna()
    _houst = _win['housing_starts_k'].dropna()
    _cpi   = _win['cpi_index'].dropna()

    if len(_mort) < 2 or len(_houst) < 2 or len(_cpi) < 2:
        _heat_list.append(np.nan)
        continue

    def _clamp(v, lo=1.0, hi=10.0):
        return max(lo, min(hi, v))

    _ms  = _clamp(1 + 9 * (_mort.iloc[-1] - _mort.min()) / max(_mort.max() - _mort.min(), 0.01))
    _hn  = (_houst.iloc[-1] - _houst.min()) / max(_houst.max() - _houst.min(), 1)
    _hs  = _clamp(1 + 9 * (1 - _hn))
    _yoy_cpi = _df.loc[_df.index[_i], 'cpi_yoy_pct'] if 'cpi_yoy_pct' in _df.columns else 0.0
    _yoy_cpi = 0.0 if pd.isna(_yoy_cpi) else float(_yoy_cpi)
    _inf = _clamp(1 + 9 * (_yoy_cpi / 5.0))
    _heat_list.append(round((_ms + _hs + _inf) / 3, 3))

_df['heat_idx'] = _heat_list

# Labelled cycle periods for shading
_labeled = [
    ('2008 Financial Crisis',     BT_CORAL,   0.15),
    ('2012 Recovery',             BT_GREEN,   0.12),
    ('2020 COVID Crash',          BT_ORANGE,  0.12),
    ('2021-22 Boom',              BT_BLUE,    0.15),
    ('2023 Rate Hike Correction', BT_LAVEND,  0.12),
]

fig_timeline = go.Figure()

# Shade labeled cycle periods
for _cname, _col, _opacity in _labeled:
    _slice = _df[_df['cycle_name'] == _cname]
    if len(_slice) == 0:
        continue
    _x0 = _slice['date'].min().strftime('%Y-%m-%d')
    _x1 = _slice['date'].max().strftime('%Y-%m-%d')
    _row_bt = backtest_results_df[backtest_results_df['cycle_name'] == _cname]
    _correct = bool(_row_bt['correct'].iloc[0]) if len(_row_bt) else None
    _border  = BT_SUCCESS if _correct else BT_DANGER if _correct is not None else '#555'
    fig_timeline.add_vrect(
        x0=_x0, x1=_x1,
        fillcolor=_col, opacity=_opacity,
        line_width=2, line_color=_border,
        annotation_text=_cname.replace(' ', '<br>'),
        annotation_position='top left',
        annotation_font=dict(color=BT_TEXT, size=9),
    )

# HPI YoY % change (right axis)
_hpi_yoy = _df[['date', 'hpi_yoy_pct']].dropna()
fig_timeline.add_trace(go.Scatter(
    x=_hpi_yoy['date'],
    y=_hpi_yoy['hpi_yoy_pct'],
    name='HPI YoY % Change (actual)',
    line=dict(color=BT_GOLD, width=2.5),
    yaxis='y2',
    opacity=0.9,
))

# Market Heat Index (left axis)
_heat_valid = _df[['date', 'heat_idx']].dropna()
fig_timeline.add_trace(go.Scatter(
    x=_heat_valid['date'],
    y=_heat_valid['heat_idx'],
    name='Market Heat Index',
    line=dict(color=BT_BLUE, width=2),
    fill='tozeroy',
    fillcolor='rgba(161,201,244,0.08)',
    yaxis='y1',
))

# Threshold lines
fig_timeline.add_hline(y=7, line_dash='dash', line_color=BT_CORAL,  line_width=1.2,
                        annotation_text=' Hot (≥7)', annotation_font_color=BT_CORAL, annotation_position='right')
fig_timeline.add_hline(y=4, line_dash='dot',  line_color=BT_GOLD,   line_width=1.0,
                        annotation_text=' Warm (4)',  annotation_font_color=BT_GOLD,  annotation_position='right')

fig_timeline.update_layout(
    title=dict(text='20-Year Timeline: Market Heat Index vs Actual Home Price Changes', font=dict(color=BT_TEXT, size=16)),
    plot_bgcolor=BT_BG, paper_bgcolor=BT_BG,
    font=dict(color=BT_TEXT),
    xaxis=dict(title='Date', gridcolor=BT_GRID, showgrid=True, tickfont=dict(color=BT_TEXT)),
    yaxis=dict(
        title='Market Heat Index (1–10)',
        range=[0, 12],
        gridcolor=BT_GRID,
        showgrid=True,
        tickfont=dict(color=BT_BLUE),
        titlefont=dict(color=BT_BLUE),
    ),
    yaxis2=dict(
        title='HPI YoY % Change',
        overlaying='y',
        side='right',
        gridcolor=BT_GRID,
        showgrid=False,
        zeroline=True, zerolinecolor=BT_GRID, zerolinewidth=1,
        tickfont=dict(color=BT_GOLD),
        titlefont=dict(color=BT_GOLD),
    ),
    legend=dict(bgcolor='rgba(29,29,32,0.8)', bordercolor=BT_GRID, font=dict(color=BT_TEXT)),
    height=500,
    margin=dict(l=60, r=80, t=80, b=50),
)

fig_timeline.show()
print("✅ Chart 1 rendered: 20-Year Timeline")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2: Confusion Matrix — Predicted vs Actual per Cycle
# ─────────────────────────────────────────────────────────────────────────────

_all_labels = ['Buy', 'Hold', 'Sell', 'Strong Buy']

# Build a 4×4 confusion matrix: rows = actual, cols = predicted
_mat = pd.DataFrame(0, index=_all_labels, columns=_all_labels)
for _, _r in backtest_results_df.iterrows():
    _act = _r['actual_outcome']
    _pred = _r['recommendation_generated']
    if _act in _mat.index and _pred in _mat.columns:
        _mat.loc[_act, _pred] += 1

_z   = _mat.values.astype(float)
_z_text = [[str(int(v)) for v in row] for row in _z]

fig_cm = go.Figure(go.Heatmap(
    z=_z,
    x=_all_labels,
    y=_all_labels,
    text=_z_text,
    texttemplate='%{text}',
    colorscale=[
        [0.0, BT_BG],
        [0.01, '#1a3a2a'],
        [0.5, '#1f6b42'],
        [1.0, BT_SUCCESS],
    ],
    showscale=True,
    colorbar=dict(tickfont=dict(color=BT_TEXT), title=dict(text='Count', font=dict(color=BT_TEXT))),
    textfont=dict(color=BT_TEXT, size=20),
    xgap=3, ygap=3,
))

# Add diagonal annotations (correct = ✅, off-diagonal = ❌)
_shapes = []
for _ri, _act in enumerate(_all_labels):
    for _ci, _pred in enumerate(_all_labels):
        _val = _z[_ri][_ci]
        if _val > 0:
            _icon = '✅' if _ri == _ci else '❌'
            fig_cm.add_annotation(
                x=_pred, y=_act,
                text=_icon,
                showarrow=False,
                font=dict(size=11),
                xshift=0, yshift=-14,
            )

fig_cm.update_layout(
    title=dict(text='Confusion Matrix: Predicted vs Actual Market Call (Buy / Hold / Sell)', font=dict(color=BT_TEXT, size=15)),
    plot_bgcolor=BT_BG, paper_bgcolor=BT_BG,
    font=dict(color=BT_TEXT),
    xaxis=dict(title='Predicted Recommendation', tickfont=dict(color=BT_TEXT, size=12),
               gridcolor=BT_GRID, showgrid=False),
    yaxis=dict(title='Actual Outcome', tickfont=dict(color=BT_TEXT, size=12),
               gridcolor=BT_GRID, showgrid=False, autorange='reversed'),
    height=450,
    margin=dict(l=80, r=60, t=80, b=70),
)

fig_cm.show()
print("✅ Chart 2 rendered: Confusion Matrix")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3: Bar Chart — Win-Rate per Market Cycle
# ─────────────────────────────────────────────────────────────────────────────

_cycles   = backtest_results_df['cycle_name'].tolist()
_recs     = backtest_results_df['recommendation_generated'].tolist()
_actuals  = backtest_results_df['actual_outcome'].tolist()
_pcts     = backtest_results_df['price_change_pct'].tolist()
_corrects = backtest_results_df['correct'].tolist()

# Win/Fail colors per cycle
_bar_colors = [BT_SUCCESS if c else BT_DANGER for c in _corrects]

# Win rate label: 100% or 0% per cycle (single prediction each)
_wr_labels = ['PASS ✅' if c else 'FAIL ❌' for c in _corrects]

fig_wr = go.Figure()

# Price change bars (background context)
fig_wr.add_trace(go.Bar(
    name='HPI Price Change %',
    x=_cycles,
    y=_pcts,
    marker=dict(
        color=[BT_GREEN if p > 0 else BT_CORAL for p in _pcts],
        opacity=0.35,
        line=dict(color=BT_GRID, width=1),
    ),
    yaxis='y2',
    text=[f'{p:+.1f}%' for p in _pcts],
    textposition='outside',
    textfont=dict(color=BT_TEXT, size=10),
))

# Win-rate indicator bars (foreground)
_wr_values = [100.0 if c else 0.0 for c in _corrects]
fig_wr.add_trace(go.Bar(
    name='Model Accuracy per Cycle',
    x=_cycles,
    y=_wr_values,
    marker=dict(color=_bar_colors, line=dict(color=BT_GRID, width=1)),
    yaxis='y1',
    text=_wr_labels,
    textposition='inside',
    textfont=dict(color=BT_TEXT, size=11, family='Arial Black'),
    opacity=0.85,
))

# Annotation: rec → actual per cycle
for _i, (_c, _r, _a) in enumerate(zip(_cycles, _recs, _actuals)):
    fig_wr.add_annotation(
        x=_c, y=-12,
        text=f'{_r}<br>→ {_a}',
        showarrow=False,
        font=dict(size=9, color=BT_GOLD),
        yref='y1',
        align='center',
    )

# Overall win rate line
fig_wr.add_hline(
    y=_win_rate,
    line_dash='dash', line_color=BT_GOLD, line_width=2,
    annotation_text=f' System Win Rate: {_win_rate}%',
    annotation_font_color=BT_GOLD,
    annotation_position='right',
    yref='y1',
)

fig_wr.update_layout(
    title=dict(text='Recommendation Win-Rate per Market Cycle (PASS = correct call, FAIL = incorrect)',
               font=dict(color=BT_TEXT, size=15)),
    plot_bgcolor=BT_BG, paper_bgcolor=BT_BG,
    font=dict(color=BT_TEXT),
    barmode='overlay',
    xaxis=dict(
        ticktext=[c.replace(' ', '<br>') for c in _cycles],
        tickvals=_cycles,
        tickfont=dict(color=BT_TEXT, size=10),
        gridcolor=BT_GRID, showgrid=False,
    ),
    yaxis=dict(
        title='Win Rate % (0 = Fail, 100 = Pass)',
        range=[-20, 130],
        gridcolor=BT_GRID, showgrid=True,
        tickfont=dict(color=BT_TEXT),
        titlefont=dict(color=BT_TEXT),
    ),
    yaxis2=dict(
        title='HPI Price Change %',
        overlaying='y',
        side='right',
        range=[-70, 130],
        gridcolor=BT_GRID, showgrid=False,
        tickfont=dict(color=BT_TEXT, size=9),
        titlefont=dict(color=BT_TEXT),
        zeroline=True, zerolinecolor=BT_GRID,
    ),
    legend=dict(bgcolor='rgba(29,29,32,0.8)', bordercolor=BT_GRID, font=dict(color=BT_TEXT)),
    height=520,
    margin=dict(l=70, r=80, t=80, b=100),
)

fig_wr.show()
print("✅ Chart 3 rendered: Win-Rate per Cycle")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY PRINT
# ─────────────────────────────────────────────────────────────────────────────
print()
print(_DIV)
print('📋  PER-CYCLE PASS/FAIL SUMMARY')
print(_DIV)
print(f"{'#':<3} {'Cycle':<32} {'Predicted':<12} {'Actual':<8} {'ΔHPI':>8}  {'Result':>8}")
print('-' * 80)
for _i, _row in backtest_results_df.iterrows():
    _flag = '✅ PASS' if _row['correct'] else '❌ FAIL'
    print(f"{_i+1:<3} {_row['cycle_name']:<32} {_row['recommendation_generated']:<12} "
          f"{_row['actual_outcome']:<8} {_row['price_change_pct']:>+7.1f}%  {_flag}")
print('-' * 80)
print(f"\n  🏆  Best cycle  : {_best[0] if _best else 'N/A'}  (model nailed it)")
print(f"  ⚠️   Worst cycle : {_worst[-1] if _worst else 'N/A'}  (most misleading signal)")
print(f"  📊  Overall Accuracy: {_win_rate}%  ({_n_correct}/{_n_total} correct)")
print(_DIV)
