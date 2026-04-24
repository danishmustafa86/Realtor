
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

# ─────────────────────────────────────────────
# LOAD CSV FILES FROM CANVAS FILESYSTEM
# Standard FRED CSVs: first column = date, second column = value
# ─────────────────────────────────────────────
def _load_fred_csv(filename: str, col_name: str) -> pd.Series:
    """Load a FRED CSV, auto-detect date column, return named Series."""
    _df = pd.read_csv(filename)
    _df.columns = [c.strip() for c in _df.columns]

    # First column is always the date column in FRED files
    _date_col  = _df.columns[0]
    _value_col = _df.columns[1]

    _df[_date_col]  = pd.to_datetime(_df[_date_col], errors='coerce')
    _df[_value_col] = pd.to_numeric(_df[_value_col], errors='coerce')  # '.' → NaN

    _df = _df.rename(columns={_date_col: 'date', _value_col: col_name})
    _df = _df.dropna(subset=['date'])
    return _df.set_index('date')[col_name]

print("📂 Loading CSV files from canvas filesystem …")

_s_mortgage = _load_fred_csv('MORTGAGE30US.csv', 'MORTGAGE30US')
_s_houst    = _load_fred_csv('HOUST.csv',        'HOUST')
_s_cpi      = _load_fred_csv('CPIAUCSL.csv',     'CPIAUCSL')

print(f"   MORTGAGE30US : {len(_s_mortgage)} rows  ({_s_mortgage.index.min().date()} → {_s_mortgage.index.max().date()})")
print(f"   HOUST        : {len(_s_houst)} rows  ({_s_houst.index.min().date()} → {_s_houst.index.max().date()})")
print(f"   CPIAUCSL     : {len(_s_cpi)} rows  ({_s_cpi.index.min().date()} → {_s_cpi.index.max().date()})")

# ─────────────────────────────────────────────
# MERGE → LAST 5 YEARS, monthly frequency
# ─────────────────────────────────────────────
_combined = pd.concat([_s_mortgage, _s_houst, _s_cpi], axis=1)
_combined = _combined.sort_index()

# Filter to last 5 years
_end   = _combined.index.max()
_start = _end - timedelta(days=5 * 365)
_combined = _combined[_combined.index >= _start]

# Resample to monthly (mortgage is weekly; HOUST/CPI are monthly)
_combined = (
    _combined
    .resample('MS')         # month-start
    .mean()                 # average weekly readings within each month
    .ffill()                # forward-fill sparse months
)
_combined.dropna(inplace=True)

macro_data = _combined.reset_index().rename(columns={'index': 'date'})
# Ensure date column is named 'date'
if 'date' not in macro_data.columns:
    macro_data = macro_data.rename(columns={macro_data.columns[0]: 'date'})

macro_data['date'] = pd.to_datetime(macro_data['date'])
macro_data.sort_values('date', inplace=True)
macro_data.reset_index(drop=True, inplace=True)

print(f"\n📊 Merged macro_data shape : {macro_data.shape}")
print(f"   Date range             : {macro_data['date'].min().date()} → {macro_data['date'].max().date()}")
print(f"   Columns                : {list(macro_data.columns)}")
print(macro_data.tail(5).to_string(index=False))

# ─────────────────────────────────────────────
# MARKET HEAT INDEX  (1 – 10)
# ─────────────────────────────────────────────
def clamp(val, lo: float = 1.0, hi: float = 10.0) -> float:
    return float(max(lo, min(hi, val)))


def compute_heat_index(df: pd.DataFrame) -> float:
    """
    Composite Market Heat Index (1–10), equal-weight:
      1. Mortgage Pressure  — current rate vs 5-yr range  (high = hot)
      2. Supply Pressure    — housing starts, inverted     (low = hot)
      3. Inflation Momentum — YoY CPI %: 0% → 1, ≥5% → 10
    """
    _mort  = df['MORTGAGE30US'].dropna()
    _houst = df['HOUST'].dropna()
    _cpi   = df['CPIAUCSL'].dropna()

    # 1. Mortgage
    _rng = max(_mort.max() - _mort.min(), 0.01)
    mort_score = clamp(1 + 9 * (_mort.iloc[-1] - _mort.min()) / _rng)

    # 2. Supply (inverted)
    _hrng = max(_houst.max() - _houst.min(), 1.0)
    houst_score = clamp(1 + 9 * (1 - (_houst.iloc[-1] - _houst.min()) / _hrng))

    # 3. Inflation YoY
    _lb  = min(13, len(_cpi) - 1)
    _yoy = (_cpi.iloc[-1] - _cpi.iloc[-1 - _lb]) / _cpi.iloc[-1 - _lb] * 100
    inf_score = clamp(1 + 9 * (_yoy / 5.0))

    composite = round((mort_score + houst_score + inf_score) / 3, 2)

    print(f"\n📐 Market Heat Index Components:")
    print(f"   Mortgage Pressure  : {mort_score:.2f}/10  (latest: {_mort.iloc[-1]:.2f}%)")
    print(f"   Supply Pressure    : {houst_score:.2f}/10  (latest: {_houst.iloc[-1]:.0f}K starts)")
    print(f"   Inflation Momentum : {inf_score:.2f}/10  (YoY CPI: {_yoy:.2f}%)")
    print(f"   ────────────────────────────────────")
    print(f"   🌡️  MARKET HEAT INDEX: {composite}/10")
    _label = (
        "🟢 Cool – Buyer-Friendly"  if composite < 4 else
        "🟡 Warm – Balanced Market" if composite < 6 else
        "🟠 Hot – Seller's Market"  if composite < 8 else
        "🔴 Very Hot – High Stress"
    )
    print(f"   Condition: {_label}")
    return composite


market_heat_index = compute_heat_index(macro_data)

# ─────────────────────────────────────────────
# ZERVE DESIGN TOKENS
# ─────────────────────────────────────────────
BG       = '#1D1D20'
TEXT     = '#fbfbff'
GRID     = '#2e2e33'
C_BLUE   = '#A1C9F4'
C_ORANGE = '#FFB482'
C_GREEN  = '#8DE5A1'
C_GOLD   = '#ffd400'


def _ax(title: str = '', suffix: str = '') -> dict:
    d = dict(
        gridcolor=GRID, linecolor=GRID, tickcolor=TEXT,
        title_font=dict(color=TEXT), tickfont=dict(color=TEXT),
    )
    if title:  d['title']      = title
    if suffix: d['ticksuffix'] = suffix
    return d


_SHARED = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(family='Inter, Arial, sans-serif', color=TEXT, size=12),
    margin=dict(l=65, r=45, t=75, b=60),
    hovermode='x unified',
)
_LEGEND = dict(bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT))

# ─────────────────────────────────────────────
# CHART 1 — 30-Year Mortgage Rates
# ─────────────────────────────────────────────
_last_mort = macro_data['MORTGAGE30US'].iloc[-1]

mortgage_chart = go.Figure()
mortgage_chart.add_trace(go.Scatter(
    x=macro_data['date'], y=macro_data['MORTGAGE30US'],
    name='30Y Fixed Rate',
    line=dict(color=C_BLUE, width=2.5),
    fill='tozeroy', fillcolor='rgba(161,201,244,0.10)',
    hovertemplate='%{x|%b %Y}: <b>%{y:.2f}%</b><extra></extra>',
))
mortgage_chart.add_annotation(
    x=macro_data['date'].iloc[-1], y=_last_mort,
    text=f"<b>{_last_mort:.2f}%</b>",
    showarrow=True, arrowhead=2, arrowcolor=C_GOLD,
    font=dict(color=C_GOLD, size=13), bgcolor=BG, bordercolor=C_GOLD,
)
mortgage_chart.update_layout(
    **_SHARED,
    title=dict(text='📈 30-Year Fixed Mortgage Rate (5-Year Trend)', font=dict(size=18, color=TEXT), x=0.02),
    legend=_LEGEND, xaxis=_ax(), yaxis=_ax('Rate (%)', suffix='%'),
)
mortgage_chart.show()
print("✅ Chart 1 — Mortgage Rate rendered")

# ─────────────────────────────────────────────
# CHART 2 — Housing Starts
# ─────────────────────────────────────────────
macro_data['HOUST_MA3'] = macro_data['HOUST'].rolling(3, min_periods=1).mean()
_last_houst = macro_data['HOUST'].iloc[-1]

housing_chart = go.Figure()
housing_chart.add_trace(go.Bar(
    x=macro_data['date'], y=macro_data['HOUST'],
    name='Monthly Starts',
    marker_color='rgba(141,229,161,0.30)',
    hovertemplate='%{x|%b %Y}: <b>%{y:.0f}K</b><extra></extra>',
))
housing_chart.add_trace(go.Scatter(
    x=macro_data['date'], y=macro_data['HOUST_MA3'],
    name='3-Month Avg',
    line=dict(color=C_GREEN, width=2.5),
    hovertemplate='%{x|%b %Y}: <b>%{y:.0f}K</b><extra></extra>',
))
housing_chart.add_annotation(
    x=macro_data['date'].iloc[-1], y=_last_houst,
    text=f"<b>{_last_houst:.0f}K</b>",
    showarrow=True, arrowhead=2, arrowcolor=C_GOLD,
    font=dict(color=C_GOLD, size=13), bgcolor=BG, bordercolor=C_GOLD,
)
housing_chart.update_layout(
    **_SHARED,
    title=dict(text='🏗️ Housing Starts — New Residential Construction', font=dict(size=18, color=TEXT), x=0.02),
    legend=_LEGEND, xaxis=_ax(), yaxis=_ax('Starts (Thousands)'), barmode='overlay',
)
housing_chart.show()
print("✅ Chart 2 — Housing Starts rendered")

# ─────────────────────────────────────────────
# CHART 3 — CPI Inflation (dual axis)
# ─────────────────────────────────────────────
macro_data['CPI_YOY'] = macro_data['CPIAUCSL'].pct_change(12) * 100

cpi_chart = go.Figure()
cpi_chart.add_trace(go.Scatter(
    x=macro_data['date'], y=macro_data['CPIAUCSL'],
    name='CPI Index', yaxis='y',
    line=dict(color=C_ORANGE, width=2.5),
    hovertemplate='%{x|%b %Y}: Index <b>%{y:.1f}</b><extra></extra>',
))
cpi_chart.add_trace(go.Bar(
    x=macro_data['date'], y=macro_data['CPI_YOY'],
    name='YoY Inflation %', yaxis='y2',
    marker_color=[
        'rgba(240,68,56,0.55)' if (pd.notna(v) and v >= 0) else 'rgba(141,229,161,0.55)'
        for v in macro_data['CPI_YOY']
    ],
    hovertemplate='%{x|%b %Y}: YoY <b>%{y:.2f}%</b><extra></extra>',
))
cpi_chart.add_hline(
    y=2, yref='y2',
    line=dict(color=C_GOLD, dash='dot', width=1.5),
    annotation_text='Fed 2% Target',
    annotation_position='top left',
    annotation_font=dict(color=C_GOLD, size=11),
)
cpi_chart.update_layout(
    **_SHARED,
    title=dict(text='💰 CPI Inflation — Index & Year-Over-Year Change', font=dict(size=18, color=TEXT), x=0.02),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT), x=0.01, y=0.99),
    xaxis=_ax(),
    yaxis=_ax('CPI Index'),
    yaxis2=dict(
        title='YoY Change (%)', title_font=dict(color=TEXT), tickfont=dict(color=TEXT),
        overlaying='y', side='right', ticksuffix='%',
        gridcolor='rgba(0,0,0,0)', linecolor=GRID, tickcolor=TEXT,
    ),
    barmode='overlay',
)
cpi_chart.show()
print("✅ Chart 3 — CPI Inflation rendered")

# ─────────────────────────────────────────────
# HEAT GAUGE
# ─────────────────────────────────────────────
heat_gauge = go.Figure(go.Indicator(
    mode='gauge+number',
    value=market_heat_index,
    title=dict(text='🌡️ Market Heat Index', font=dict(color=TEXT, size=18)),
    number=dict(font=dict(color=C_GOLD, size=52), suffix='/10'),
    gauge=dict(
        axis=dict(range=[1, 10], tickcolor=TEXT, tickfont=dict(color=TEXT)),
        bar=dict(color=C_GOLD, thickness=0.28),
        bgcolor='#2e2e33',
        borderwidth=0,
        steps=[
            dict(range=[1, 4],  color='rgba(141,229,161,0.30)'),
            dict(range=[4, 6],  color='rgba(255,180,130,0.30)'),
            dict(range=[6, 8],  color='rgba(255,159,155,0.30)'),
            dict(range=[8, 10], color='rgba(240,68,56,0.40)'),
        ],
        threshold=dict(line=dict(color='#f04438', width=3), thickness=0.75, value=7),
    ),
))
heat_gauge.update_layout(
    paper_bgcolor=BG,
    font=dict(family='Inter, Arial, sans-serif', color=TEXT),
    margin=dict(l=40, r=40, t=80, b=40),
    height=360,
)
heat_gauge.show()

print(f"\n✅ All 4 charts rendered successfully.")
print(f"   market_heat_index = {market_heat_index}/10")
print(f"   macro_data shape  = {macro_data.shape}")
print(f"   date range        : {macro_data['date'].min().date()} → {macro_data['date'].max().date()}")
print(f"   columns           : {list(macro_data.columns)}")
