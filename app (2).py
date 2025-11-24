import dash
from dash import html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta

# =========================================================
# 1. Helper Functions 
# =========================================================
def SMA(values, n):
    return pd.Series(values).rolling(n).mean()

def RSI(values, n=14):
    delta = pd.Series(values).diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    rs = gain.rolling(n).mean() / loss.rolling(n).mean()
    rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)
    return 100 - (100 / (1 + rs))

def BBANDS(values, n=20, k=2):
    ma = pd.Series(values).rolling(n).mean()
    std = pd.Series(values).rolling(n).std()
    upper = ma + k * std
    lower = ma - k * std
    return ma, upper, lower

def MOMENTUM(values, n):
    return (pd.Series(values) / pd.Series(values).shift(n)) - 1

def calculate_var(equity_curve):
    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0: return 0
    return returns.quantile(0.05)

def calculate_benchmark_metrics(equity_curve):
    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0: return 0, 0, 0, 0, 0
    
    var_95 = returns.quantile(0.05)
    annual_return = returns.mean() * 252
    annual_std = returns.std() * np.sqrt(252)
    sharpe = (annual_return / annual_std) if annual_std != 0 else 0
    
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (annual_return / downside_std) if downside_std != 0 else 0
    
    cum_max = equity_curve.cummax()
    drawdown = (equity_curve - cum_max) / cum_max
    max_dd = drawdown.min() * 100
    
    win_rate = (len(returns[returns > 0]) / len(returns)) * 100
    
    return sharpe, sortino, max_dd, win_rate, var_95

# =========================================================
# 2. Strategy Classes
# =========================================================
class SmaCross(Strategy):
    n1 = 10
    n2 = 20
    def init(self):
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
    def next(self):
        if crossover(self.sma1, self.sma2): self.buy(size=0.95)
        elif crossover(self.sma2, self.sma1): self.position.close()

class RsiOscillator(Strategy):
    upper = 70
    lower = 30
    def init(self):
        self.rsi = self.I(RSI, self.data.Close, 14)
    def next(self):
        if self.rsi[-1] < self.lower: self.buy(size=0.95)
        elif self.rsi[-1] > self.upper: self.position.close()

class BollingerBandStrategy(Strategy):
    n = 20
    k = 2.0
    def init(self):
        self.mid, self.upper, self.lower = self.I(BBANDS, self.data.Close, self.n, self.k)
    def next(self):
        price = self.data.Close[-1]
        if price < self.lower[-1] and not self.position: self.buy(size=0.95)
        elif price > self.mid[-1] and self.position: self.position.close()

class MomentumStrategy(Strategy):
    lookback = 20
    threshold = 0.0 
    def init(self):
        self.mom = self.I(MOMENTUM, self.data.Close, self.lookback)
    def next(self):
        if self.mom[-1] > self.threshold and not self.position: self.buy(size=0.95)
        elif self.mom[-1] < 0 and self.position: self.position.close()

class XgbStrategy(Strategy):
    def init(self): pass
    def next(self): pass

class RnnStrategy(Strategy):
    def init(self): pass
    def next(self): pass

# =========================================================
# 3. Data Loading 
# =========================================================
df_local = pd.DataFrame()
local_data_exists = False

try:
    df_local = pd.read_parquet("master_stock_data.parquet")
    df_local["Date"] = pd.to_datetime(df_local["Date"])
    local_data_exists = True
except:
    try:
        df_local = pd.read_csv("master_stock_data.csv")
        df_local["Date"] = pd.to_datetime(df_local["Date"])
        local_data_exists = True
    except:
        pass

def get_stock_data(ticker, start, end):
    """Fetch data from Local or yfinance"""
    data = pd.DataFrame()
    if local_data_exists and ticker in df_local["Ticker"].values:
        mask = (df_local["Ticker"] == ticker) & (df_local["Date"] >= start) & (df_local["Date"] <= end)
        data = df_local.loc[mask].copy().set_index("Date")
    
    if data.empty:
        try:
            # Fix for yfinance multi-level column index
            yf_data = yf.download(ticker, start=start, end=end, progress=False)
            if not yf_data.empty:
                if isinstance(yf_data.columns, pd.MultiIndex):
                    yf_data.columns = yf_data.columns.get_level_values(0)
                data = yf_data
        except: return pd.DataFrame()

    if data.empty: return pd.DataFrame()
    
    req_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in req_cols:
        if col not in data.columns and "Close" in data.columns:
            data[col] = data["Close"]
            
    return data.dropna(subset=["Close"])[req_cols]

# Init Variables
if local_data_exists and not df_local.empty:
    all_tickers = sorted(df_local["Ticker"].unique())
    min_date = df_local['Date'].min()
    max_date = df_local['Date'].max()
else:
    all_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'SPY', 'QQQ', 'IWM']
    min_date = '2000-01-01'
    max_date = pd.Timestamp.today().strftime('%Y-%m-%d')

STRATEGIES = {
    "1. Benchmark: Buy & Hold": None,
    "2. Traditional: SMA Crossover": SmaCross,
    "3. Traditional: RSI Mean Reversion": RsiOscillator,
    "4. Traditional: Bollinger Bands": BollingerBandStrategy,
    "5. Traditional: Absolute Momentum": MomentumStrategy,
    "6. ML: XGBoost": XgbStrategy,
    "7. DL: RNN (Deep Learning)": RnnStrategy
}

# =========================================================
# 4. App Layout - Professional Dark Theme
# =========================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE, "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"])
server = app.server

# Custom CSS for that "Premium Financial" look
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>QuantPro Analytics</title>
        {%favicon%}
        {%css%}
        <style>
            body { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #000; }
            .card { border: 1px solid #333; background-color: #222; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
            .card-header { background-color: #2c2c2c; border-bottom: 1px solid #444; font-weight: 600; color: #e0e0e0; letter-spacing: 0.5px; }
            .metric-card { background: linear-gradient(145deg, #2a2a2a, #222); border-radius: 8px; border: 1px solid #333; }
            .metric-val { font-family: 'Roboto Mono', monospace; font-weight: 700; }
            .nav-tabs .nav-link { color: #aaa; border: none; }
            .nav-tabs .nav-link.active { background-color: #222; border-color: #333; border-bottom-color: #222; color: #00bc8c; font-weight: bold; }
            .range-btn { font-size: 0.8rem; font-weight: 600; margin-right: 5px; border-radius: 20px; }
            /* Custom scrollbar */
            ::-webkit-scrollbar { width: 8px; }
            ::-webkit-scrollbar-track { background: #1a1a1a; }
            ::-webkit-scrollbar-thumb { background: #444; border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: #555; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

def InfoIcon(id_name, text):
    return html.Span([
        html.I(className="fas fa-info-circle", id=id_name, style={"cursor": "pointer", "color": "#17a2b8", "marginLeft": "8px", "fontSize": "0.9rem"}),
        dbc.Tooltip(text, target=id_name, placement="top", style={"maxWidth": "300px"})
    ])

# --- Reusable Components ---
def MetricCard(title, id_name, color_class="text-white"):
    return dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.H6(title, className="text-muted text-uppercase", style={"fontSize": "0.75rem", "letterSpacing": "1px"}),
                html.H3("0.00", id=id_name, className=f"metric-val {color_class} mb-0")
            ], className="p-3")
        ], className="metric-card h-100"), 
        width=6, lg=2, className="mb-2" # Responsive width
    )

# Sidebar Controls
control_panel = dbc.Card([
    dbc.CardHeader([html.I(className="fas fa-sliders-h me-2"), "Strategy Config"]),
    dbc.CardBody([
        html.Label("Asset Ticker", className="text-light fw-bold mt-2"),
        dcc.Dropdown(id="ticker-drop", options=all_tickers, value=all_tickers[0] if all_tickers else None, clearable=False, className="mb-3", style={"color": "#000"}),
        
        html.Label("Strategy Model", className="text-light fw-bold"),
        dcc.Dropdown(id="strategy-drop", options=[{"label": k, "value": k} for k in STRATEGIES.keys()], value="2. Traditional: SMA Crossover", clearable=False, className="mb-4", style={"color": "#000"}),
        
        html.Hr(className="border-secondary"),
        
        # Dynamic Parameters
        html.Div(id="sma-params", children=[
            html.Label(["Fast MA Period", InfoIcon("i-n1", "Short-term moving average")]),
            dcc.Slider(id="n1-slider", min=5, max=50, step=1, value=10, marks={10:'10', 30:'30', 50:'50'}, tooltip={"always_visible": False, "placement": "bottom"}),
            html.Br(),
            html.Label(["Slow MA Period", InfoIcon("i-n2", "Long-term moving average")]),
            dcc.Slider(id="n2-slider", min=20, max=100, step=5, value=20, marks={20:'20', 60:'60', 100:'100'}, tooltip={"always_visible": False, "placement": "bottom"}),
        ], style={"display": "none"}),

        html.Div(id="boll-params", children=[
            html.Label(["Lookback Window", InfoIcon("i-bbn", "Period for MA Calculation")]),
            dcc.Slider(id="bb-n-slider", min=10, max=50, step=1, value=20, marks={10:'10', 30:'30', 50:'50'}, tooltip={"always_visible": False}),
            html.Br(),
            html.Label(["Std Dev Multiplier", InfoIcon("i-bbk", "Width of the bands")]),
            dcc.Slider(id="bb-k-slider", min=1.0, max=3.0, step=0.1, value=2.0, marks={1:'1', 2:'2', 3:'3'}, tooltip={"always_visible": False}),
        ], style={"display": "none"}),

        html.Div(id="mom-params", children=[
            html.Label(["Lookback Days", InfoIcon("i-lb", "Period to compare price against")]),
            dcc.Slider(id="lb-slider", min=5, max=126, step=1, value=20, marks={5:'1W', 21:'1M', 63:'3M'}, tooltip={"always_visible": False}),
        ], style={"display": "none"}),
    ])
], className="border-0 h-100")

# Main Layout Structure
app.layout = dbc.Container([
    # --- Header ---
    dbc.Row([
        dbc.Col(html.Div([
            html.H2([html.I(className="fas fa-chart-line text-success me-3"), "QUANT TRADING DASHBOARD"], className="text-light fw-bold mb-0"),
            html.P("Advanced Algorithmic Trading Analysis System", className="text-muted small mb-0")
        ]), width=12, className="py-3 border-bottom border-secondary mb-4")
    ]),

    # --- Top Level Metrics Bar (Compact & Visible) ---
    dbc.Row([
        MetricCard("Total Return", "res-return", "text-warning"),
        MetricCard("Sharpe Ratio", "res-sharpe", "text-success"),
        MetricCard("Sortino Ratio", "res-sortino", "text-info"),
        MetricCard("Max Drawdown", "res-mdd", "text-danger"),
        MetricCard("Win Rate", "res-win", "text-primary"),
        MetricCard("Profit Factor", "res-pf", "text-light"),
    ], className="mb-4 g-2"),

    dbc.Row([
        # --- Sidebar Column ---
        dbc.Col(control_panel, width=12, lg=3, className="mb-4"),

        # --- Main Content Column ---
        dbc.Col([
            # Time Range Controls
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Date Range:", className="fw-bold text-light me-2"),
                            # Improved Date Picker
                            dcc.DatePickerRange(
                                id='date-picker',
                                min_date_allowed=min_date,
                                max_date_allowed=max_date,
                                start_date=(pd.Timestamp(max_date) - pd.DateOffset(years=5)).strftime('%Y-%m-%d'),
                                end_date=max_date,
                                style={"fontSize": "0.8rem"},
                                className="d-inline-block"
                            )
                        ], width=12, md=6, className="mb-2 mb-md-0"),
                        dbc.Col([
                            html.Div([
                                dbc.Button("1Y", id="btn-1y", color="dark", size="sm", className="range-btn border-secondary"),
                                dbc.Button("3Y", id="btn-3y", color="dark", size="sm", className="range-btn border-secondary"),
                                dbc.Button("5Y", id="btn-5y", color="dark", size="sm", className="range-btn border-secondary"),
                                dbc.Button("10Y", id="btn-10y", color="dark", size="sm", className="range-btn border-secondary"),
                                dbc.Button("MAX", id="btn-max", color="dark", size="sm", className="range-btn border-secondary"),
                            ], className="d-flex justify-content-md-end")
                        ], width=12, md=6)
                    ], align="center")
                ], className="py-2 px-3")
            ], className="mb-3"),

            # Tabs and Large Charts
            dbc.Tabs([
                dbc.Tab(label="Equity Curve & Signals", tab_id="tab-perf", children=[
                    dcc.Loading(dcc.Graph(id="graph-equity", style={"height": "600px"}), type="graph", color="#00bc8c")
                ]),
                dbc.Tab(label="Risk Analysis", tab_id="tab-risk", children=[
                    dcc.Loading(dcc.Graph(id="graph-risk", style={"height": "600px"}), type="graph", color="#00bc8c")
                ]),
                dbc.Tab(label="Monte Carlo Sim", tab_id="tab-monte", children=[
                    dbc.CardBody([
                        dbc.Button("Run Simulation (50 Paths)", id="btn-monte", color="primary", className="mb-3"),
                        dcc.Loading(dcc.Graph(id="graph-monte", style={"height": "550px"}), type="graph", color="#00bc8c")
                    ])
                ]),
                dbc.Tab(label="Strategy Optimization", tab_id="tab-opt", children=[
                    dbc.CardBody([
                        dbc.Button("Run Grid Search Optimization", id="btn-opt", color="danger", className="mb-3"),
                        dcc.Loading(dcc.Graph(id="graph-heat", style={"height": "550px"}), type="graph", color="#00bc8c")
                    ])
                ]),
            ], active_tab="tab-perf", className="nav-tabs")
            
        ], width=12, lg=9)
    ])
], fluid=True, className="pb-5")

# =========================================================
# 5. Callbacks (Logic)
# =========================================================

# 1. Time Range Buttons Logic
@app.callback(
    [Output('date-picker', 'start_date'), Output('date-picker', 'end_date')],
    [Input('btn-1y', 'n_clicks'), Input('btn-3y', 'n_clicks'), 
     Input('btn-5y', 'n_clicks'), Input('btn-10y', 'n_clicks'), Input('btn-max', 'n_clicks')],
    [State('date-picker', 'end_date')]
)
def update_date_range(b1, b3, b5, b10, bmax, current_end):
    triggered = ctx.triggered_id
    end_dt = pd.Timestamp(max_date) # Default to max available
    
    if triggered == 'btn-1y': start_dt = end_dt - pd.DateOffset(years=1)
    elif triggered == 'btn-3y': start_dt = end_dt - pd.DateOffset(years=3)
    elif triggered == 'btn-5y': start_dt = end_dt - pd.DateOffset(years=5)
    elif triggered == 'btn-10y': start_dt = end_dt - pd.DateOffset(years=10)
    elif triggered == 'btn-max': start_dt = pd.Timestamp(min_date)
    else: return dash.no_update, dash.no_update # No change
    
    return start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')

# 2. Parameter Visibility
@app.callback(
    [Output("sma-params", "style"), Output("boll-params", "style"), Output("mom-params", "style")],
    Input("strategy-drop", "value")
)
def toggle_params(strategy):
    show, hide = {"display": "block"}, {"display": "none"}
    if "SMA" in strategy: return show, hide, hide
    if "Bollinger" in strategy: return hide, show, hide
    if "Momentum" in strategy: return hide, hide, show
    return hide, hide, hide

# 3. Main Backtest Engine
@app.callback(
    [Output("graph-equity", "figure"), Output("graph-risk", "figure"),
     Output("res-return", "children"), Output("res-sharpe", "children"), 
     Output("res-sortino", "children"), Output("res-mdd", "children"), 
     Output("res-win", "children"), Output("res-pf", "children")],
    [Input("ticker-drop", "value"), Input("strategy-drop", "value"),
     Input("date-picker", "start_date"), Input("date-picker", "end_date"),
     Input("n1-slider", "value"), Input("n2-slider", "value"),
     Input("bb-n-slider", "value"), Input("bb-k-slider", "value"),
     Input("lb-slider", "value")]
)
def run_backtest(ticker, strategy_name, start, end, n1, n2, bb_n, bb_k, lb):
    if not ticker: return go.Figure(), go.Figure(), *["-"]*6
    
    data = get_stock_data(ticker, start, end)
    if data.empty or len(data) < 10:
        return go.Figure(), go.Figure(), "No Data", "N/A", "N/A", "N/A", "N/A", "N/A"

    StrategyClass = STRATEGIES[strategy_name]
    initial_cash = 10000
    
    # Benchmark (Buy & Hold) Logic
    benchmark_equity = (data["Close"] / data["Close"].iloc[0]) * initial_cash
    
    equity = pd.Series()
    trades_df = None
    strategy_failed = False
    
    # Metrics defaults
    ret, sharpe, sortino, mdd, win_rate, pf, var = 0,0,0,0,0,0,0

    if StrategyClass is None:
        equity = benchmark_equity
        sharpe, sortino, mdd, win_rate, var = calculate_benchmark_metrics(equity)
        ret = ((equity.iloc[-1] - initial_cash) / initial_cash) * 100
        pf = "N/A"
    else:
        # Set params
        if "SMA" in strategy_name: SmaCross.n1, SmaCross.n2 = n1, n2
        elif "Bollinger" in strategy_name: BollingerBandStrategy.n, BollingerBandStrategy.k = bb_n, bb_k
        elif "Momentum" in strategy_name: MomentumStrategy.lookback = lb
        
        try:
            bt = Backtest(data, StrategyClass, cash=initial_cash, commission=.001)
            stats = bt.run()
            equity = stats["_equity_curve"]["Equity"]
            ret = stats["Return [%]"]
            sharpe = stats["Sharpe Ratio"]
            sortino = stats["Sortino Ratio"]
            mdd = stats["Max. Drawdown [%]"]
            win_rate = stats["Win Rate [%]"]
            pf = stats["Profit Factor"]
            trades_df = stats['_trades']
            var = calculate_var(equity)
        except:
            strategy_failed = True
            equity = benchmark_equity
            sharpe, sortino, mdd, win_rate, var = calculate_benchmark_metrics(equity)
            ret = ((equity.iloc[-1] - initial_cash) / initial_cash) * 100
            pf = "N/A"

    # --- Enhanced Charts ---
    # 1. Equity Curve
    fig_eq = go.Figure()
    
    # Strategy Line (Green/Red)
    line_color = '#00bc8c' if not strategy_failed else '#e74c3c'
    name = 'Active Strategy' if not strategy_failed else 'Strategy Failed (Showing Benchmark)'
    fig_eq.add_trace(go.Scatter(x=equity.index, y=equity, mode='lines', name=name, line=dict(color=line_color, width=2)))
    
    # Benchmark Line (Gray)
    if StrategyClass is not None and not strategy_failed:
        fig_eq.add_trace(go.Scatter(x=benchmark_equity.index, y=benchmark_equity, mode='lines', name='Buy & Hold', line=dict(color='#6c757d', width=1.5, dash='dash')))

    # Trade Markers
    if trades_df is not None and not trades_df.empty:
        # Backtesting.py uses CamelCase EntryTime/ExitTime
        # Using standard markers for clarity
        entry_t = trades_df['EntryTime'] if 'EntryTime' in trades_df.columns else []
        exit_t = trades_df['ExitTime'] if 'ExitTime' in trades_df.columns else []
        
        if len(entry_t) > 0:
            fig_eq.add_trace(go.Scatter(x=entry_t, y=trades_df['EntryPrice'], mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', size=10, color='#00ff00')))
        if len(exit_t) > 0:
            fig_eq.add_trace(go.Scatter(x=exit_t, y=trades_df['ExitPrice'], mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', size=10, color='#ff0000')))

    # Layout Polish
    fig_eq.update_layout(
        title=dict(text=f"Equity Curve Analysis ({ticker})", font=dict(size=18, color="#e0e0e0")),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.03)',
        xaxis=dict(title=None, showgrid=True, gridcolor='#333'),
        yaxis=dict(title="Account Value ($)", showgrid=True, gridcolor='#333'),
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    # 2. Risk Dist
    daily_rets = equity.pct_change().dropna()
    fig_risk = go.Figure(data=[go.Histogram(x=daily_rets, nbinsx=60, marker_color='#375a7f', opacity=0.8)])
    fig_risk.add_vline(x=var, line_color="#f39c12", line_dash="dash", annotation_text="VaR 95%", annotation_font_color="#f39c12")
    fig_risk.update_layout(
        title="Daily Returns Distribution",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.03)',
        xaxis_title="Daily Return",
        yaxis_title="Frequency",
        showlegend=False
    )

    # Formatting Helper
    fmt = lambda x: f"{x:.2f}" if isinstance(x, (float, int)) else str(x)
    fmt_pct = lambda x: f"{x:.2f}%" if isinstance(x, (float, int)) else str(x)

    return fig_eq, fig_risk, fmt_pct(ret), fmt(sharpe), fmt(sortino), fmt_pct(mdd), fmt_pct(win_rate), fmt(pf)

# 4. Monte Carlo
@app.callback(Output("graph-monte", "figure"), [Input("btn-monte", "n_clicks")], [State("ticker-drop", "value"), State("date-picker", "start_date"), State("date-picker", "end_date")])
def run_monte(n, ticker, start, end):
    if not n: return go.Figure()
    data = get_stock_data(ticker, start, end)
    if data.empty: return go.Figure()
    
    last = data["Close"].iloc[-1]
    vol = data["Close"].pct_change().std()
    
    fig = go.Figure()
    for _ in range(50): # 50 Paths
        prices = [last]
        for _ in range(60):
            prices.append(prices[-1] * (1 + np.random.normal(0, vol)))
        fig.add_trace(go.Scatter(y=prices, mode='lines', line=dict(width=1, color='rgba(0,188,140,0.2)'), showlegend=False))
    
    fig.update_layout(title="Monte Carlo Simulation (Next 60 Days)", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.03)')
    return fig

# 5. Optimization
@app.callback(Output("graph-heat", "figure"), [Input("btn-opt", "n_clicks")], [State("ticker-drop", "value"), State("strategy-drop", "value"), State("date-picker", "start_date"), State("date-picker", "end_date")])
def run_opt(n, ticker, strategy, start, end):
    if not n: return go.Figure()
    data = get_stock_data(ticker, start, end)
    if data.empty: return go.Figure()
    
    res, x_ax, y_ax, xl, yl = [], [], [], "", ""
    
    # Optimized range logic for demo speed
    if "SMA" in strategy:
        x_r, y_r = range(20, 80, 10), range(5, 35, 5)
        xl, yl = "Slow MA", "Fast MA"
        x_ax, y_ax = list(x_r), list(y_r)
        for y in y_r:
            row = []
            for x in x_r:
                if y >= x: row.append(0)
                else:
                    SmaCross.n1, SmaCross.n2 = y, x
                    try: row.append(Backtest(data, SmaCross, cash=10000, commission=.001).run()["Return [%]"])
                    except: row.append(0)
            res.append(row)
            
    elif "Bollinger" in strategy:
        x_r, y_r = [1.5, 2.0, 2.5], range(10, 50, 10)
        xl, yl = "StdDev (K)", "Window (N)"
        x_ax, y_ax = x_r, list(y_r)
        for y in y_r:
            row = []
            for x in x_r:
                BollingerBandStrategy.n, BollingerBandStrategy.k = y, x
                try: row.append(Backtest(data, BollingerBandStrategy, cash=10000, commission=.001).run()["Return [%]"])
                except: row.append(0)
            res.append(row)
            
    elif "Momentum" in strategy:
        x_r, y_r = [-0.01, 0.0, 0.01], range(10, 60, 10)
        xl, yl = "Threshold", "Lookback"
        x_ax, y_ax = x_r, list(y_r)
        for y in y_r:
            row = []
            for x in x_r:
                MomentumStrategy.lookback, MomentumStrategy.threshold = y, x
                try: row.append(Backtest(data, MomentumStrategy, cash=10000, commission=.001).run()["Return [%]"])
                except: row.append(0)
            res.append(row)
    else: return go.Figure()

    fig = go.Figure(data=go.Heatmap(z=res, x=x_ax, y=y_ax, colorscale='Viridis', colorbar=dict(title="Return %")))
    fig.update_layout(title=f"Parameter Heatmap: {yl} vs {xl}", xaxis_title=xl, yaxis_title=yl, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
    return fig

if __name__ == "__main__":
    app.run(debug=True)
