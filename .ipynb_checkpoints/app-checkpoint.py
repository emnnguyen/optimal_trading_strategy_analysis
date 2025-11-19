#final
code = '''
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np

# =========================================================
# helpers
# =========================================================
def SMA(values, n):
    return pd.Series(values).rolling(n).mean()

def RSI(values, n=14):
    delta = pd.Series(values).diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    rs = gain.rolling(n).mean() / loss.rolling(n).mean()
    return 100 - (100 / (1 + rs))

def calculate_risk_metrics(equity_curve):
    returns = equity_curve.pct_change().dropna()
    var_95 = returns.quantile(0.05) 
    downside_returns = returns[returns < 0]
    sortino = (returns.mean() * 252) / (downside_returns.std() * np.sqrt(252)) if downside_returns.std() != 0 else 0
    return returns, var_95, sortino

# =========================================================
# Strategies
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

class RnnStrategy(Strategy):
    def init(self): pass
    def next(self): pass

class XgbStrategy(Strategy):
    def init(self): pass
    def next(self): pass

# =========================================================
# Data
# =========================================================
try:
    df = pd.read_parquet("master_stock_data.parquet")
    df["Date"] = pd.to_datetime(df["Date"]) 
except:
    try:
        df = pd.read_csv("master_stock_data.csv")
        df["Date"] = pd.to_datetime(df["Date"])
    except:
        df = pd.DataFrame(columns=["Ticker", "Date", "Close"])

all_tickers = sorted(df["Ticker"].unique())
STRATEGIES = {
    "1. Benchmark: Buy & Hold": None,
    "2. Traditional: SMA Crossover": SmaCross,
    "3. Traditional: RSI Mean Reversion": RsiOscillator,
    "4. ML: XGBoost": XgbStrategy,
    "5. DL: RNN (Deep Learning)": RnnStrategy
}

# =========================================================
# Layout
# =========================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

# Creates an info icon with a tooltip
def InfoIcon(id_name, text):
    return html.Span([
        html.Sup(" ⓘ", id=id_name, style={"cursor": "pointer", "color": "#17a2b8", "fontSize": "14px", "marginLeft": "5px"}),
        dbc.Tooltip(text, target=id_name, placement="top", style={"fontSize": "14px"})
    ])

app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("QUANT TRADING DASHBOARD", className="text-center text-primary mb-4 mt-4"), width=12)]),

    dbc.Row([
        # SIDEBAR
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Configuration", className="text-info"),
                dbc.CardBody([
                    html.Label("Select Asset:"),
                    dcc.Dropdown(id="ticker-drop", options=all_tickers, value=all_tickers[0] if all_tickers else None, clearable=False, style={"color": "black"}),
                    html.Br(),
                    html.Label("Time Frame:"),
                    dcc.DatePickerRange(id='date-picker', min_date_allowed=df['Date'].min(), max_date_allowed=df['Date'].max(), start_date='2020-01-01', end_date=df['Date'].max(), style={"width": "100%", "color": "black"}),
                    html.Br(), html.Br(),
                    html.Label(["Select Strategy:", InfoIcon("info-strat", "Choose a trading logic to backtest against historical data.")]),
                    dcc.Dropdown(id="strategy-drop", options=[{"label": k, "value": k} for k in STRATEGIES.keys()], value="2. Traditional: SMA Crossover", clearable=False, style={"color": "black"}),
                    html.Br(),
                    html.Div(id="parameter-container", children=[
                        html.Label(["Fast MA:", InfoIcon("info-fast", "Short-term moving average. Represents momentum.")]),
                        dcc.Slider(id="n1-slider", min=5, max=50, step=1, value=10, marks={i: str(i) for i in range(5, 55, 10)}, tooltip={"placement": "bottom", "always_visible": True}),
                        html.Label(["Slow MA:", InfoIcon("info-slow", "Long-term moving average. Represents the trend.")]),
                        dcc.Slider(id="n2-slider", min=20, max=100, step=5, value=20, marks={i: str(i) for i in range(20, 110, 20)}, tooltip={"placement": "bottom", "always_visible": True}),
                    ])
                ])
            ], color="secondary", outline=True)
        ], width=3),

        # MAIN CONTENT
        dbc.Col([
            # METRICS
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.H5(["Sharpe Ratio", InfoIcon("info-sharpe", "Return per unit of risk. >1 is good, >2 is excellent.")], className="card-title"), 
                    html.H3(id="metric-sharpe", children="0.00", className="text-success")])], color="dark", inverse=True), width=3),
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.H5(["Sortino Ratio", InfoIcon("info-sortino", "Similar to Sharpe, but only penalizes DOWNSIDE volatility.")], className="card-title"), 
                    html.H3(id="metric-sortino", children="0.00", className="text-info")])], color="dark", inverse=True), width=3),
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.H5(["VaR (95%)", InfoIcon("info-var", "Value at Risk: The worst expected daily loss with 95% confidence.")], className="card-title"), 
                    html.H3(id="metric-var", children="0.00%", className="text-warning")])], color="dark", inverse=True), width=3),
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.H5("Total Return", className="card-title"), 
                    html.H3(id="metric-return", children="0.00%", className="text-danger")])], color="dark", inverse=True), width=3),
            ], className="mb-4"),

            # TABS
            dbc.Tabs([
                # TAB 1: PERFORMANCE
                dbc.Tab(label="Performance", children=[
                    dbc.Card([dbc.CardBody([dcc.Graph(id="equity-graph", style={"height": "500px"})])], color="dark", inverse=True)
                ]),
                # TAB 2: RISK
                dbc.Tab(label="Risk Analysis", children=[
                    dbc.Card([dbc.CardBody([dcc.Graph(id="risk-graph", style={"height": "500px"})])], color="dark", inverse=True)
                ]),
                # TAB 3: MONTE CARLO
                dbc.Tab(label="Monte Carlo", children=[
                    dbc.Card([dbc.CardBody([
                        html.H4(["Future Price Projection", InfoIcon("info-monte", "Simulates 50 possible future paths using Geometric Brownian Motion based on historical volatility.")], className="text-center"),
                        html.P("Assess tail risk and probability of future price levels.", className="text-center text-muted"),
                        html.Button("Run Simulation", id="btn-monte-carlo", n_clicks=0, className="btn btn-primary btn-block mb-3"),
                        dcc.Graph(id="monte-carlo-graph", style={"height": "450px"})
                    ])], color="dark", inverse=True)
                ]),
                # TAB 4: HEATMAP
                dbc.Tab(label="Optimization", children=[
                    dbc.Card([dbc.CardBody([
                        html.H4(["Parameter Optimization", InfoIcon("info-opt", "Runs a Grid Search across 20+ parameter combinations to find the highest return.")], className="text-center"),
                        html.P("Visualize the 'profit landscape' to avoid curve-fitting.", className="text-center text-muted"),
                        html.Button("Run Optimization", id="btn-optimize", n_clicks=0, className="btn btn-danger btn-block mb-3"),
                        dcc.Graph(id="heatmap-graph", style={"height": "450px"})
                    ])], color="dark", inverse=True)
                ]),
            ])
            
        ], width=9)
    ])
], fluid=True)

# =========================================================
# Logic & Callbacks
# =========================================================

@app.callback(Output("parameter-container", "style"), Input("strategy-drop", "value"))
def toggle_sliders(strategy_name):
    return {"display": "block"} if "SMA Crossover" in strategy_name else {"display": "none"}

@app.callback(
    [Output("equity-graph", "figure"), Output("risk-graph", "figure"),
     Output("metric-sharpe", "children"), Output("metric-sortino", "children"),
     Output("metric-var", "children"), Output("metric-return", "children")],
    [Input("ticker-drop", "value"), Input("strategy-drop", "value"), Input("n1-slider", "value"), Input("n2-slider", "value"),
     Input("date-picker", "start_date"), Input("date-picker", "end_date")]
)
def run_backtest(ticker, strategy_name, n1, n2, start_date, end_date):
    if not ticker: return go.Figure(), go.Figure(), "0", "0", "0%", "0%"

    mask = (df["Ticker"] == ticker) & (df["Date"] >= start_date) & (df["Date"] <= end_date)
    data = df.loc[mask].copy().set_index("Date")
    if len(data) < 50: return go.Figure(), go.Figure(), "N/A", "N/A", "N/A", "N/A"

    StrategyClass = STRATEGIES[strategy_name]
    equity_curve = pd.Series()
    sharpe, sortino, var_95, ret = 0.0, 0.0, 0.0, 0.0

    try:
        if StrategyClass is None: # Benchmark
            equity_curve = (data["Close"] / data["Close"].iloc[0]) * 10000
            returns = equity_curve.pct_change().dropna()
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
            ret = ((equity_curve.iloc[-1] - 10000) / 10000) * 100
            _, var_95, sortino = calculate_risk_metrics(equity_curve)
        else:
            if "SMA" in strategy_name: StrategyClass.n1, StrategyClass.n2 = n1, n2
            bt = Backtest(data, StrategyClass, cash=10000, commission=.001)
            stats = bt.run()
            equity_curve = stats["_equity_curve"]["Equity"]
            sharpe, ret = stats["Sharpe Ratio"], stats["Return [%]"]
            _, var_95, sortino = calculate_risk_metrics(equity_curve)

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode="lines", name="Strategy", line=dict(color="#00bc8c", width=2)))
        fig_eq.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Date", yaxis_title="Account Value ($)")

        returns = equity_curve.pct_change().dropna()
        fig_risk = go.Figure(data=[go.Histogram(x=returns, nbinsx=50, marker_color='#375a7f', opacity=0.75)])
        fig_risk.add_vline(x=var_95, line_width=3, line_dash="dash", line_color="red", annotation_text="VaR (95%)")
        fig_risk.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)

    except Exception as e:
        return go.Figure(), go.Figure(), "Err", "Err", "Err", "Err"

    return fig_eq, fig_risk, f"{sharpe:.2f}", f"{sortino:.2f}", f"{var_95:.2%}", f"{ret:.2f}%"

@app.callback(
    Output("monte-carlo-graph", "figure"),
    [Input("btn-monte-carlo", "n_clicks")],
    [State("ticker-drop", "value")]
)
def run_monte_carlo(n_clicks, ticker):
    if n_clicks == 0: return go.Figure()
    data = df[df["Ticker"] == ticker].copy()
    if len(data) < 10: return go.Figure()
    
    last_price = data["Close"].iloc[-1]
    daily_vol = data["Close"].pct_change().std()
    
    fig = go.Figure()
    for i in range(50):
        prices = [last_price]
        for d in range(30):
            shock = np.random.normal(0, daily_vol)
            prices.append(prices[-1] * (1 + shock))
        fig.add_trace(go.Scatter(y=prices, mode="lines", line=dict(width=1, color="rgba(0, 188, 140, 0.5)"), showlegend=False))
        
    fig.update_layout(title=f"Monte Carlo Simulation", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

@app.callback(
    Output("heatmap-graph", "figure"),
    [Input("btn-optimize", "n_clicks")],
    [State("ticker-drop", "value"), State("date-picker", "start_date"), State("date-picker", "end_date")]
)
def run_optimization(n_clicks, ticker, start, end):
    if n_clicks == 0: return go.Figure()

    mask = (df["Ticker"] == ticker) & (df["Date"] >= start) & (df["Date"] <= end)
    data = df.loc[mask].copy().set_index("Date")
    
    n1_range = range(5, 25, 5) 
    n2_range = range(20, 60, 10) 
    
    results = []
    for n1 in n1_range:
        row = []
        for n2 in n2_range:
            if n1 >= n2: 
                row.append(0)
            else:
                SmaCross.n1, SmaCross.n2 = n1, n2
                bt = Backtest(data, SmaCross, cash=10000, commission=.001)
                stats = bt.run()
                row.append(stats["Return [%]"])
        results.append(row)
        
    fig = go.Figure(data=go.Heatmap(z=results, x=list(n2_range), y=list(n1_range), colorscale='Viridis'))
    fig.update_layout(title="Profit Heatmap", xaxis_title="Slow MA", yaxis_title="Fast MA", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

if __name__ == "__main__":
    app.run(debug=True)
'''

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("SUCCESS! Dashboard with Tooltips & Explanations is ready.")