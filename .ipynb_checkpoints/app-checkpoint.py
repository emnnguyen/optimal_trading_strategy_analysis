code = '''
import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.figure_factory as ff
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np

# =========================================================
# Helper Functions
# =========================================================
def SMA(values, n):
    return pd.Series(values).rolling(n).mean()

def RSI(values, n=14):
    delta = pd.Series(values).diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    rs = gain.rolling(n).mean() / loss.rolling(n).mean()
    return 100 - (100 / (1 + rs))

# Risk Metrics Calculator
def calculate_risk_metrics(equity_curve):
    returns = equity_curve.pct_change().dropna()
    var_95 = returns.quantile(0.05) 
    downside_returns = returns[returns < 0]
    expected_return = returns.mean()
    downside_dev = downside_returns.std()
    
    if downside_dev == 0:
        sortino = 0
    else:
        sortino = (expected_return * 252) / (downside_dev * np.sqrt(252))
        
    return returns, var_95, sortino

# =========================================================
# Strategy Definitions
# =========================================================
class SmaCross(Strategy):
    n1 = 10 
    n2 = 20
    def init(self):
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
    def next(self):
        if crossover(self.sma1, self.sma2): 
            # Buy 95% of equity to save room for commission fees
            self.buy(size=0.95)
        elif crossover(self.sma2, self.sma1): 
            self.position.close()

class RsiOscillator(Strategy):
    upper = 70
    lower = 30
    def init(self):
        self.rsi = self.I(RSI, self.data.Close, 14)
    def next(self):
        if self.rsi[-1] < self.lower: 
            self.buy(size=0.95)
        elif self.rsi[-1] > self.upper: 
            self.position.close()

class RnnStrategy(Strategy):
    def init(self): pass
    def next(self): pass

class XgbStrategy(Strategy):
    def init(self): pass
    def next(self): pass

# =========================================================
# Data Loading
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
# App Layout
# =========================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("Quant Trading Dashboard", className="text-center text-primary mb-4 mt-4"), width=12)]),

    dbc.Row([
        # Sidebar Controls
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Settings", className="text-info"),
                dbc.CardBody([
                    html.Label("Select Asset:"),
                    dcc.Dropdown(id="ticker-drop", options=all_tickers, value=all_tickers[0] if all_tickers else None, clearable=False, style={"color": "black"}),
                    html.Br(),
                    
                    html.Label("Select Time Frame:"),
                    dcc.DatePickerRange(
                        id='date-picker',
                        min_date_allowed=df['Date'].min(),
                        max_date_allowed=df['Date'].max(),
                        start_date='2020-01-01', 
                        end_date=df['Date'].max(),
                        style={"width": "100%", "color": "black"}
                    ),
                    html.Br(), html.Br(),

                    html.Label("Select Strategy:"),
                    dcc.Dropdown(id="strategy-drop", options=[{"label": k, "value": k} for k in STRATEGIES.keys()], value="2. Traditional: SMA Crossover", clearable=False, style={"color": "black"}),
                    html.Br(),
                    
                    html.Div(id="parameter-container", children=[
                        html.Label("Fast MA:", className="text-warning"),
                        dcc.Slider(id="n1-slider", min=5, max=50, step=1, value=10, marks={i: str(i) for i in range(5, 55, 10)}, tooltip={"placement": "bottom", "always_visible": True}),
                        html.Label("Slow MA:", className="text-warning"),
                        dcc.Slider(id="n2-slider", min=20, max=100, step=5, value=20, marks={i: str(i) for i in range(20, 110, 20)}, tooltip={"placement": "bottom", "always_visible": True}),
                    ])
                ])
            ], color="secondary", outline=True)
        ], width=3),

        # Results Area
        dbc.Col([
            # Scorecards
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody([html.H4("Sharpe Ratio", className="card-title"), html.H2(id="metric-sharpe", children="0.00", className="text-success")])], color="dark", inverse=True), width=3),
                dbc.Col(dbc.Card([dbc.CardBody([html.H4("Sortino Ratio", className="card-title"), html.H2(id="metric-sortino", children="0.00", className="text-info")])], color="dark", inverse=True), width=3),
                dbc.Col(dbc.Card([dbc.CardBody([html.H4("Value at Risk (95%)", className="card-title"), html.H2(id="metric-var", children="0.00%", className="text-warning")])], color="dark", inverse=True), width=3),
                dbc.Col(dbc.Card([dbc.CardBody([html.H4("Total Return", className="card-title"), html.H2(id="metric-return", children="0.00%", className="text-danger")])], color="dark", inverse=True), width=3),
            ], className="mb-4"),

            # Charts Tabs
            dbc.Tabs([
                dbc.Tab(label="Equity Curve", children=[
                    dbc.Card([dbc.CardBody([dcc.Graph(id="equity-graph", style={"height": "500px"})])], color="dark", inverse=True)
                ]),
                dbc.Tab(label="Risk Distribution (VaR)", children=[
                    dbc.Card([dbc.CardBody([dcc.Graph(id="risk-graph", style={"height": "500px"})])], color="dark", inverse=True)
                ]),
            ])
            
        ], width=9)
    ])
], fluid=True)

# =========================================================
# Logic and Callbacks
# =========================================================
@app.callback(Output("parameter-container", "style"), Input("strategy-drop", "value"))
def toggle_sliders(strategy_name):
    return {"display": "block"} if "SMA Crossover" in strategy_name else {"display": "none"}

@app.callback(
    [Output("equity-graph", "figure"), 
     Output("risk-graph", "figure"),
     Output("metric-sharpe", "children"), 
     Output("metric-sortino", "children"),
     Output("metric-var", "children"),
     Output("metric-return", "children")],
    [Input("ticker-drop", "value"), Input("strategy-drop", "value"), Input("n1-slider", "value"), Input("n2-slider", "value"),
     Input("date-picker", "start_date"), Input("date-picker", "end_date")]
)
def run_backtest(ticker, strategy_name, n1, n2, start_date, end_date):
    if not ticker: return go.Figure(), go.Figure(), "0.00", "0.00", "0.00%", "0.00%"

    mask = (df["Ticker"] == ticker) & (df["Date"] >= start_date) & (df["Date"] <= end_date)
    data = df.loc[mask].copy().set_index("Date")
    
    if len(data) < 50: return go.Figure(), go.Figure(), "N/A", "N/A", "N/A", "N/A"

    StrategyClass = STRATEGIES[strategy_name]
    equity_curve = pd.Series()
    sharpe, sortino, var_95, ret = 0.0, 0.0, 0.0, 0.0

    try:
        if StrategyClass is None: # Benchmark
            start_price = data["Close"].iloc[0]
            equity_curve = (data["Close"] / start_price) * 10000
            returns = equity_curve.pct_change().dropna()
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
            ret = ((equity_curve.iloc[-1] - 10000) / 10000) * 100
            _, var_val, sort_val = calculate_risk_metrics(equity_curve)
            var_95 = var_val
            sortino = sort_val
            
        else: # Active Strategy
            if "SMA" in strategy_name:
                StrategyClass.n1, StrategyClass.n2 = n1, n2
            
            bt = Backtest(data, StrategyClass, cash=10000, commission=.001)
            stats = bt.run()
            equity_curve = stats["_equity_curve"]["Equity"]
            
            sharpe = stats["Sharpe Ratio"]
            ret = stats["Return [%]"]
            _, var_val, sort_val = calculate_risk_metrics(equity_curve)
            var_95 = var_val
            sortino = sort_val

        # Plot Equity Curve
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode="lines", name="Strategy", line=dict(color="#00bc8c", width=2)))
        fig_eq.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Date", yaxis_title="Account Value ($)")

        # Plot Risk Distribution
        returns = equity_curve.pct_change().dropna()
        fig_risk = go.Figure(data=[go.Histogram(x=returns, nbinsx=50, marker_color='#375a7f', opacity=0.75)])
        fig_risk.add_vline(x=var_95, line_width=3, line_dash="dash", line_color="red", annotation_text="VaR (95%)", annotation_position="top left")
        fig_risk.update_layout(title="Daily Return Distribution (Fat Tails Check)", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Daily Return (%)", yaxis_title="Frequency", showlegend=False)

    except Exception as e:
        # Return empty figures on failure
        return go.Figure(), go.Figure(), "Err", "Err", "Err", "Err"

    return fig_eq, fig_risk, f"{sharpe:.2f}", f"{sortino:.2f}", f"{var_95:.2%}", f"{ret:.2f}%"

if __name__ == "__main__":
    app.run(debug=True)
'''

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("SUCCESS! 'app.py' updated with clean formatting.")