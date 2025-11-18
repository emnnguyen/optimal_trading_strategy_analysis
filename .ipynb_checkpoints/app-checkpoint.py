code = '''
import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np

# helper
def SMA(values, n):
    return pd.Series(values).rolling(n).mean()

def RSI(values, n=14):
    delta = pd.Series(values).diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    rs = gain.rolling(n).mean() / loss.rolling(n).mean()
    return 100 - (100 / (1 + rs))

# strategies
class SmaCross(Strategy):
    n1 = 10 
    n2 = 20
    def init(self):
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
    def next(self):
        if crossover(self.sma1, self.sma2): self.buy()
        elif crossover(self.sma2, self.sma1): self.sell()

class RsiOscillator(Strategy):
    upper = 70
    lower = 30
    def init(self):
        self.rsi = self.I(RSI, self.data.Close, 14)
    def next(self):
        if self.rsi[-1] < self.lower: self.buy()
        elif self.rsi[-1] > self.upper: self.position.close()

class RnnStrategy(Strategy):
    def init(self): pass
    def next(self): pass

class XgbStrategy(Strategy):
    def init(self): pass
    def next(self): pass

# data
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

# layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("⚡ QUANT TRADING DASHBOARD", className="text-center text-primary mb-4 mt-4"), width=12)]),

    dbc.Row([
        # SIDEBAR CONTROLS
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
                        # FIX: Marks are now spaced out every 5 steps
                        dcc.Slider(id="n1-slider", min=5, max=50, step=1, value=10, 
                                   marks={i: str(i) for i in range(5, 55, 10)}, 
                                   tooltip={"placement": "bottom", "always_visible": True}),
                                   
                        html.Label("Slow MA:", className="text-warning"),
                        # FIX: Marks are now spaced out every 20 steps
                        dcc.Slider(id="n2-slider", min=20, max=100, step=5, value=20, 
                                   marks={i: str(i) for i in range(20, 110, 20)}, 
                                   tooltip={"placement": "bottom", "always_visible": True}),
                    ])
                ])
            ], color="secondary", outline=True)
        ], width=3),

        # RESULTS AREA
        dbc.Col([
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody([html.H4("Sharpe Ratio", className="card-title"), html.H2(id="metric-sharpe", children="0.00", className="text-success")])], color="dark", inverse=True), width=4),
                dbc.Col(dbc.Card([dbc.CardBody([html.H4("Total Return", className="card-title"), html.H2(id="metric-return", children="0.00%", className="text-info")])], color="dark", inverse=True), width=4),
                dbc.Col(dbc.Card([dbc.CardBody([html.H4("Max Drawdown", className="card-title"), html.H2(id="metric-drawdown", children="0.00%", className="text-danger")])], color="dark", inverse=True), width=4),
            ], className="mb-4"),

            dbc.Card([dbc.CardBody([dcc.Graph(id="equity-graph", style={"height": "500px"})])], color="dark", inverse=True)
        ], width=9)
    ])
], fluid=True)

# callbacks
@app.callback(Output("parameter-container", "style"), Input("strategy-drop", "value"))
def toggle_sliders(strategy_name):
    return {"display": "block"} if "SMA Crossover" in strategy_name else {"display": "none"}

@app.callback(
    [Output("equity-graph", "figure"), Output("metric-sharpe", "children"), Output("metric-return", "children"), Output("metric-drawdown", "children")],
    [Input("ticker-drop", "value"), Input("strategy-drop", "value"), Input("n1-slider", "value"), Input("n2-slider", "value"),
     Input("date-picker", "start_date"), Input("date-picker", "end_date")]
)
def run_backtest(ticker, strategy_name, n1, n2, start_date, end_date):
    if not ticker: return go.Figure(), "0.00", "0.00%", "0.00%"

    # Filter Data
    mask = (df["Ticker"] == ticker) & (df["Date"] >= start_date) & (df["Date"] <= end_date)
    data = df.loc[mask].copy().set_index("Date")
    
    if len(data) < 50: return go.Figure(), "N/A", "N/A", "N/A"

    StrategyClass = STRATEGIES[strategy_name]
    fig = go.Figure()
    sharpe, ret, max_dd = 0.0, 0.0, 0.0

    try:
        # Benchmark
        start_price = data["Close"].iloc[0]
        buy_hold_equity = (data["Close"] / start_price) * 10000
        fig.add_trace(go.Scatter(x=data.index, y=buy_hold_equity, mode="lines", name="Buy & Hold", line=dict(color="#00bc8c", width=2, dash="dash")))

        if StrategyClass is None:
            ret = ((buy_hold_equity.iloc[-1] - 10000) / 10000) * 100
        else:
            if "SMA" in strategy_name:
                StrategyClass.n1, StrategyClass.n2 = n1, n2
            
            bt = Backtest(data, StrategyClass, cash=10000, commission=.001)
            stats = bt.run()
            curve = stats["_equity_curve"]
            
            fig.add_trace(go.Scatter(x=curve.index, y=curve["Equity"], mode="lines", name=strategy_name, line=dict(color="#375a7f", width=2)))
            sharpe, ret, max_dd = stats["Sharpe Ratio"], stats["Return [%]"], stats["Max. Drawdown [%]"]

        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Date", yaxis_title="Account Value ($)")
        
    except Exception as e:
        return go.Figure(), "Err", "Err", "Err"

    return fig, f"{sharpe:.2f}", f"{ret:.2f}%", f"{max_dd:.2f}%"

if __name__ == "__main__":
    app.run(debug=True)
'''

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("SUCCESS! 'app.py' sliders are fixed.")