# forex_ml_bot.py

import MetaTrader5 as mt5
import pandas as pd
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import time
import os

# ==============================
# CONFIGURAÇÕES GERAIS
# ==============================
SYMBOLS = ['EURUSD', 'USDCAD', 'GBPUSD', 'USDJPY',"USDCHF","USDCNH","AUDUSD","NVDA"]
TIMEFRAME_ENTRY = mt5.TIMEFRAME_M30
TIMEFRAME_TRAILING = mt5.TIMEFRAME_M5
LOT = 0.1
FEATURES = ['ema8', 'ema21', 'rsi', 'atr', 'bullish_candle', 'bearish_candle', 'engulfing', 'pinbar']

# ==============================
# INICIALIZAÇÃO DO MT5
# ==============================
def initialize_mt5():
    if not mt5.initialize():
        raise RuntimeError("Erro ao inicializar MT5")
    print("MT5 iniciado com sucesso")

# ==============================
# PREPARAÇÃO DE DADOS
# ==============================
def detect_candle_patterns(df):
    df['bullish_candle'] = (df['close'] > df['open']).astype(int)
    df['bearish_candle'] = (df['close'] < df['open']).astype(int)

    df['engulfing'] = (
        ((df['bullish_candle'].shift(1).fillna(0).astype(bool)) & (df['bearish_candle'].astype(bool))) |
        ((df['bearish_candle'].shift(1).fillna(0).astype(bool)) & (df['bullish_candle'].astype(bool)))
    ).astype(int)

    df['pinbar'] = (
        (abs(df['open'] - df['close']) < (df['high'] - df['low']) * 0.3) &
        ((df['high'] - df[['open', 'close']].max(axis=1)) >
         (df[['open', 'close']].min(axis=1) - df['low']) * 2)
    ).astype(int)

    return df

def prepare_data(symbol, timeframe, bars=3000, future_candles=5):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['ema8'] = EMAIndicator(df['close'], 8).ema_indicator()
    df['ema21'] = EMAIndicator(df['close'], 21).ema_indicator()
    df['rsi'] = RSIIndicator(df['close'], 14).rsi()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
    df = detect_candle_patterns(df)
    df['future_return'] = df['close'].shift(-future_candles) - df['close']
    df['label'] = (df['future_return'] > df['atr'] * 1.5).astype(int)
    df.dropna(inplace=True)
    return df

# ==============================
# TREINAMENTO DO MODELO
# ==============================
def train_and_save_model(df, symbol):
    model_path = f"./models/{symbol}_model.pkl"
    X = df[FEATURES].copy()
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"[{symbol}] Acurácia do modelo: {acc:.2f}")
    joblib.dump(model, model_path)
    return model

# ==============================
# VERIFICA EXISTÊNCIA DE POSIÇÃO GLOBAL
# ==============================
def has_any_open_position():
    positions = mt5.positions_get()
    return bool(positions)

# ==============================
# ENVIO DE ORDENS E TRAILING
# ==============================
def send_order(symbol, signal, atr):
    if has_any_open_position():
        print(f"[{symbol}] Já existe uma ordem ativa em outro par. Ignorado.")
        return

    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if signal == 'buy' else tick.bid
    sl_pips = atr * 2
    tp_pips = atr * 2.5
    sl = price - sl_pips if signal == 'buy' else price + sl_pips
    tp = price + tp_pips if signal == 'buy' else price - tp_pips
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": LOT,
        "type": mt5.ORDER_TYPE_BUY if signal == 'buy' else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 42,
        "comment": "ML ENTRY",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    print(f"[{symbol}] Ordem enviada: {result}")

def update_trailing_stop(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return

    df = prepare_data(symbol, TIMEFRAME_TRAILING, 50)
    atr = df.iloc[-1]['atr']
    tick = mt5.symbol_info_tick(symbol)

    for position in positions:
        if position.type == mt5.ORDER_TYPE_BUY:
            new_sl = tick.bid - atr
            if new_sl > position.sl:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": position.ticket,
                    "sl": new_sl,
                    "tp": position.tp,
                    "symbol": symbol,
                }
                result = mt5.order_send(request)
                print(f"[{symbol}] Trailing SL atualizado: {result}")
        elif position.type == mt5.ORDER_TYPE_SELL:
            new_sl = tick.ask + atr
            if new_sl < position.sl:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": position.ticket,
                    "sl": new_sl,
                    "tp": position.tp,
                    "symbol": symbol,
                }
                result = mt5.order_send(request)
                print(f"[{symbol}] Trailing SL atualizado: {result}")

# ==============================
# EXECUÇÃO DO BOT
# ==============================
def run_bot():
    for symbol in SYMBOLS:
        try:
            model_path = f"./models/{symbol}_model.pkl"
            if not os.path.exists(model_path):
                print(f"[{symbol}] Modelo não encontrado. Ignorado.")
                continue

            model = joblib.load(model_path)
            df = prepare_data(symbol, TIMEFRAME_ENTRY, 300)
            latest = df.iloc[-1:]
            X_latest = latest[FEATURES].copy()

            # Validação de integridade dos dados
            missing_features = [f for f in FEATURES if f not in X_latest.columns]
            if missing_features:
                print(f"[{symbol}] Faltando colunas nos dados mais recentes: {missing_features}")
                continue

            if X_latest.isnull().any().any():
                print(f"[{symbol}] Dados mais recentes contêm NaN. Ignorado.")
                continue

            atr = latest['atr'].values[0]
            prob = model.predict_proba(X_latest)[0][1]
            print(f"[{symbol}] Probabilidade: {prob:.2f}")

            #if prob > 0.8:
            if prob >= 0.6:
                if latest['bullish_candle'].values[0]:
                    send_order(symbol, 'buy', atr)
                elif latest['bearish_candle'].values[0]:
                    send_order(symbol, 'sell', atr)

            update_trailing_stop(symbol)

        except Exception as e:
            print(f"Erro com {symbol}: {e}")

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    initialize_mt5()
    for symbol in SYMBOLS:
        df = prepare_data(symbol, TIMEFRAME_ENTRY, 3000)
        train_and_save_model(df, symbol)
    while True:
        run_bot()
        
        time.sleep(TIMEFRAME_ENTRY * 60)  # Executa a cada 1 minuto
