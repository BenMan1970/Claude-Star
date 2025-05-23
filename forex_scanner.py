import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Configuration de la page
st.set_page_config(
    page_title="Scanner Confluence Forex",
    page_icon="⭐",
    layout="wide"
)

st.title("🔍 Scanner Confluence Forex Premium")
st.markdown("*Filtrage automatique 5-6 étoiles*")

# Liste réduite des paires principales
FOREX_PAIRS = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 
    'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X', 'EURJPY=X', 
    'GBPJPY=X', 'EURGBP=X', 'GC=F'  # XAU/USD
]

def get_data(symbol, period='2d'):
    """Récupère les données avec cache"""
    try:
        data = yf.download(symbol, period=period, interval='1h', progress=False)
        if data.empty:
            return None
        return data.dropna()
    except:
        return None

def wma(data, length):
    """Weighted Moving Average simple"""
    weights = np.arange(1, length + 1)
    return data.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def hma(data, length):
    """Hull Moving Average simplifié"""
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    
    wma1 = wma(data, half_length)
    wma2 = wma(data, length)
    diff = 2 * wma1 - wma2
    return wma(diff, sqrt_length)

def rsi(data, length):
    """RSI simplifié"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def atr(high, low, close, length=14):
    """Average True Range"""
    h_l = high - low
    h_c = np.abs(high - close.shift())
    l_c = np.abs(low - close.shift())
    tr = np.maximum(h_l, np.maximum(h_c, l_c))
    return tr.rolling(length).mean()

def adx_simple(high, low, close, length=14):
    """ADX simplifié sans TA-Lib"""
    # Plus/Minus Directional Movement
    up_move = high.diff()
    down_move = -low.diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=high.index).rolling(length).mean()
    minus_dm = pd.Series(minus_dm, index=high.index).rolling(length).mean()
    
    atr_val = atr(high, low, close, length)
    
    plus_di = 100 * (plus_dm / atr_val)
    minus_di = 100 * (minus_dm / atr_val)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(length).mean()
    
    return adx.fillna(0)

def calculate_signals(data):
    """Calcule tous les signaux de confluence"""
    if data is None or len(data) < 50:
        return None
    
    try:
        close = data['Close']
        high = data['High']
        low = data['Low']
        open_price = data['Open']
        
        # 1. HMA Signal (longueur 20)
        hma_val = hma(close, 20)
        hma_signal = 1 if hma_val.iloc[-1] > hma_val.iloc[-2] else -1
        
        # 2. RSI Signal (longueur 10 sur OHLC/4)
        ohlc4 = (open_price + high + low + close) / 4
        rsi_val = rsi(ohlc4, 10)
        rsi_signal = 1 if rsi_val.iloc[-1] > 50 else -1
        
        # 3. ADX Signal
        adx_val = adx_simple(high, low, close, 14)
        adx_momentum = adx_val.iloc[-1] > 20
        
        # 4. Heiken Ashi simple
        ha_close = ohlc4
        ha_open = (open_price.shift(1) + close.shift(1)) / 2
        ha_signal = 1 if ha_close.iloc[-1] > ha_open.iloc[-1] else -1
        
        # 5. EMA Signal (remplace HA lissé)
        ema_fast = close.ewm(span=10).mean()
        ema_slow = close.ewm(span=20).mean()
        ema_signal = 1 if ema_fast.iloc[-1] > ema_slow.iloc[-1] else -1
        
        # 6. Ichimoku simple (Tenkan vs Kijun)
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        ichimoku_signal = 1 if tenkan.iloc[-1] > kijun.iloc[-1] else -1
        
        # Comptage des confluences
        signals = [hma_signal, rsi_signal, ema_signal, ha_signal, ichimoku_signal]
        if adx_momentum:
            signals.append(1 if sum(s for s in signals if s == 1) > sum(s for s in signals if s == -1) else -1)
        
        bull_count = sum(1 for s in signals if s == 1)
        bear_count = sum(1 for s in signals if s == -1)
        confluence = max(bull_count, bear_count)
        direction = "HAUSSIER" if bull_count > bear_count else "BAISSIER"
        
        return {
            'confluence': confluence,
            'direction': direction,
            'rsi': rsi_val.iloc[-1],
            'adx': adx_val.iloc[-1],
            'adx_momentum': adx_momentum,
            'signals': {
                'HMA': "▲" if hma_signal == 1 else "▼",
                'RSI': "▲" if rsi_signal == 1 else "▼",
                'EMA': "▲" if ema_signal == 1 else "▼",
                'HA': "▲" if ha_signal == 1 else "▼",
                'Ichimoku': "▲" if ichimoku_signal == 1 else "▼",
                'ADX': "✔" if adx_momentum else "✖"
            }
        }
    except Exception as e:
        return None

def get_stars(confluence):
    """Étoiles selon confluence"""
    stars = ["WAIT", "⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐⭐"]
    return stars[min(confluence, 6)]

# Interface principale
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("⚙️ Paramètres")
    min_stars = st.selectbox("Étoiles minimum", [4, 5, 6], index=1)
    
    scan_button = st.button("🔍 Scanner", type="primary", use_container_width=True)

with col2:
    if scan_button:
        results = []
        progress_bar = st.progress(0)
        status = st.empty()
        
        for i, symbol in enumerate(FOREX_PAIRS):
            progress_bar.progress((i + 1) / len(FOREX_PAIRS))
            status.text(f"Scan: {symbol.replace('=X', '').replace('=F', ' (Gold)')}")
            
            data = get_data(symbol)
            if data is not None:
                signals = calculate_signals(data)
                if signals and signals['confluence'] >= min_stars:
                    pair_name = symbol.replace('=X', '').replace('=F', ' (Gold)')
                    results.append({
                        'Paire': pair_name,
                        'Direction': signals['direction'],
                        'Étoiles': get_stars(signals['confluence']),
                        'Confluence': signals['confluence'],
                        'RSI': f"{signals['rsi']:.0f}",
                        'ADX': f"{signals['adx']:.0f}",
                        'HMA': signals['signals']['HMA'],
                        'RSI_S': signals['signals']['RSI'],
                        'EMA': signals['signals']['EMA'],
                        'HA': signals['signals']['HA'],
                        'Ichi': signals['signals']['Ichimoku'],
                        'ADX_S': signals['signals']['ADX']
                    })
            
            time.sleep(0.2)  # Pause pour éviter les limites
        
        progress_bar.empty()
        status.empty()
        
        if results:
            st.success(f"🎯 {len(results)} paire(s) trouvée(s)!")
            df = pd.DataFrame(results).sort_values('Confluence', ascending=False)
            
            # Tableau compact
            st.dataframe(
                df[['Paire', 'Direction', 'Étoiles', 'RSI', 'ADX', 'HMA', 'RSI_S', 'EMA', 'HA', 'Ichi', 'ADX_S']],
                use_container_width=True,
                hide_index=True
            )
            
            # Alertes rapides
            st.subheader("🚨 Signaux forts")
            for _, row in df.iterrows():
                color = "🟢" if row['Direction'] == "HAUSSIER" else "🔴"
                st.write(f"{color} **{row['Paire']}** - {row['Étoiles']} - {row['Direction']} (RSI: {row['RSI']}, ADX: {row['ADX']})")
        else:
            st.warning(f"❌ Aucune paire avec {min_stars}+ étoiles trouvée.")

# Légende
with st.expander("ℹ️ Légende"):
    st.markdown("""
    **Signaux:** ▲ = Haussier, ▼ = Baissier, ✔ = Momentum, ✖ = Pas de momentum
    
    **Indicateurs:**
    - HMA: Hull Moving Average (20)
    - RSI: Relative Strength Index (10)
    - EMA: Moyennes mobiles exponentielles (10/20)
    - HA: Heiken Ashi
    - Ichi: Ichimoku (Tenkan vs Kijun)
    - ADX: Directional Movement (seuil 20)
    """)

# Note importante
st.info("💡 **Version allégée** - Calculs optimizés pour Streamlit Cloud")

# Auto-refresh optionnel
if st.checkbox("🔄 Auto-refresh (30s)"):
    time.sleep(30)
    st.rerun()