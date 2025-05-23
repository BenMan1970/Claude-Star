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
st.markdown("*Filtrage automatique 5-6 étoiles avec validation tendance D1/H4*")

# Liste réduite des paires principales
FOREX_PAIRS = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 
    'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X', 'EURJPY=X', 
    'GBPJPY=X', 'EURGBP=X', 'GC=F'  # XAU/USD
]

@st.cache_data(ttl=300)  # Cache 5 minutes
def get_data(symbol, period='2d', interval='1h'):
    """Récupère les données avec cache"""
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
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

def check_trend_alignment(symbol):
    """
    Vérifie si D1 et H4 tendent dans le même sens
    Retourne: (True/False, direction_D1, direction_H4, force_trend)
    """
    try:
        # Données D1 (5 derniers jours)
        data_d1 = get_data(symbol, period='5d', interval='1d')
        # Données H4 (5 derniers jours en H4)
        data_h4 = get_data(symbol, period='5d', interval='4h')
        
        if data_d1 is None or data_h4 is None or len(data_d1) < 3 or len(data_h4) < 6:
            return False, "INCONNU", "INCONNU", 0
        
        # Analyse D1 - EMA 10 vs EMA 20
        close_d1 = data_d1['Close']
        ema10_d1 = close_d1.ewm(span=10).mean()
        ema20_d1 = close_d1.ewm(span=20).mean()
        
        # Force de la tendance D1 (distance entre EMAs en %)
        trend_strength_d1 = abs((ema10_d1.iloc[-1] - ema20_d1.iloc[-1]) / ema20_d1.iloc[-1] * 100)
        direction_d1 = "HAUSSIER" if ema10_d1.iloc[-1] > ema20_d1.iloc[-1] else "BAISSIER"
        
        # Analyse H4 - EMA 10 vs EMA 20
        close_h4 = data_h4['Close']
        ema10_h4 = close_h4.ewm(span=10).mean()
        ema20_h4 = close_h4.ewm(span=20).mean()
        
        # Force de la tendance H4 (distance entre EMAs en %)
        trend_strength_h4 = abs((ema10_h4.iloc[-1] - ema20_h4.iloc[-1]) / ema20_h4.iloc[-1] * 100)
        direction_h4 = "HAUSSIER" if ema10_h4.iloc[-1] > ema20_h4.iloc[-1] else "BAISSIER"
        
        # Vérification de l'alignement
        is_aligned = direction_d1 == direction_h4
        
        # Force combinée (moyenne des deux timeframes)
        combined_strength = (trend_strength_d1 + trend_strength_h4) / 2
        
        return is_aligned, direction_d1, direction_h4, combined_strength
        
    except Exception as e:
        return False, "ERREUR", "ERREUR", 0

def calculate_signals(data):
    """Calcule tous les signaux de confluence sur H1"""
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

def get_trend_emoji(strength):
    """Emoji selon la force de tendance"""
    if strength >= 0.5:
        return "🚀"  # Tendance très forte
    elif strength >= 0.3:
        return "📈"  # Tendance forte
    elif strength >= 0.1:
        return "↗️"   # Tendance modérée
    else:
        return "➡️"   # Tendance faible

# Interface principale
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("⚙️ Paramètres")
    min_stars = st.selectbox("Étoiles minimum", [4, 5, 6], index=1)
    
    st.markdown("---")
    st.markdown("**🎯 Filtre Tendance Multi-TF**")
    st.markdown("✅ D1 et H4 alignés requis")
    
    scan_button = st.button("🔍 Scanner", type="primary", use_container_width=True)

with col2:
    if scan_button:
        results = []
        filtered_out = []
        progress_bar = st.progress(0)
        status = st.empty()
        
        for i, symbol in enumerate(FOREX_PAIRS):
            progress_bar.progress((i + 1) / len(FOREX_PAIRS))
            status.text(f"📊 Analyse multi-TF: {symbol.replace('=X', '').replace('=F', ' (Gold)')}")
            
            # 1. Vérification alignement D1/H4
            is_aligned, dir_d1, dir_h4, trend_strength = check_trend_alignment(symbol)
            
            pair_name = symbol.replace('=X', '').replace('=F', ' (Gold)')
            
            if not is_aligned:
                filtered_out.append({
                    'Paire': pair_name,
                    'Raison': f"D1: {dir_d1} ≠ H4: {dir_h4}",
                    'Force': f"{trend_strength:.2f}%"
                })
                continue
            
            # 2. Si aligné, analyse des confluences H1
            data_h1 = get_data(symbol, period='2d', interval='1h')
            if data_h1 is not None:
                signals = calculate_signals(data_h1)
                if signals and signals['confluence'] >= min_stars:
                    # Vérification cohérence H1 avec D1/H4
                    if signals['direction'] == dir_d1:
                        results.append({
                            'Paire': pair_name,
                            'Direction': signals['direction'],
                            'Étoiles': get_stars(signals['confluence']),
                            'Confluence': signals['confluence'],
                            'Tendance': get_trend_emoji(trend_strength),
                            'Force_TF': f"{trend_strength:.2f}%",
                            'D1': "📈" if dir_d1 == "HAUSSIER" else "📉",
                            'H4': "📈" if dir_h4 == "HAUSSIER" else "📉",
                            'RSI': f"{signals['rsi']:.0f}",
                            'ADX': f"{signals['adx']:.0f}",
                            'HMA': signals['signals']['HMA'],
                            'RSI_S': signals['signals']['RSI'],
                            'EMA': signals['signals']['EMA'],
                            'HA': signals['signals']['HA'],
                            'Ichi': signals['signals']['Ichimoku'],
                            'ADX_S': signals['signals']['ADX']
                        })
            
            time.sleep(0.3)  # Pause plus longue car plus de requêtes
        
        progress_bar.empty()
        status.empty()
        
        # Affichage des résultats
        if results:
            st.success(f"🎯 {len(results)} paire(s) validée(s) multi-timeframe!")
            df = pd.DataFrame(results).sort_values(['Force_TF', 'Confluence'], ascending=[False, False])
            
            # Tableau principal
            st.dataframe(
                df[['Paire', 'Direction', 'Étoiles', 'Tendance', 'Force_TF', 'D1', 'H4', 'RSI', 'ADX', 'HMA', 'RSI_S', 'EMA', 'HA', 'Ichi', 'ADX_S']],
                use_container_width=True,
                hide_index=True
            )
            
            # Top signaux
            st.subheader("🚨 Signaux Premium (Multi-TF validés)")
            for _, row in df.head(3).iterrows():
                color = "🟢" if row['Direction'] == "HAUSSIER" else "🔴"
                st.write(f"{color} **{row['Paire']}** {row['Tendance']} - {row['Étoiles']} - Force: {row['Force_TF']} (RSI: {row['RSI']}, ADX: {row['ADX']})")
        else:
            st.warning(f"❌ Aucune paire avec {min_stars}+ étoiles ET alignement D1/H4.")
        
        # Paires filtrées (optionnel)
        if filtered_out and st.checkbox("📋 Voir paires filtrées (tendances non alignées)"):
            st.subheader("⚠️ Paires filtrées")
            df_filtered = pd.DataFrame(filtered_out)
            st.dataframe(df_filtered, use_container_width=True, hide_index=True)
            st.caption(f"🚫 {len(filtered_out)} paires écartées pour défaut d'alignement")

# Légende enrichie
with st.expander("ℹ️ Légende Multi-Timeframe"):
    st.markdown("""
    **🎯 Processus de filtrage:**
    1. **Étape 1:** Vérification alignement D1 ↔ H4 (EMA 10 vs 20)
    2. **Étape 2:** Si aligné → Analyse confluences H1
    3. **Étape 3:** Validation cohérence H1 avec tendance supérieure
    
    **📊 Symboles tendance:**
    - 🚀 Très forte (>0.5%) | 📈 Forte (>0.3%) | ↗️ Modérée (>0.1%) | ➡️ Faible
    
    **⭐ Signaux H1:** ▲ = Haussier, ▼ = Baissier, ✔ = Momentum, ✖ = Pas momentum
    
    **🔍 Indicateurs:**
    - **D1/H4:** Tendance principale (EMA 10/20)
    - **H1:** HMA(20), RSI(10), EMA(10/20), Heiken Ashi, Ichimoku, ADX(14)
    """)

# Note professionnelle
st.info("💡 **Scanner Multi-Timeframe** - Seules les paires avec alignement D1/H4 passent l'analyse de confluence")

# Auto-refresh avec délai adapté
if st.checkbox("🔄 Auto-refresh (60s)", help="Délai plus long car analyse multi-timeframe"):
    time.sleep(60)
    st.rerun()
   
