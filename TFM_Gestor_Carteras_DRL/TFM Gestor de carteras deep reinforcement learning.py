import yfinance as yf
from curl_cffi import requests
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator
from sklearn.preprocessing import StandardScaler
import gym
from gym import spaces
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

HIST_GLOBAL = []


session = requests.Session(impersonate="chrome")
# --------------------
# 1. PARÁMETROS INICIALES
# --------------------
tickers = [
            'APD', 'UNP', 'AFL', 'CSCO', 'FAST', 'AMGN', 'PLD', 'PCAR', 'RTX', 'BK',  
            'FTNT', 'NXPI', 'MPC', 'AMAT', 'KLAC', 'LRCX', 'VLO', 'URI', 'AVGO',
            'MRK', 'T', 'PSA', 'BDX', 'SO', 'PEG', 'XEL', 'PM', 'MDLZ', 'MO', 
            'RCL', 'AXON', 'MU', 'NFLX', 'AMD', 'NVDA', 'FCX', 'TSLA',
            'AAPL', 'MSCI', 'INTU', 'SNPS', 'BAC', 'ODFL', 'PH', 'MCO', 'BKNG', 'ADBE',
            'EOG', 'BKR', 'COP', 'BA', 'WMB', 'F', 'INTC', 'OKE', 'SPG',
            "SHY", "TLT", "IEF"
]
start_date = "2012-01-03"
end_date   = "2025-05-16"

# --------------------
# 2. FUNCIÓN PARA CONSTRUIR FEATURES
# --------------------
def build_features(df):
    df = df.copy()
    # 1) Aplanar MultiIndex si existe
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(map(str, col)).strip().lower() for col in df.columns]
    else:
        df.columns = [c.lower().strip() for c in df.columns]

    # 2) Buscar columna de precios close (prefiere adj close si existe)
    close_cols = [c for c in df.columns if "close" in c]
    if not close_cols:
        raise KeyError("No se encontró columna con 'close'")
    # si hay alguna que contenga 'adj', esa tiene prioridad
    adj  = [c for c in close_cols if "adj" in c]
    if adj:
        close_col = adj[0]
    else:
        # buscar exactamente 'close'
        exact = [c for c in close_cols if c == "close"]
        close_col = exact[0] if exact else close_cols[0]
    df["close"] = df[close_col]

    # 3) Buscar high / low
    high_cols = [c for c in df.columns if "high" in c]
    low_cols  = [c for c in df.columns if "low" in c]
    

    if not high_cols or not low_cols:
        raise KeyError("Falta columna con 'high' o 'low'")
    df["high"] = df[high_cols[0]]
    df["low"]  = df[low_cols[0]]

    # 4) Indicadores técnicos
    df["rsi"]      = RSIIndicator(df["close"], window=14).rsi()
    st = StochasticOscillator(df["high"], df["low"], df["close"], window=14)
    df["stoch_k"]  = st.stoch()
    df["stoch_d"]  = st.stoch_signal()
    macd_ind = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    adx_ind = ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"] = adx_ind.adx()
    

    # 5) Retorno con winsorización
    df["return"]     = df["close"].pct_change().clip(-1.0, 1.5)


    # 6) Limpiar NaNs
    return df.dropna()


from datetime import timedelta

def marcar_ventana_resultados(df, earnings_dates, pre=2, post=2):
    """
    Añade una columna 'earnings_window' que vale 1 en los días cercanos a resultados, 0 si no.
    - df: DataFrame con índice de fechas
    - earnings_dates: lista de fechas (datetime.date)
    - pre/post: días antes/después de la publicación que marcamos
    """
    df['earnings_window'] = 0
    fechas = set(df.index.date)
    for fecha in earnings_dates:
        for delta in range(-pre, post + 1):
            dia = fecha + timedelta(days=delta)
            if dia in fechas:
                df.loc[df.index.date == dia, 'earnings_window'] = 1
    return df

# --------------------
# 3. DESCARGA Y MERGE CON TIPOS DE INTERÉS (CON FILTRADO Y LISTADO DE DESCARTADOS)
# --------------------
datasets = {}
descartados = []  # lista para guardar los símbolos descartados

for t in tickers:
    print(f"Descargando {t}...")
    data = yf.download(t, start=start_date, end=end_date, session=session, progress=False)

    # 1) Sin datos: descartamos
    if data.empty:
        print(f"  → Sin datos para {t}, se descarta.")
        descartados.append(t)
        continue

    # 2) Fecha de primer cotización después del inicio: descartamos
    primera_fecha = data.index.min()
    if primera_fecha > pd.to_datetime(start_date):
        print(f"  → {t} empezó a cotizar el {primera_fecha.date()}, después de {start_date}. Se descarta.")
        descartados.append(t)
        continue

    # 3) Sólo construimos features de los válidos
    feats = build_features(data)
   
    try:
        ticker_obj = yf.Ticker(t)
        earnings_dates = ticker_obj.get_earnings_dates(limit=60)  # Puedes aumentar el límite si quieres
        result_dates = earnings_dates.index.date.tolist()
        feats = marcar_ventana_resultados(feats, result_dates, pre=2, post=2)
    except Exception as e:
        print(f"Error obteniendo earnings para {t}: {e}")
        feats['earnings_window'] = 0  # Si hay error, todo a cero

    # --------------------------------------------------------------

    if not feats.empty:
        datasets[t] = feats

# Al terminar, imprimimos el resumen de descartados
if descartados:
    print("\nTickers descartados por comenzar a cotizar despues de 2012:")
    print(", ".join(descartados))
else:
    print("\nNo se han descartado símbolos; todos cotizaban desde el inicio.")

# --------------------
# 4. DESCARGA Y PROCESAMIENTO DE RATES + VIX + CITI economic surprise
# --------------------
# Descargamos los ETF de tipos de interés
from pandas_datareader import data as web

rate_tickers = [
    'DGS3MO', 'DGS1', 'DGS2', 'DGS5', 'DGS7', 'DGS10',  # yields soberanos
    'DBAA', 'DAAA',           # Moody's Baa y Aaa Corporate Bond Yields
    'BAA10Y', 'AAA10Y'        # Spreads directos
]
rates = web.DataReader(rate_tickers, 'fred', start_date, end_date)
rates.columns = [
    'rate_3m', 'rate_1y', 'rate_2y', 'rate_5y', 'rate_7y', 'rate_10y',
    'moody_baa', 'moody_aaa', 'spread_baa_10y', 'spread_aaa_10y'
]
rates = rates.ffill()

rates['spread_10y_2y'] = rates['rate_10y'] - rates['rate_2y']
rates['spread_5y_3m']  = rates['rate_5y']  - rates['rate_3m']


# --- NUEVO: descarga VIX ---
vix_raw = yf.download("^VIX", start=start_date, end=end_date, session=session, progress=False)
# aplanar columnas
if isinstance(vix_raw.columns, pd.MultiIndex):
    vix_raw.columns = [" ".join(map(str,c)).lower() for c in vix_raw.columns]
else:
    vix_raw.columns = [c.lower() for c in vix_raw.columns]
# elegir adj close > close
vix_cols = [c for c in vix_raw.columns if "close" in c]
vix_adj  = [c for c in vix_cols if "adj" in c]
col_vix  = vix_adj[0] if vix_adj else vix_cols[0]

# añadimos serie VIX
rates["vix"] = vix_raw[col_vix].ffill()

# --------------------
# 4.b. AÑADIR VARIABLES MACROECONÓMICAS DIARIAS
# --------------------
# (insertar justo después de haber construido el DataFrame `rates` con rate_10y, rate_3m y vix)

# 1) Descarga y merge de Commodities: Oro y Petróleo WTI
commodities = yf.download(['GC=F', 'CL=F'], start=start_date, end=end_date, session=session, progress=False)[['Close']]
# Renombrar columnas a minúsculas
commodities.columns = ['gold', 'wti']
# Forward-fill para evitar NaNs
commodities = commodities.ffill()
# Merge con rates
rates = rates.merge(commodities, left_index=True, right_index=True)

# 3) Descarga e incorporación del CBOE Skew Index
skew_raw = yf.download('^SKEW', start=start_date, end=end_date, session=session, progress=False)[['Close']]
skew_raw.columns = ['skew']
skew_raw = skew_raw.ffill()
# Merge con rates
rates = rates.merge(skew_raw, left_index=True, right_index=True)

# 4) Descarga e incorporación de tipos de cambio: EUR/USD, CNY/USD, MXN/USD y CAD/USD
fx_raw = yf.download(
    ['EURUSD=X', 'CNYUSD=X', 'MXNUSD=X', 'CADUSD=X'],
    start=start_date,
    end=end_date,
    session=session, 
    progress=False
)[['Close']]
# Renombrar columnas a minúsculas
fx_raw.columns = ['eur_usd', 'cny_usd', 'mxn_usd', 'cad_usd']
# Forward-fill para evitar NaNs
fx_raw = fx_raw.ffill()
# Merge con rates
rates = rates.merge(fx_raw, left_index=True, right_index=True)

from statsmodels.tsa.vector_ar.vecm import VECM

vecm_df = pd.concat([
    rates[['rate_3m','rate_1y','rate_2y','rate_5y','rate_7y','rate_10y','moody_baa','moody_aaa']],
    fx_raw[['eur_usd','cny_usd','mxn_usd','cad_usd']]
], axis=1).dropna()


fecha_corte = "2022-06-30"  #(el último índice de train)
fecha_corte = pd.to_datetime(fecha_corte)
start_test = fecha_corte + timedelta(days=1)
vecm_train = VECM(vecm_df.loc[:fecha_corte], k_ar_diff=2, coint_rank=4, deterministic="n").fit()

# 3) Cálculo de los error-correction terms (ECT)
beta = vecm_train.beta                     # matriz (n_vars × 4)
ects = vecm_df.values.dot(beta)             # resultado (T × 4)
ects_df = pd.DataFrame(
    ects,
    index=vecm_df.index,
    columns=[f"ect_{i+1}" for i in range(beta.shape[1])]
)
# Escalado opcional para comparabilidad
scaler_ect = StandardScaler().fit(ects_df.loc[:fecha_corte])
ects_df.loc[:fecha_corte] = scaler_ect.transform(ects_df.loc[:fecha_corte])
ects_df.loc[start_test:] = scaler_ect.transform(ects_df.loc[start_test:])

# 4) Merge de los ECT en cada DataFrame de activo (antes de reindex/fill global)
for ticker in tickers:
    datasets[ticker] = datasets[ticker].merge(
        ects_df, left_index=True, right_index=True
    )

sp500_raw = yf.download("^GSPC", start=start_date, end=end_date,
                        session=session, progress=False)[['Close']]
sp500_raw.columns = ['sp500']       # renombrar
sp500 = sp500_raw.ffill()           # evitar NaN
# 1) Construimos el DataFrame de las nuevas variables
#    - Aprovechamos rates (DGS3MO, DGS1, DGS2, DGS5, DGS10, DBAA, DAAA)
#    - Usamos sp500_df para '^GSPC'
#    - Usamos commodities y fx_raw que ya tienes
vecm2_df = pd.concat([
    rates[['rate_3m','rate_1y','rate_2y','rate_5y','rate_10y','moody_baa','moody_aaa']],
    sp500,           # '^GSPC'
    commodities['gold'],                        # 'GC=F'
    commodities['wti'],                         # 'CL=F'
    fx_raw[['eur_usd','cny_usd']]               # 'EURUSD=X', 'CNYUSD=X'
], axis=1).dropna()

# 2) Entrenamos el VECM sobre train (mismo fecha_corte y start_test que antes)
vecm2_train = VECM(
    vecm2_df.loc[:fecha_corte],
    k_ar_diff=2,        # lag order = 2
    coint_rank=4,       # rango cointegración = 4
    deterministic="n"
).fit()

# 3) Calculamos los ECTs y los escalamos
beta2 = vecm2_train.beta                    # (n_vars × 4)
ects2 = vecm2_df.values.dot(beta2)          # (T × 4)
ects2_df = pd.DataFrame(
    ects2,
    index=vecm2_df.index,
    columns=[f"ect2_{i+1}" for i in range(beta2.shape[1])]
)

# Escalado con los datos de train solamente
scaler2 = StandardScaler().fit(ects2_df.loc[:fecha_corte])
ects2_df.loc[:fecha_corte] = scaler2.transform(ects2_df.loc[:fecha_corte])
ects2_df.loc[start_test:] = scaler2.transform(ects2_df.loc[start_test:])

# 4) Merge de los nuevos ECTs en cada DataFrame de activo
for ticker in tickers:
    datasets[ticker] = datasets[ticker].merge(
        ects2_df,
        left_index=True, right_index=True
    )


# --------------------
# 5. ÍNDICE UNIÓN + REINDEX + RELLENO + MÁSCARAS DE VALIDEZ
# --------------------
# 5.1. Para cada ticker, guardamos su primera fecha real (sin fill)
first_valid_idx = {t: df.index.min() for t, df in datasets.items()}

# 5.2. Construimos la unión de todos los índices junto con rates
union_idx = sorted(
    set(rates.index)
    .union(*[df.index for df in datasets.values()])
)

# 5.3. Reindexamos y rellenamos (forward then backward) cada DataFrame
for t in list(datasets):
    df = datasets[t].reindex(union_idx).ffill().bfill()
    # merge de rates (incluye rate_10y, rate_3m, vix)
    datasets[t] = df.merge(rates.reindex(union_idx).ffill(), 
                           left_index=True, right_index=True)

# 5.4. Dividir train/test por fecha
n         = len(union_idx)
cut       = int(n * 0.8)
train_idx = union_idx[:cut]
test_idx  = union_idx[cut:]

# Descarga SP500 completo una sola vez (antes del env)
sp500_df = yf.download("^GSPC", start=start_date, end=end_date, session=session, progress=False)
# Aplana MultiIndex / minúsculas
if isinstance(sp500_df.columns, pd.MultiIndex):
    sp500_df.columns = [" ".join(map(str,c)).strip().lower()
                        for c in sp500_df.columns]
else:
    sp500_df.columns = [c.lower().strip() for c in sp500_df.columns]
# Elegir adj close > close
close_cols = [c for c in sp500_df.columns if "close" in c]
adj_cols   = [c for c in close_cols       if "adj"   in c]
col_sp     = adj_cols[0] if adj_cols else close_cols[0]


sp = sp500_df[col_sp].ffill().reindex(union_idx).ffill()
sp_ret_full = sp.pct_change().fillna(0).values


sp = sp500_df[col_sp].ffill().reindex(union_idx).ffill()
sp_ret_full = sp.pct_change().fillna(0).values  # array alineado a union_idx
#separa en train/test:
sp_ret_train = sp_ret_full[:cut]
sp_ret_test  = sp_ret_full[cut:]
# Creamos una Serie de retornos de SP500 alineada
sp_ret_series = pd.Series(sp_ret_full, index=union_idx)
# ─── 5.4.1. CÁLCULO DE FEATURES DE CLUSTERING EN VENTANAS DE 50 DÍAS ───
for t, df in datasets.items():
    ret = df['close'].pct_change()

    clustering_feats = [
        'mean_return_cluster', 'volatility_cluster', 'cvar95_cluster',
        'skewness_cluster', 'beta_cluster', 'corr_cluster',
        'sharpe_cluster', 'tracking_error_cluster', 'ir_cluster'
    ]
    # inicializar
    for c in clustering_feats:
        df[c] = np.nan

    n = len(df)
    for start in range(50, n, 50):
        win   = slice(start-50, start)
        block = slice(start, min(start + 50, n))

        win_ret = ret.iloc[win]
        sp_win  = sp_ret_series.iloc[win]

        # 1) Retorno medio y volatilidad
        m_ret = win_ret.mean()
        vol   = win_ret.std()

        # 2) CVaR95
        var95 = win_ret.quantile(0.05)
        cvar  = win_ret[win_ret <= var95].mean()

        # 3) Skewness
        skew  = win_ret.skew()

        # 4) Beta y correlación 
        cov = win_ret.cov(sp_win)
        var_sp = sp_win.var()
        beta   = cov / var_sp if (var_sp and not np.isnan(var_sp)) else 0
        std_ret = win_ret.std()
        std_sp  = sp_win.std()
        corr = (
            win_ret.corr(sp_win)
            if (std_ret and std_sp and not np.isnan(std_ret) and not np.isnan(std_sp))
            else 0
        )

        # 5) Sharpe 
        sharpe = m_ret / vol if (vol and not np.isnan(vol)) else 0

        # 6) Tracking Error e Information Ratio 
        active = win_ret - sp_win
        te     = active.std()
        ir     = (active.mean() / te) if (te and not np.isnan(te)) else 0

        # asignar toda la ventana “block”
        idx_block = df.index[block]
        df.loc[idx_block, 'mean_return_cluster']    = m_ret
        df.loc[idx_block, 'volatility_cluster']     = vol
        df.loc[idx_block, 'cvar95_cluster']         = cvar
        df.loc[idx_block, 'skewness_cluster']       = skew
        df.loc[idx_block, 'beta_cluster']           = beta
        df.loc[idx_block, 'corr_cluster']           = corr
        df.loc[idx_block, 'sharpe_cluster']         = sharpe
        df.loc[idx_block, 'tracking_error_cluster'] = te
        df.loc[idx_block, 'ir_cluster']             = ir

    # rellenar primeros 50 días con primer valor válido
    for c in clustering_feats:
        df[c] = df[c].bfill()

# ---------------------------------------------------------
# CORRECCIÓN LEAK CLUSTERING: Fit solo en TRAIN
# ---------------------------------------------------------
from sklearn.cluster import KMeans

numero_de_clusters = 5

# Definimos fecha de corte igual que para el split global (basado en índices)
# Nota: 'train_idx' ya lo calculaste en la sección 5.4
train_cutoff_date = train_idx[-1] 

for t, df in datasets.items():
    features = df[clustering_feats].dropna()
    
    if len(features) < numero_de_clusters:
        df['regimen_mercado'] = 0
    else:
        # 1. Separar features en Train y 'Resto' para el fit
        # Usamos solo lo que esté dentro de train_idx para ajustar el modelo
        feat_train = features.loc[features.index.intersection(train_idx)]
        
        if len(feat_train) < numero_de_clusters:
             # Fallback si no hay suficientes datos en train
             kmeans = KMeans(n_clusters=numero_de_clusters, random_state=42)
             kmeans.fit(features) 
        else:
             # 2. FIT solo con datos pasados
             kmeans = KMeans(n_clusters=numero_de_clusters, random_state=42)
             kmeans.fit(feat_train)
        
        # 3. PREDICT sobre todo el dataset (Train + Test)
        # Esto asigna el cluster basándose en los centroides aprendidos en el pasado
        labels = kmeans.predict(features)
        
        cluster_labels = pd.Series(labels, index=features.index)
        
        # Asigna cluster, rellenando días vacíos
        df['regimen_mercado'] = cluster_labels.reindex(df.index).ffill().astype(int)
        
    datasets[t] = df

for t, df in datasets.items():
    dummies = pd.get_dummies(df['regimen_mercado'], prefix='regime')
    df = pd.concat([df, dummies], axis=1)
    datasets[t] = df
#dependencia de la cola, como se comporta los activos en la cola de distribución cuando el sp500 tambien lo está
windowcola = 252   # ventana rolling de 1 año (si los datos son diarios)
alphacola = 0.05   # cuantil para la cola
#cola inferior
for t, df in datasets.items():
    # Asegúrate de tener retornos diarios del activo y del SP500
    ret_asset = df['return']
    ret_sp = sp_ret_series  

    tail_dep = np.full(len(df), np.nan)
    for i in range(windowcola, len(df)):
        r1 = ret_asset.iloc[i-windowcola:i]
        r2 = ret_sp.iloc[i-windowcola:i]
        q1 = np.quantile(r1, alphacola)
        q2 = np.quantile(r2, alphacola)
        # Calcula la frecuencia conjunta de estar ambos en su cola inferior
        freq = np.mean((r1 <= q1) & (r2 <= q2))
        tail_dep[i] = freq

    df['tail_dep_5pct'] = tail_dep
    datasets[t] = df
#cola superior
for t, df in datasets.items():
    
    ret_asset = df['return']
    ret_sp = sp_ret_series  

    tail_dep = np.full(len(df), np.nan)
    for i in range(windowcola, len(df)):
        r1 = ret_asset.iloc[i-windowcola:i]
        r2 = ret_sp.iloc[i-windowcola:i]
        q1 = np.quantile(r1, 1-alphacola)
        q2 = np.quantile(r2, 1-alphacola)
        # Calcula la frecuencia conjunta de estar ambos en su cola inferior
        freq = np.mean((r1 >= q1) & (r2 >= q2))
        tail_dep[i] = freq

    df['tail_dep_95pct'] = tail_dep
    datasets[t] = df



# 5.5. Construir máscaras de validez para cada ticker (antes, sin CASH)
tickers_list = list(datasets.keys())  # length = m, sin CASH
valid_mask = np.zeros((n, len(tickers_list)), dtype=bool)
for j, t in enumerate(tickers_list):
    fv = first_valid_idx[t]
    valid_mask[:, j] = np.array(union_idx) >= fv

# ─── NUEVO: añadir columna CASH siempre válida ───
# ahora tenemos m tickers + 1 CASH = n_activos
valid_mask = np.hstack([
    valid_mask,
    np.ones((n, 1), dtype=bool)
])
# actualizamos lista de activos
activos_finales = tickers_list + ["CASH"]

# 5.6. Partir train/test incluyendo CASH
valid_mask_train = valid_mask[:cut]
valid_mask_test  = valid_mask[cut:]


# 5.6. Obtener train_dfs y test_dfs con los índices seleccionados
train_dfs = {t: df.loc[train_idx] for t, df in datasets.items()}
test_dfs  = {t: df.loc[test_idx]  for t, df in datasets.items()}

# --------------------
# 5.7. ESCALADO, CREACIÓN DE TENSORES Y CORRECCIÓN DE DATA LEAK
# --------------------
feat_names = train_dfs[next(iter(train_dfs))].columns.tolist()
scaled_tr, scaled_te = [], []

for t in tickers_list:
    # 1. Obtener valores y limpiar infinitos
    tr = train_dfs[t].replace([np.inf, -np.inf], np.nan).fillna(0).values
    te = test_dfs[t].replace([np.inf, -np.inf], np.nan).fillna(0).values
    
    # 2. Ajustar Scaler solo con Train
    sc = StandardScaler().fit(tr)
    tr_sc = sc.transform(tr)
    te_sc = sc.transform(te)
    
    # --- CORRECCIÓN CRÍTICA (DATA LEAK) ---
    # Desplazamos las features 1 día hacia abajo.
    # Lo que ocurrió en t-1 se convierte en la observación disponible en t.
    
    # Shift para Train
    tr_shifted = np.zeros_like(tr_sc)
    tr_shifted[1:] = tr_sc[:-1]  # La fila i toma el valor de i-1
    tr_shifted[0] = 0            # El primer día no tiene "ayer", rellenamos con 0
    
    # Shift para Test
    te_shifted = np.zeros_like(te_sc)
    te_shifted[1:] = te_sc[:-1]
    te_shifted[0] = 0
    
    scaled_tr.append(tr_shifted)
    scaled_te.append(te_shifted)

# Añadir CASH como columna neutra (sin shift necesario pues es todo ceros, pero por consistencia)
num_feats = len(feat_names)
scaled_tr.append(np.zeros((cut, num_feats)))
scaled_te.append(np.zeros((n-cut, num_feats)))

# Apilar para crear el tensor de mercado (Observaciones)
# Ahora market_train[t] contiene la info del cierre de t-1
market_train = np.stack(scaled_tr, axis=1)  # shape (cut, n_assets+1, n_feats)
market_test  = np.stack(scaled_te, axis=1)

# --- PRECIOS SIN SHIFT ---
# Los precios NO se desplazan porque 'PortfolioEnv' usa el índice 'i' 
# para calcular el retorno real que ocurre ese día.
prices_train = np.column_stack(
    [train_dfs[t]['close'].values for t in tickers_list] + [np.ones(cut)]
)
prices_test  = np.column_stack(
    [test_dfs[t]['close'].values  for t in tickers_list] + [np.ones(n-cut)]
)

# Actualizamos lista final de activos
activos_finales = tickers_list + ["CASH"]

print("Data leak corregido: Features desplazadas t-1. Dimensiones mantenidas.")



# --------------------
# Funciones auxiliares (deben ir antes de PortfolioEnv)
# --------------------

def normalize_action(a, cash_index=-1, max_cash=0.2):
   
    # 1) Softmax (numéricamente estable)
    e = np.exp(a - np.max(a))
    w = e / e.sum()

    # 2) Si excede el cash cap, lo recortamos y redistribuimos el resto
    if w[cash_index] > max_cash:
        # fijamos cash al máximo
        w[cash_index] = max_cash

        # redistribuimos 1-max_cash entre los otros activos
        mask_otros = np.arange(len(w)) != cash_index
        suma_otros = w[mask_otros].sum()
        if suma_otros > 0:
            w[mask_otros] = w[mask_otros] / suma_otros * (1 - max_cash)
        else:
            # en caso extremo, repartir uniformemente entre no-cash
            n_otros = mask_otros.sum()
            w[mask_otros] = (1 - max_cash) / n_otros

    return w


def compute_reward(net_return, sp_return, past_net, past_port, past_sp, current_wealth, max_wealth, penalty_coef=0.5, cvar_coef=1, cvar_window=50, alpha=5, minimoperiodos=10):
    # Recompensa base: logaritmo del retorno neto de costes
    eps = 1e-8
    # 1) Normalización del retorno neto
    periodos= np.array(past_net)
    if len(periodos)< minimoperiodos:
        rolling_std = 1.0
    else:
        raw_std = np.std(past_net) if len(past_net)>1 else 1.0
        rolling_std = max(raw_std, 1e-3)    # nunca por debajo de 0.001
    norm_r = net_return / (rolling_std+eps)
    norm_r = np.maximum(norm_r, -0.999)
    base   = np.log1p(norm_r)


    # Penalización por underperformance: si el SP500 sube más que tu cartera
    # tras rellenar los buffers:
    cum_port = np.prod([1 + x for x in past_port]) - 1
    cum_sp   = np.prod([1 + x for x in past_sp])   - 1
    diferencial = cum_sp - cum_port
    diferencial_norm= diferencial/(rolling_std+eps)
    penalty   = penalty_coef * diferencial_norm
    # Penalización extra por drawdown severo
    drawdown = (current_wealth - max_wealth) / max_wealth
    drawdown =drawdown/(rolling_std+eps)
    dd_pen   = drawdown * 100 if drawdown < -0.2 else 0.0
    #CVAR penalizacion cola 5% peores rendimientos
    histcvar = np.array(past_net)
    if len(histcvar) >= cvar_window:
        # recortamos a la ventana más reciente
        tail = np.percentile(histcvar[-cvar_window:], alpha)
        cvar = histcvar[-cvar_window:][histcvar[-cvar_window:] <= tail].mean()
    else:
        cvar = 0.0

    # 2) Como cvar es negativo (pérdidas), lo convertimos en penalización positiva
    cvar_pen = (cvar_coef * (-cvar))/(rolling_std+eps)

    return base - penalty - cvar_pen + dd_pen
def compute_corr_matrix(prices: np.ndarray, n_assets: int) -> np.ndarray:
    """
    Dada una matriz de precios shape (T, n_assets),
    devuelve la matriz de correlaciones shape (n_assets, n_assets),
    limpiando NaNs e infinitos.
    """
    # 1) retornos día a día (shape (T-1, n_assets))
    #    Evitamos división por cero con un pequeño epsilon:
    eps = 1e-8
    rets = (prices[1:] - prices[:-1]) / (prices[:-1] + eps)

    # 2) correlación cruzada (shape (n_assets, n_assets))
    corr = np.corrcoef(rets.T)

    # 3) limpieza de NaN/Inf y casting a float32
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr.astype(np.float32)
# --------------------
# 6. ENTORNO GYM CON MÁSCARA
# --------------------
import collections

class PortfolioEnv(gym.Env):
    def __init__(self, market, prices, valid_mask, sp_returns, penalty_coef=0.5, 
                 init_wealth=1.0, tc_coef=0.001, norm_window=50, perf_window=30, window=60):
        super().__init__()
        self.market     = market
        self.prices     = prices
        self.valid_mask = valid_mask
        self.sp_ret     = sp_returns
        self.penalty_coef = penalty_coef
        self.n_steps, self.n_assets, self.n_feats = market.shape
        self.init_w = init_wealth
        self.tc     = tc_coef
        self.past_net = collections.deque(maxlen=50)
        self.window   = window
        
        # Espacios de observación y acción (sin cambios)
        self.observation_space = spaces.Dict({
            'market': spaces.Box(-np.inf, np.inf, shape=(self.n_assets, self.n_feats), dtype=np.float32),
            'drift':  spaces.Box(0.0, 1.0, shape=(self.n_assets,), dtype=np.float32),
            'corr':   spaces.Box(0.0, 1.0, shape=(self.n_assets, self.n_assets), dtype=np.float32),
            'time_idx': spaces.Box(low=0, high=self.n_steps-1, shape=(1,), dtype=np.int32),                     
        })
        self.action_space = spaces.Box(0, 1, shape=(self.n_assets,), dtype=np.float32)
        
        self.past_net  = collections.deque(maxlen=norm_window)
        self.past_port = collections.deque(maxlen=perf_window)
        self.past_sp   = collections.deque(maxlen=perf_window)
        self.hist = []
        global HIST_GLOBAL
        HIST_GLOBAL.clear()
        self.reset()

    def _get_corr_matrix(self, current_step):
        """
        Helper para obtener la correlación SIN mirar al futuro.
        Si estamos en el paso 'current_step', y las features 'market' vienen de t-1,
        la correlación también debe calcularse solo hasta t-1 (prices[current_step-1]).
        """
        # El índice en prices 'current_step' es HOY. Queremos hasta AYER.
        # Slice en python [a:b] llega hasta b-1.
        # Por tanto, prices[:current_step] toma índices 0...current_step-1 (AYER).
        
        end_idx = current_step 
        start_idx = max(0, end_idx - self.window)
        
        if end_idx <= 0:
            # Al inicio no hay historial suficiente, devolvemos identidad o ceros
            return np.eye(self.n_assets, dtype=np.float32)
            
        prices_window = self.prices[start_idx:end_idx]
        
        # Si la ventana es muy pequeña (ej. paso 1), compute_corr_matrix podría fallar o dar ruido
        if len(prices_window) < 2:
             return np.eye(self.n_assets, dtype=np.float32)
             
        return compute_corr_matrix(prices_window, self.n_assets)

    def reset(self):
        self.i        = 0
        self.wealth   = self.init_w
        self.max_w    = self.init_w
        
        valid0 = self.valid_mask[0].astype(float)
        self.prev_w  = valid0 / valid0.sum()
        
        self.past_net.clear()
        self.past_port.clear()
        self.past_sp.clear()
        self.hist    = []
        
        # ### CORRECCIÓN 1: Correlación inicial ###
        # En t=0 no hay pasado, usamos identidad o ventana pre-calculada si hubiera historial
        corr = np.eye(self.n_assets, dtype=np.float32) 
        
        obs = {
            'market': self.market[0],
            'drift':  self.prev_w.copy(),
            'corr':   corr,
            'time_idx': np.array([self.i], dtype=np.int32),
        }
        
        obs['market'] = np.nan_to_num(obs['market']).astype(np.float32)
        obs['drift']  = np.nan_to_num(obs['drift']).astype(np.float32)
        obs['corr']   = np.nan_to_num(obs['corr']).astype(np.float32)
        return obs
    
    def step(self, action_logits):
        mask = self.valid_mask[self.i].astype(float)
        action = action_logits * mask

        # 2) Calcular drift y retornos
        if self.i > 0:
            p0 = self.prices[self.i - 1]
            p1 = self.prices[self.i]
            # Retorno de HOY
            r = (p1 - p0) / (p0 + 1e-8)
        else:
            r = np.zeros(self.n_assets)

        # Crecimiento intra-día de los pesos
        g = 1 + r
        expo = self.prev_w * g
        S = expo.sum()
        w_drift = expo / (S + 1e-8)

        # 3) Acción del Agente
        w_new = normalize_action(action, cash_index=-1, max_cash=0.2)

        # 4) Costes de transacción
        # Se pagan por cambiar desde la cartera 'drifted' a la 'new'
        tc = self.tc * np.abs(w_new - w_drift).sum()

        # 5) Actualizar prev_w para mañana
        self.prev_w_old = self.prev_w.copy() # Guardar para log
        self.prev_w = w_new.copy()

        # ### CORRECCIÓN 2: Cálculo del Retorno ###
        # El retorno bruto de la cartera es: Pesos al INICIO del día * Retorno del día.
        # Usar 'w_drift' sería usar los pesos al FINAL del día (error matemático).
        port_r = np.dot(self.prev_w_old, r) 
        
        net_r = port_r - tc
        sp_r = self.sp_ret[self.i]

        # 7) Buffers
        self.past_net.append(net_r)
        self.past_port.append(net_r)
        self.past_sp.append(sp_r)

        # 8) Recompensa
        rew = compute_reward(net_return=net_r, sp_return=sp_r, past_net=self.past_net, 
                             past_port=self.past_port, past_sp=self.past_sp, 
                             current_wealth=self.wealth, max_wealth=self.max_w, 
                             penalty_coef=self.penalty_coef, cvar_coef=1, 
                             cvar_window=50, alpha=5)

        # 9) Wealth
        self.wealth *= (1 + net_r)
        self.max_w = max(self.max_w, self.wealth)

        self.hist.append({'return': net_r, 'wealth': self.wealth, 'weights_drift': w_drift, 'weights': w_new})
        global HIST_GLOBAL
        HIST_GLOBAL.append({'return': net_r, 'wealth': self.wealth, 'weights_drift': w_drift, 'weights': w_new})

        # 11) Next Step
        self.i += 1
        done = self.i >= self.n_steps - 1
        
        if done:
            empty_obs = {
                'market': np.zeros((self.n_assets, self.n_feats), dtype=np.float32),
                'drift':  np.zeros(self.n_assets, dtype=np.float32),
                'corr':   np.zeros((self.n_assets, self.n_assets), dtype=np.float32),
                'time_idx': np.array([self.i], dtype=np.int32),
            }
            return empty_obs, rew, done, {}
        
        # ### CORRECCIÓN 1 (parte 2): Usar helper corregido ###
        # Usamos el helper que corta en 'self.i' (excluyendo el precio de self.i)
        corr = self._get_corr_matrix(self.i)
        
        obs = {
            'market': self.market[self.i],
            'drift':  w_drift, # El drift sí es conocido (es mi cartera actual)
            'corr':   corr,
            'time_idx': np.array([self.i], dtype=np.int32),
        }
        
        obs['market'] = np.nan_to_num(obs['market']).astype(np.float32)
        obs['drift']  = np.nan_to_num(obs['drift']).astype(np.float32)
        obs['corr']   = np.nan_to_num(obs['corr']).astype(np.float32)
        
        return obs, rew, done, {}

# --------------------
# 7. ENTRENAMIENTO / EVALUACIÓN
# --------------------

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentActorCriticPolicy
# 1) Feature extractor: del state (n_assets, n_feats) al vector flatten + encod.
# ── 1) Subclase de la política LSTM para usar SquashedDiagGaussian ─────────
import torch.nn as nn
from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentActorCriticPolicy

class SquashedGaussianLSTMWithDropoutPolicy(RecurrentActorCriticPolicy):
    """
    Igual que RecurrentActorCriticPolicy, pero:
     - fuerza SquashedDiagGaussianDistribution
     - añade dropout entre capas de la LSTM
    """
    def __init__(self, *args, lstm_dropout: float = 0.1, **kwargs):
        # Guardamos el dropout deseado antes de llamar al init base
        self.lstm_dropout = lstm_dropout
        super().__init__(*args, **kwargs)
        # Forzamos la distribución de salida
        self.dist_class = SquashedDiagGaussianDistribution

    def _build_lstm(self, input_dim: int, lstm_hidden_size: int, n_lstm_layers: int):
        """
        Este método es llamado internamente por __init__ tras crear
        el extractor de características y la MLP. Aquí definimos
        la LSTM con dropout entre capas (solo si n_lstm_layers>1).
        """
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=self.lstm_dropout if n_lstm_layers > 1 else 0.0,
        )
    def forward_actor(self, features, lstm_states, episode_start, **kwargs):
        # Paso 1: Pasa la salida del Transformer por la LSTM del actor
        lstm_output, lstm_states = self.lstm_actor(features, lstm_states, episode_start)
        # Paso 2: Pasa la salida de la LSTM por el MLP de la policy
        latent_pi = self.mlp_extractor.policy_net(lstm_output)
        return latent_pi, lstm_states

    def forward_critic(self, features, lstm_states, episode_start, **kwargs):
        # Igual para el crítico
        lstm_output, lstm_states = self.lstm_critic(features, lstm_states, episode_start)
        latent_vf = self.mlp_extractor.value_net(lstm_output)
        return latent_vf, lstm_states

    def extract_features(self, obs):
        
        return super().extract_features(obs)

import math
def sinusoidal_time_embedding(time_idx, d_model):
    """
    Devuelve el embedding sinusoidal (batch, d_model) para los índices de tiempo dados.
    """
    device = time_idx.device
    # (batch,)
    position = time_idx.float().unsqueeze(1)  # (batch, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-np.log(10000.0) / d_model))
    pe = torch.zeros((time_idx.shape[0], d_model), device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def contar_numero_dias(index_fechas):
    """
    Recibe una lista, array o pandas Index de fechas (datetime, pd.Timestamp, strings...).
    Devuelve el número de días únicos.
    """
    import pandas as pd
    fechas = pd.to_datetime(index_fechas)         # Asegura tipo fecha
    dias_unicos = fechas.normalize().unique()     # Normaliza a día y elimina duplicados
    return len(dias_unicos)
num_dias = contar_numero_dias(union_idx)
class SimpleMLPExtractor(BaseFeaturesExtractor):
    """
    Aplana todas las observaciones (market, drift, corr) y usa una MLP densa.
    """
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        
        # Calculamos dimensiones de entrada
        n_assets, n_feats = observation_space.spaces['market'].shape
        
        # Flatten sizes:
        # market: n_assets * n_feats
        # drift:  n_assets
        # corr:   n_assets * n_assets
        input_dim = (n_assets * n_feats) + n_assets + (n_assets * n_assets)
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Concatenamos todo en un vector gigante
        market = observations['market'].flatten(start_dim=1)
        drift  = observations['drift']
        corr   = observations['corr'].flatten(start_dim=1)
        
        x = torch.cat([market, drift, corr], dim=1)
        return self.net(x)    


# ── 3) policy_kwargs sin dist_class ─────────────────────────────────────────
policy_kwargs = dict(
    features_extractor_class=SimpleMLPExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[64,62], vf=[64, 32]),
    lstm_hidden_size=128,
    n_lstm_layers=2,
    lstm_dropout=0
)

from stable_baselines3.common.vec_env import VecNormalize

# Pasamos también la máscara al crear el env
env_tr = DummyVecEnv([lambda: PortfolioEnv(market_train, prices_train, valid_mask_train, sp_returns=sp_ret_train, penalty_coef=0.5)])
#normalizacion para entrenamiento estable
env_tr = VecNormalize(
     env_tr,
     norm_obs=False,       # normaliza observaciones
     norm_reward=True,    # normaliza recompensas
      )     # recorta observaciones normalizadas a ±10σ
model = RecurrentPPO(SquashedGaussianLSTMWithDropoutPolicy, env_tr, policy_kwargs=policy_kwargs, verbose=1, ent_coef=1.5210089527581027e-05, learning_rate=3.3896027707827866e-06, 
                     batch_size=512, n_steps=768, n_epochs=20, gamma=0.8587760378271281, clip_range=0.14188380242544762, clip_range_vf= 0.1, gae_lambda=0.9208035456282112, max_grad_norm=0.9562075837093198, vf_coef=0.8139391556754874,  device="cuda")

model.learn(total_timesteps=20000)
model.save("ppo_portfolio_model_TRFMLSTM3")

# 1. Construimos el entorno de evaluación (NO training para la normalización)
base_env_ev = PortfolioEnv(market_test, prices_test, valid_mask_test, sp_returns=sp_ret_test, penalty_coef=0.5)
env_ev = DummyVecEnv([lambda: base_env_ev])

# 2. Normalización, compartiendo estadísticas con el entorno de entrenamiento
env_ev = VecNormalize(
    env_ev,
    norm_obs=False,
    norm_reward=True,
    training=False  # Muy importante: en test, no actualizar stats
)
env_ev.ret_rms = env_tr.ret_rms  # Copiamos estadísticas de recompensas
env_ev.gamma = env_tr.gamma

print(f"market_test.shape: {market_test.shape}")  # ¿Cuántos pasos deberías tener en test?

obs = env_ev.reset()
state = None
episode_start = [True]
done = [False]   # <-- ¡ojo! debe ser lista para VecEnv
total_reward = 0
step_count = 0

while not done[0]:
    act, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=True)
    obs, reward, done, _ = env_ev.step(act)
    total_reward += reward[0]
    episode_start = [done[0]]
    step_count += 1

print(f"Número de pasos recorridos en test: {step_count}")

portfolio_env_real = env_ev.venv.envs[0]
hist = pd.DataFrame(HIST_GLOBAL)

if hist.empty:
    print("¡OJO! El historial sigue vacío tras la evaluación. Debug info:")
    print("¿Quizá el entorno termina en el primer paso?")
    print("Posibles causas: longitud de test, error en lógica de step/reset, máscara de activos sin válidos, etc.")
    # Salir o lanzar excepción aquí si lo deseas
    import sys; sys.exit(1)
else:
    hist.index = test_idx[:len(hist)]
    print("Columnas en hist:", hist.columns.tolist())
    print(hist.head())
# --------------------
# BENCHMARK SP500 y RETORNOS (sin volver a descargar)
# --------------------
# convierto el array de retornos de SP500 en la etapa de test en Series
sp_series   = pd.Series(sp_ret_test, index=test_idx)
print("Columnas en hist:", hist.columns.tolist())
print("Número de filas en hist:", len(hist))

# calculo los retornos de tu cartera y alineo fechas
port_ret    = hist['return']
sp_aligned  = sp_series.loc[port_ret.index]

# exceso y métricas
excess       = port_ret.values - sp_aligned.values
mean_exc_ann = excess.mean() * 252
te_ann       = excess.std() * np.sqrt(252)
IR           = mean_exc_ann / te_ann

ret_ann = (np.prod(1+port_ret)**(252/len(port_ret)) -1)
vol_ann = port_ret.std()*np.sqrt(252)
rf_10y  = rates['rate_10y'].iloc[-1]/100
SR      = (ret_ann - rf_10y)/vol_ann
max_dd  = (hist['wealth'].cummax()-hist['wealth']).max()

print(f"Wealth final: {hist['wealth'].iloc[-1]:.4f}")
print(f"Sharpe ratio anual: {SR:.2f}")
print(f"Tracking error anual: {te_ann:.4f}")
print(f"Information Ratio: {IR:.2f}")
print(f"Máximo drawdown: {max_dd:.2%}")
print(f'rentabilidad anual: {ret_ann:.4f}')
print(f'volatilidad anual: {vol_ann:.4f}')
# --------------------
# 6. GRAFICAS
# --------------------
# 6.1 Valor vs SP500 (alineado y acumulado)
sp_cum = (1 + sp_aligned).cumprod()

plt.figure(figsize=(10,5))
plt.plot(hist.index,   hist['wealth'], label="Portafolio")
plt.plot(sp_cum.index, sp_cum.values, label="SP500")
plt.title("Valor del portafolio VS SP500")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 6.2 Evolución de pesos (Top10)
weights = np.vstack(hist['weights'].values)
wdf     = pd.DataFrame(weights, columns=activos_finales, index=hist.index)
top10   = wdf.mean().sort_values(ascending=False).head(10).index

plt.figure(figsize=(12,6))
for t in top10:
    plt.plot(wdf.index, wdf[t], label=t)
plt.title("Pesos asignados (Top 10)")
plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

# 6.3 Bonos y efectivo
plt.figure(figsize=(10,4))
for b in ["TLT","IEF","SHY"]:
    if b in wdf:
        plt.plot(wdf.index, wdf[b], label=f"{b} (Bono)")
plt.plot(wdf.index, wdf['CASH'], label="CASH")
plt.title("Exposición a bonos y efectivo")
plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

# Guardar pesos
wdf.to_csv("pesos_portafolio_TRFMLSTM3.csv")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# 1. Preparar datos del Benchmark (S&P 500)
# Aseguramos que sp_aligned tiene el mismo índice que hist
sp_aligned = sp_series.loc[hist.index] 
sp_cum_ret = (1 + sp_aligned).cumprod()
sp_peak = sp_cum_ret.cummax()
sp_drawdown = (sp_cum_ret - sp_peak) / sp_peak

# 2. Preparar datos del Portafolio (Modelo)
port_peak = hist['wealth'].cummax()
port_drawdown = (hist['wealth'] - port_peak) / port_peak

# 3. Graficar
plt.figure(figsize=(12, 6))

# Área del S&P 500 (Gris/Naranja suave)
plt.fill_between(hist.index, sp_drawdown, 0, color='gray', alpha=0.3, label='S&P 500 Drawdown')
plt.plot(hist.index, sp_drawdown, color='gray', alpha=0.6, linewidth=1)

# Línea del Portafolio (Azul corporativo)
plt.plot(hist.index, port_drawdown, label='Modelo DRL (Transformer+LSTM)', color='#1f77b4', linewidth=1.5)

# Formato
plt.title('Análisis de Drawdown Underwater: Modelo vs Benchmark', fontsize=14, fontweight='bold')
plt.ylabel('Drawdown (%)')
plt.xlabel('Fecha')
plt.legend(loc='lower right')
plt.grid(True, which='major', linestyle='--', alpha=0.6)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0)) # Formato porcentaje

plt.tight_layout()
plt.show()