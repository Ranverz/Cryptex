import streamlit as st
from datetime import date
import aiohttp
from plotly import graph_objs as go
import pandas as pd
from tradingview_ta import TA_Handler, Interval
from io import StringIO, BytesIO
import asyncio
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from datetime import datetime, timedelta

feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

lookback = 30
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=128, return_sequences=True, input_shape=(lookback, len(feature_cols))),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units=64),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=32),
    tf.keras.layers.Dense(units=1)
])
model.compile(optimizer='adam', loss='mean_squared_error')


async def get_tv_crypto(ticker):
    tickertv = f'{ticker.split("-")[0]}USDT'
    handlertv = TA_Handler(
        symbol=f"{tickertv}",
        exchange="OKX",
        screener="crypto",
        interval=Interval.INTERVAL_1_MINUTE,
    )
    s = handlertv.get_analysis().summary
    l = handlertv.get_analysis().indicators

    return s, l


async def fetch(session, url, params, headers, ticker):
    async with session.get(url, params=params, headers=headers) as response:
        return await response.text()


async def get_data(ticker):
    url = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.57'
    }
    params = {
        'range': '100000d',
        'interval': '1d',
        'events': 'history'
    }
    async with aiohttp.ClientSession() as session:
        response = await fetch(session, url, params, headers, ticker)
        df = pd.read_csv(StringIO(response)).dropna()
    return df


st.title("Cryptex")

# num_neurons = st.sidebar.slider('Количество нейронов:', 1, 64)
num_epochs = st.sidebar.slider('Количество эпох:', 1, 20, value=10)
batch_size = st.sidebar.slider('Batch size:', 16, 64, value=32)
# activation = st.sidebar.selectbox('Функция активации', ('relu'))

crypto = (
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'MATIC-USD', 'XRP-USD', 'SOL-USD', 'LTC-USD', 'CFX-USD', 'DOGE-USD',
    'STX-USD', 'LINK-USD', 'MAGIC-USD', 'SHIB-USD')

selected_stock = st.selectbox("Выберите криптовалюту", crypto)


async def get_data_and_train(ticker_symbol):
    # Load the data from yfinance
    df = await get_data(ticker_symbol)

    # Create a numpy array of the feature data
    features = np.array(df[feature_cols])
    nn_state = st.text(f'Модель обучается...эпохи:{num_epochs}batch:{batch_size}')

    # Scale the data using a MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)

    # Define the training data and labels
    train_data = []
    train_labels = []
    for i in range(lookback, len(scaled_features)):
        train_data.append(scaled_features[i - lookback:i])
        train_labels.append(scaled_features[i][3])  # target is the 'Close' column of the last day in the window

    train_data, train_labels = np.array(train_data), np.array(train_labels)

    # Train the model
    model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size, verbose=1)  # epochs = 50

    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)
    nn_state.text('')
    return df, scaler


async def get_prediction(ticker, days):
    # Get the ticker symbol selected by the user
    ticker_symbol = ticker

    # Get the data and scaler for the selected cryptocurrency
    df, scaler = await get_data_and_train(ticker_symbol)

    # Predict the next n days' prices

    last_window = scaler.transform(df[-lookback:][feature_cols])
    last_window = np.expand_dims(last_window, axis=0)

    predictions = []
    for i in range(days):
        prediction = model.predict(last_window)
        predicted_price = scaler.inverse_transform([[0, 0, 0, prediction[0][0], 0]])[0][3]
        predictions.append(predicted_price)
        last_window = np.roll(last_window, -1, axis=1)
        last_window[0][-1] = prediction
    # Add the predicted prices to the DataFrame
    next_days = pd.date_range(start=df.index[-1], periods=days + 1, freq='D')[1:]
    for i, price in enumerate(predictions):
        df.loc[next_days[i], 'Close'] = price

    fig1 = go.Figure()
    fig1.add_scattergl(x=df.index, y=df['Close'][:-days], line=dict(color="#37ff00"), name='Реальная цена')
    fig1.add_scattergl(x=df.index[-days:], y=df['Close'][-days:], line=dict(color="#0b6db8"), name='Прогноз')
    fig1.layout.update(title_text=f'', xaxis_rangeslider_visible=True)

    change = df["Close"].iloc[-1] - df["Close"].iloc[-(1 + days)]
    pct_change = ((df["Close"].iloc[-1] / df["Close"].iloc[-(1 + days)]) - 1) * 100
    df_pr = df[['Close']][-days:]
    # Send the predicted prices and plot to the user
    df.drop(index=df.index[-days:], axis=0, inplace=True)

    return predictions, pct_change, change, fig1, df_pr


async def showing_data():
    data_tbl = await get_data(selected_stock)

    st.subheader(f'Данные {selected_stock}')
    st.dataframe(data_tbl)
    data = data_tbl.tail(200)
    fig = go.Figure()
    gr_tp = st.selectbox('Выберите тип графика', ('Свечи', 'Линия'))
    if gr_tp == 'Свечи':
        fig.add_trace(
            go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
                           name='Цена закрытия'))
        fig.layout.update(title_text=f'График {selected_stock} за последние 200 дней', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    if gr_tp == 'Линия':
        fig.add_trace(
            go.Scatter(x=data['Date'], y=data['Close'], line=dict(color="#0ccbcf"), name='Цена закрытия'))
        fig.layout.update(title_text=f'График {selected_stock} за последние 200 дней', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    sm, ind = await get_tv_crypto(selected_stock)

    indf = f'RSI {ind["RSI"]}\nMACD {ind["MACD.macd"]}\nEMA30 {ind["EMA30"]}'

    st.write(f'Рекомендация: {sm["RECOMMENDATION"]}')
    st.write(f'покупать: {sm["BUY"]}')
    st.write(f'продавать: {sm["SELL"]}')
    st.write(f'держать: {sm["NEUTRAL"]}')
    st.write(indf)

    selected_stock_for_pred = st.selectbox("Выберите криптовалюту для прогноза", crypto)
    n_days = st.number_input('Количество дней для прогноза:', value=1)

    pred, pct_ch, ch, fg, df_pre = await get_prediction(selected_stock_for_pred, n_days)
    st.subheader(f'Прогноз для {selected_stock_for_pred}')
    st.dataframe(df_pre)
    st.plotly_chart(fg)

    if ch > 0:
        st.write(
            f'Предполагаемая цена на {(datetime.now() + timedelta(days=n_days)).strftime("%d-%m-%Y")} для {selected_stock_for_pred}:\n{pred[-1]:.9f} $')
        st.write(f'Рост: +{ch:.9f} (+{pct_ch:.3f}%)')
    else:
        st.write(
            f'Предполагаемая цена на {(datetime.now() + timedelta(days=n_days)).strftime("%d-%m-%Y")} для {selected_stock_for_pred}:\n{pred[-1]:.9f} $')
        st.write(f'Падение: {ch:.9f} ({pct_ch:.3f}%)')


asyncio.run(showing_data())
