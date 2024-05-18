import streamlit as st
import aiohttp
from plotly import graph_objs as go
import pandas as pd
from tradingview_ta import TA_Handler, Interval
from io import StringIO
import asyncio
import yfinance as yf

st.title("Cryptex")


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


async def fetch(session, url, params, headers):
    async with session.get(url, params=params, headers=headers) as response:
        response.raise_for_status()
        return await response.text()


async def get_data1(ticker):
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
        response = await fetch(session, url, params, headers)
        df = pd.read_table(StringIO(response), sep=',').dropna()
    return df


async def get_data(ticker):
    s = yf.Ticker(ticker)
    df = s.history('10000d')
    df.index = pd.to_datetime(df.index)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date'}, inplace=True)
    return df



crypto = ('Свой вариант ниже',
          'BTC-USD', 'ETH-USD', 'BNB-USD', 'MATIC-USD', 'XRP-USD', 'SOL-USD', 'LTC-USD', 'CFX-USD', 'DOGE-USD',
          'STX-USD', 'LINK-USD', 'MATIC-USD', 'SHIB-USD', 'ADA-USD', 'STETH-USD', 'HEX-USD', 'DOT-USD', 'AVAX-USD',
          'TRX-USD',
          'LINK-USD', 'ATOM-USD',)


async def showing_data():
    dfi = crypto.index('BTC-USD')
    selected_stock = st.selectbox("Выберите криптовалюту", crypto, index=dfi)
    selected_stock1 = st.text_input('Ваша криптовалюта (*symbol*-USD)')
    if selected_stock == 'Свой вариант ниже':
        selected_stock = selected_stock1

    data_tbl = await get_data(selected_stock)
    k = data_tbl.iloc[::-1]
    st.subheader(f'Данные {selected_stock}')
    st.dataframe(k)
    fig = go.Figure()
    gr_tp = st.selectbox('Выберите тип графика', ('Свечи', 'Линия'))
    gr_d = st.slider('Выберите количество дней для отображения', 1, 365, value=100)
    data = data_tbl.tail(gr_d)
    if gr_tp == 'Свечи':
        fig.add_trace(
            go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
                           name='Цена закрытия'))
        fig.layout.update(title_text=f'График {selected_stock}', xaxis_rangeslider_visible=True)
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1 месяц",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6 месяцев",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="1 год",
                             step="year",
                             stepmode="backward"),
                        dict(label='Все',
                             step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )
        st.plotly_chart(fig)

    if gr_tp == 'Линия':
        fig.add_trace(
            go.Scatter(x=data['Date'], y=data['Close'], line=dict(color="#0ccbcf"), name='Цена закрытия'))
        fig.layout.update(title_text=f'График {selected_stock}', xaxis_rangeslider_visible=True)
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1 месяц",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6 месяцев",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="1 год",
                             step="year",
                             stepmode="backward"),
                        dict(label='Все',
                             step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )
        st.plotly_chart(fig)

    sm, ind = await get_tv_crypto(selected_stock)

    # indf = f'RSI {ind["RSI"]}\nMACD {ind["MACD.macd"]}\nEMA30 {ind["EMA30"]}'

    total = sm["BUY"] + sm["SELL"] + sm["NEUTRAL"]

    st.write(f'Рекомендация: {sm["RECOMMENDATION"]}')
    st.write(f':green[покупать]: {sm["BUY"]}/{total}')
    st.write(f':red[продавать]: {sm["SELL"]}/{total}')
    st.write(f'держать: {sm["NEUTRAL"]}/{total}')
    # st.write(indf)
    st.write(ind)
    st.caption(
        "Информация на сайте не является индивидуальной инвестиционной рекомендацией, и финансовые инструменты, упомянутые на нем, могут не соответствовать Вашему инвестиционному профилю и инвестиционным целям (ожиданиям). Определение соответствия финансового инструмента либо операции Вашим интересам, инвестиционным целям, инвестиционному горизонту и уровню допустимого риска является Вашей задачей.", )


asyncio.run(showing_data())
