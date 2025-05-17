import pmdarima as pm
import pandas as pd

df = pd.read_csv('data.tsv', sep='\t')
df.set_index('fielddate', inplace=True)
print(df.shape)
df.head()

def train_arima(y, test_size=36):
    y_train, y_test = y[:-test_size], y[-test_size:]

    arima_model = pm.auto_arima(
        y_train,

        start_p=0, start_q=0, max_p=5, max_q=5,
        start_P=1, start_Q=1, max_P=3, max_Q=3,
        seasonal=True, m=12,
        max_D=2, max_d=2,
        max_order=10,
        information_criterion='bic',

        trace=True)

    arima_model.fit(y_train)

    return arima_model

model_rtrd = train_arima(df.RTRD.values)
model_usd = train_arima(df.USD.values)
model_rtrd.summary()
model_usd.summary()