import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from statsmodels.tsa.arima_model import ARIMA

portfolio = ['CELG', 'PXD', 'WAT', 'LH', 'AMGN', 'AOS', 'EFX', 'CRM', 'NEM', 'JNPR', 'LB', 'CTAS', 'MAT', 'MDLZ', 'VLO', 'APH', 'ADM', 'MLM', 'BK', 'NOV', 'BDX', 'RRC', 'IVZ', 'ED', 'SBUX', 'GRMN', 'CI', 'ZION', 'COO', 'TIF', 'RHT', 'FDX', 'LLL', 'GLW', 'GPN', 'IPGP', 'GPC', 'HPQ', 'ADI', 'AMG', 'MTB', 'YUM', 'SYK', 'KMX', 'AME', 'AAP', 'DAL', 'A', 'MON', 'BRK', 'BMY', 'KMB', 'JPM', 'CCI', 'AET', 'DLTR', 'MGM', 'FL', 'HD', 'CLX', 'OKE', 'UPS', 'WMB', 'IFF', 'CMS', 'ARNC', 'VIAB', 'MMC', 'REG', 'ES', 'ITW', 'NDAQ', 'AIZ', 'VRTX', 'CTL', 'QCOM', 'MSI', 'NKTR', 'AMAT', 'BWA', 'ESRX', 'TXT', 'EXR', 'VNO', 'BBT', 'WDC', 'UAL', 'PVH', 'NOC', 'PCAR', 'NSC', 'UAA', 'FFIV', 'PHM', 'LUV', 'HUM', 'SPG', 'SJM', 'ABT', 'CMG', 'ALK', 'ULTA', 'TMK', 'TAP', 'SCG', 'CAT', 'TMO', 'AES', 'MRK', 'RMD', 'MKC', 'WU', 'ACN', 'HIG', 'TEL', 'DE', 'ATVI', 'O', 'UNM', 'VMC', 'ETFC', 'CMA', 'NRG', 'RHI', 'RE', 'FMC', 'MU', 'CB', 'LNT', 'GE', 'CBS', 'ALGN', 'SNA', 'LLY', 'LEN', 'MAA', 'OMC', 'F', 'APA', 'CDNS', 'SLG', 'HP', 'XLNX', 'SHW', 'AFL', 'STT', 'PAYX', 'AIG', 'FOX', 'MA']
df = pd.read_csv("C:/Users/Froilan/Desktop/myFiles/JupyterFiles/stock_correlation_prediction/stock08_price.csv")
universe = list(df.columns.values[1:])
universe.remove("SP500")
unselected_universe = list(set(universe)-set(portfolio))

random.shuffle(unselected_universe)
random.seed(1)
new_assets = unselected_universe[:10].copy()
print(new_assets)


def rolling_corr(item1, item2):
    # import data
    stock_price_df = pd.read_csv(
        "C:/Users/Froilan/Desktop/myFiles/JupyterFiles/stock_correlation_prediction/stock08_price.csv")
    pd.to_datetime(stock_price_df['Date'], format='%Y-%m-%d')
    stock_price_df = stock_price_df.set_index(pd.DatetimeIndex(stock_price_df['Date']))

    # calculate
    df_pair = pd.concat([stock_price_df[item1], stock_price_df[item2]], axis=1)
    df_pair.columns = [item1, item2]
    df_corr = df_pair[item1].rolling(window=100).corr(df_pair[item2])
    return df_corr


data_matrix = []

for i in range(len(new_assets)):
    for j in range(len(new_assets)-1-i):
        a = new_assets[i]
        b = new_assets[9-j]
        corr_series = rolling_corr(a,b)[99:]
        corr_strided = list(corr_series[[100*k for k in range(24)]])
        data_matrix.append(corr_strided)

data_dictionary = {}
for i in range(len(data_matrix)):
    data_dictionary[str(i)] = data_matrix[i]
data_df = pd.DataFrame(data_dictionary)

before_arima_dataset = []
for i in range(45):
    before_arima_dataset.append(data_df[str(i)][:21])
    before_arima_dataset.append(data_df[str(i)][1:22])
    before_arima_dataset.append(data_df[str(i)][2:23])
    before_arima_dataset.append(data_df[str(i)][3:])
before_arima_dataset = pd.DataFrame(np.array(before_arima_dataset))
before_arima_dataset.to_csv('C:/Users/Froilan/Desktop/myFiles/JupyterFiles/stock_correlation_prediction/train_dev_test/new_asset_before_arima.csv')