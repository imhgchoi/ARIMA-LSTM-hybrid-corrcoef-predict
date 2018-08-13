# Correlation Prediction with ARIMA-LSTM Hybrid Model
We applied an ARIMA-LSTM hybrid model to predict future price correlation coefficients of two assets

My draft paper is uploaded on https://arxiv.org/abs/1808.01560.
I'm open for any comments on my work. Please email me at imhgchoi@korea.ac.kr .
I'd really appreciate feedbacks :)

## Paper Abstract
Predicting the price correlation of two assets for future time periods is important in portfolio optimization. We apply LSTM recurrent neural networks (RNN) in predicting the stock price correlation coefficient of two individual stocks. RNNs are competent in understanding temporal dependencies. The use of LSTM cells further enhances its long term predictive properties. To encompass both linearity and nonlinearity in the model, we adopt the ARIMA model as well. The ARIMA model filters linear tendencies in the data and passes on the residual value to the LSTM model. The ARIMA LSTM hybrid model is tested against other traditional predictive financial models such as the full historical model, constant correlation model, single index model and the multi group model. In our empirical study, the predictive ability of the ARIMA-LSTM model turned out superior to all other financial models by a significant scale. Our work implies that it is worth considering the ARIMA LSTM model to forecast correlation coefficient for portfolio optimization.


## Github code reading guideline

My source codes and files can be segmented into three parts.

### PART 1. DATASET FILES

(1) SP500_list.csv : The list of S&P500 firms and their tickers. I scraped the data from wikipedia

(2) SP500_index.csv : The market value corresponding to the time span used in my research

(3) stock08_price.csv : The final price dataset after preprocessing

(4) train_dev_test folder : All the train / development / test1 / test2 datasets needed for each step

(5) stock_data folder : Individual stock price data downloaded


### PART 2. ARIMA MODEL SECTION

(6) Data_Scraping_Codes.ipynb : Web scraping code to get data in need

(7) DATA_PREPROCESSING.ipynb : Data preprocessing codes such as NA imputation, reshaping etc..

(8) ARIMA_MODELING.ipynb : The ARIMA codes to compute the residual values

(9) NEW_ASSET_ARIMA_MODELING.ipynb : After generating model, we test on different assets iteratively. This is the ARIMA section of it


### PART 3. LSTM-CELL RNN MODEL SECTION

(10) raw_python_codes folder : The python codes used to model the LSTM RNN 

(11) models folder : The LSTM models saved for each epoch + The evaluated metric values for each epoch

(12) MODEL_EVALUATOR.ipynb : The model performance is tested against other financial models

(13) MISCELLANEOUS.ipynb : Literally... MISCELLANEOUS!


## MAIN API's/Modules Used in project

(1) BeautifulSoup : For web scraping S&P500 firms

(2) Quandl : To download financial data

(3) pyramid-arima : ARIMA modeling section

(4) Keras : LSTM modeling section

## How to Use the Model

(1) Download two assets' most recent 2000-day-long price data that you wish the correlation coefficient to be predicted.

(2) With a 100-day-window, calculate their correlation coefficient, with a 100 day stride. That will generate 20 time steps.

(3) With the 20 time step, fit the most appropriate ARIMA model and calculate its residuals. Ideally, you could use the 'auto-arima' function in 'pyramid-arima' module. (NOTE: You should also forecast the 21st time step for later use!)

(4) Now you have your 20 time step data for the LSTM model ready. Download the 'epoch28' from the 'models/hybrid_LSTM' folder.

(5) Pass the residual data through the epoch28 model and get your result.

(6) Add the output result to the 21st ARIMA forecast value to get your final result! (NOTE: Rarely, the value might be out of bound; that is, smaller than -1 or bigger than 1. If that's the case, apply MIN(prediction, 1) or MAX(prediction, -1) )
