# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 10-3-2026



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
data = pd.read_csv(r"/content/global_sports_footwear_sales_2018_2026.csv", parse_dates=['order_date'])
print("Available columns:", data.columns.tolist())

# Select one company (Example: Nike)
data = data[data['brand'] == 'Nike'].copy() # Use 'brand' instead of 'Ticker'

# Set order_date as index
data.set_index('order_date', inplace=True)

# Select revenue_usd as the time series data
X = data['revenue_usd']

# Set figure size
plt.rcParams['figure.figsize'] = [12,6]

# Plot Original Data
plt.plot(X)
plt.title('ORIGINAL SALES REVENUE DATA (Nike)')
plt.xlabel('Order Date')
plt.ylabel('Revenue (USD)')
plt.show()

# ACF and PACF of Original Data
plt.subplot(2,1,1)
plot_acf(X, lags=40, ax=plt.gca())
plt.title('AUTOCORRELATION FUNCTION (ACF) OF ORIGINAL DATA')

plt.subplot(2,1,2)
plot_pacf(X, lags=40, ax=plt.gca())
plt.title('PARTIAL AUTOCORRELATION FUNCTION (PACF) OF ORIGINAL DATA')

plt.tight_layout()
plt.show()

# -----------------------------
# ARMA(1,1)
# -----------------------------

arma11_model = ARIMA(X, order=(1,0,1)).fit()

phi1 = arma11_model.params['ar.L1']
theta1 = arma11_model.params['ma.L1']

ar1 = np.array([1, -phi1])
ma1 = np.array([1, theta1])

N = 1000

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('SIMULATED ARMA(1,1) PROCESS')
plt.xlim([0,500])
plt.xlabel('time')
plt.ylabel('value')
plt.show()

plot_acf(ARMA_1)
plt.title("AUTOCORRELATION FUNCTION (ACF) OF ARMA(1,1)")
plt.show()

plot_pacf(ARMA_1)
plt.title("PARTIAL AUTOCORRELATION FUNCTION (PACF) OF ARMA(1,1)")
plt.show()

# -----------------------------
# ARMA(2,2)
# -----------------------------

arma22_model = ARIMA(X, order=(2,0,2)).fit()

phi1 = arma22_model.params['ar.L1']
phi2 = arma22_model.params['ar.L2']

theta1 = arma22_model.params['ma.L1']
theta2 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1, -phi2])
ma2 = np.array([1, theta1, theta2])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)

plt.plot(ARMA_2)
plt.title('SIMULATED ARMA(2,2) PROCESS')
plt.xlim([0,500])
plt.xlabel('time')
plt.ylabel('value')
plt.show()

plot_acf(ARMA_2)
plt.title("AUTOCORRELATION FUNCTION (ACF) OF ARMA(2,2)")
plt.show()

plot_pacf(ARMA_2)
plt.title("PARTIAL AUTOCORRELATION FUNCTION (PACF) OF ARMA(2,2)")
plt.show()
```
### output:
<img width="1319" height="597" alt="Screenshot 2026-03-10 100638" src="https://github.com/user-attachments/assets/056ea933-da32-4e13-bebe-4159946f70f2" />
<img width="1098" height="559" alt="Screenshot 2026-03-10 100717" src="https://github.com/user-attachments/assets/77a0bee9-3866-4844-b4f8-4992ab80f11f" />
RESULT:
Thus, a python program is created to fir ARMA Model successfully.
