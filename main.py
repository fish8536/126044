# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from os import listdir
from os.path import isfile, join
from scipy.fftpack import fft,ifft
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

def read_xls(file):
	xls = pd.read_excel(file, index_col=None, header=None)
	return xls

def predata(xls):
	columns = xls[:-1][:].astype('float64')
	return columns

def predata_fft(col):
	fcolumns = np.fft.fft(col)
	yf=abs(fft(fcolumns))
	yf1=abs(fft(fcolumns))/7500
	yf2 = yf1[range(int(7500/2))]
	return yf2

def col_split(col):
	columns_split = np.array_split(col, 5, axis=0)
	return columns_split

def col_max_average(col):
	columns = []
	for i in range(0,5):
		columns.append(np.amax(col[i]))
		columns.append(np.average(col[i]))
	return columns
# 40 file, 4 sensor, 5 fft max and avg
tsresult = np.zeros((40, 40))
tsfileans = []
tran_folder="traning_source/"
tfi = 0
for f in listdir(tran_folder):
	if isfile(join(tran_folder, f)):
		xls = read_xls(tran_folder+f)
		pxls = predata(xls)
		fcolumns = []
		for fc in range(4):
			fcolumns.extend(col_max_average(col_split(predata_fft(pxls[fc]))))
		tsresult[tfi] = fcolumns
		tsfileans.append(float(xls[0][7500].split(':')[1]))
		tfi=tfi+1

X_train, X_test, y_train, y_test = train_test_split(tsresult, tsfileans, test_size = 0.1, random_state = 0)
regressor = LinearRegression()
regressor.fit(X_train,y_train)
#取得截距。如果公式是y=a+bx，a即是截距
print(regressor.intercept_)

#使用測試組資料來預測結果
y_pred = regressor.predict(X_test)
print(y_pred)

#比較實際及預測的關係
plt.scatter(y_test,y_pred)
plt.show()

#看實際值及預測值之間的殘差分佈圖

# sns.distplot((y_test-y_pred))

#載入迴歸常見的評估指標

#Root Mean Squared Error (RMSE)代表MSE的平方根。比起MSE更為常用，因為更容易解釋y。
RMSE = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
print(RMSE)