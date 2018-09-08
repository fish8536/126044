# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from scipy.fftpack import fft,ifft

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
result = np.zeros((40, 4, 10))
tran_folder="traning_source/"
tfi = 0
for f in listdir(tran_folder):
	if isfile(join(tran_folder, f)):
		xls = read_xls(tran_folder+f)
		pxls = predata(xls)
		fcolumns = np.zeros((4,10))
		for fc in range(4):
			fcolumns[fc] = col_max_average(col_split(predata_fft(pxls[fc])))
		result[tfi] = fcolumns
		tfi=tfi+1

print(result)

