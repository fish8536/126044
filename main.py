# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

def read_xls(file):
	xls = pd.read_excel(file, index_col=None, header=None)
	return xls

def predata(xls):
	columns = xls[:-1][:].astype('float64')
	return columns

def predata_fft(xls):
	fcolumns1 = np.fft.fft(xls[0][0:7499])
	fcolumns2 = np.fft.fft(xls[1][0:7499])
	fcolumns3 = np.fft.fft(xls[2][0:7499])
	fcolumns4 = np.fft.fft(xls[3][0:7499])
	yf=abs(fft(fcolumns1))
	yf1=abs(fft(fcolumns1))/7500
	yf2 = yf1[range(int(7500/2))]


# tran_folder="traning_source/"
# for f in listdir(tran_folder):
# 	if isfile(join(tran_folder, f)):
# 		read_xls(tran_folder+f)




xls = pd.read_excel("traning_source/20160419001_2016419_114348.xls", index_col=None, header=None)
result = predata(xls)
# for i in result:
print(len(result))
