import streamlit as st

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install(xlsxwriter)
import xlsxwriter
install(scipy)
import scipy


import numpy as np 
import math
import pandas as pd 
from pandas import DataFrame, read_csv
#import statsmodels.api as sm

from scipy.stats import norm
import matplotlib.pyplot as plt
import source as ss

from lifelines import KaplanMeierFitter
#from sklearn.datasets import load_iris
import pandas as pd
import scipy.stats as scs
import seaborn as sns
import statistics as stat 

batchNo = st.text_input('Type Batch # or FileName')
#SideID1 = st.text_input('Side ID 1')
#SideID2 = st.text_input('Side ID 2')
numOfTest = st.text_input('How many samples per test?')
if numOfTest is not None:
	try:
		numOfTest = int(numOfTest)
	except:
		st.write('Need the number of sample per test (integer, please)')

#######################################################
Target_weight = st.text_input('Target Weight')
try:
	Target_weight = int(Target_weight)
except:
	st.write("Target weight must be given") 

Limit_weight_low = st.text_input('low limit of weight')
try:
	Limit_weight_low = int(Limit_weight_low)
except:
	st.write("LSL of weight must be given")

Limit_weight_high = st.text_input('high limit of weight')
try:
	Limit_weight_high = int(Limit_weight_high)
except:
	st.write("USL of weight must be given")

#######################################################
Target_hardness = st.text_input('Target hardness')
try:
	Target_hardness = float(Target_hardness)
except:
	st.write("Target hardness must be given") 

Limit_hardness_low = st.text_input('low limit of hardness')
try:
	Limit_hardness_low = float(Limit_hardness_low)
except:
	st.write("LSL of hardness must be given")

Limit_hardness_high = st.text_input('high limit of hardness')
try:
	Limit_hardness_high = float(Limit_hardness_high)
except:
	st.write("USL of hardness must be given")

#######################################################
Target_thickness = st.text_input('Target thickness')
try:
	Target_thickness = float(Target_thickness)
except:
	st.write("Target thickness must be given")

Limit_thickness_low = st.text_input('low limit of thickness')
try:
	Limit_thickness_low = float(Limit_thickness_low)
except:
	st.write("LSL of thickness must be given")

Limit_thickness_high = st.text_input('high limit of thickness')
try:
	Limit_thickness_high = float(Limit_thickness_high)
except:
	st.write("USL of thickness must be given")


SideID = [1,2]

if not isinstance(Target_weight,str):
	Limitsprint ={'Weight':
					{'Target': str(Target_weight),
					 'USL': str(Limit_weight_high),
					 'LSL': str(Limit_weight_low)},
					'Hardness':
					{'Target': str(Target_hardness),
					 'USL': str(Limit_hardness_high),
					 'LSL': str(Limit_hardness_low)},
					'Thickness':
					{'Target': str(Target_thickness),
					 'USL': str(Limit_thickness_high),
					 'LSL':str(Limit_thickness_low)}
				  }

	LimitsTarget ={'Weight':[Target_weight,Limit_weight_high,Limit_weight_low],
               	   'Hardness':[Target_hardness,Limit_hardness_high,Limit_hardness_low],
                   'Thickness':[Target_thickness,Limit_thickness_high,Limit_thickness_low]}
	
	df = pd.DataFrame(Limitsprint)
	st.write(df)

else:
	st.write("Please, type all the necessary parameters to proceed the program")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
	st.write('File has been successfully uploaded')
	[data1,data2,time] = ss.Exporter(batchNo,SideID,numOfTest,uploaded_file)
	ss.normal_dist(batchNo,SideID,numOfTest,uploaded_file)
	ss.ECDFgen(batchNo,SideID,numOfTest,uploaded_file)
	ss.IChart(batchNo,SideID,numOfTest,LimitsTarget,uploaded_file)
	ss.AChart(batchNo,SideID,numOfTest,LimitsTarget,uploaded_file)
	ss.IMChart(batchNo,SideID,numOfTest,uploaded_file)