import numpy as np 
import math
import pandas as pd 
from pandas import DataFrame, read_csv
#import statsmodels.api as sm
import xlsxwriter
from scipy.stats import norm
import matplotlib.pyplot as plt
import source as ss
import scipy
from lifelines import KaplanMeierFitter
#from sklearn.datasets import load_iris
import pandas as pd
import scipy.stats as scs
import seaborn as sns
import statistics as stat 
import streamlit as st 

def DataExtract(batchNo,
                SideID,
                NoOfTest,
                uploaded_file):
    
    filename = batchNo
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    attributes = sheet_names[0:3]

    rawDataNoOrder = {}
    rawData = {}

    InitialDataIndex = 2

    for attr in attributes:
        df = pd.read_excel(xls,attr,header=None, index_col=False) #<---- This is necessary to use the first row as the data, not the name of columns
        numOfSampleTest = df.shape[1]-1
        time = list(df.iloc[0,1:])
        side = list(df.iloc[1,1:]) # <---- sideID from each test and will be compared against the target id we have 

        rawData[attr] = {}
        rawDataNoOrder[attr] = []

        for ID in SideID:
            rawData[attr][str(ID)] = []

        for numOfcolumn in np.arange(numOfSampleTest):
            data = df.iloc[InitialDataIndex:InitialDataIndex+NoOfTest,numOfcolumn+1]
            dataList = list(data)
            for i in np.arange(10):
                individualData = dataList[i]
                rawDataNoOrder[attr].append(individualData)
            
            # splitting side 1 and side 2 data - possibly side 3
            for ID in SideID:
                if side[numOfcolumn] == ID:
                    for i in np.arange(10):
                        side_ind_data = dataList[i]
                        rawData[attr][str(ID)].append(side_ind_data)

    return rawDataNoOrder, rawData, time


def Exporter(batchNo,
             SideID,
             numOfTest,
             uploaded_file):

    filename = batchNo

    [data1, data2, time] = DataExtract(batchNo, SideID, numOfTest,uploaded_file)

    workbook = xlsxwriter.Workbook(filename+'_SPC_byPython.xlsx')
    for key in data1.keys():
        # For individual side of SPC 
        for ID in SideID:
            worksheet = workbook.add_worksheet(key+'_side'+str(ID)+'_SPC')
            worksheet.write(0,0,'SampleID')
            
            if key == 'Weight':
                column_name = 'Tablet Weigth (mg)'
            elif key == 'Thickness':
                column_name = 'Tablet Thickness (mm)'
            else:
                column_name = 'Tablet Hardness (kp)'

            worksheet.write(0,1,column_name)
            row = 1
            col = 0
            for n, element in enumerate(data2[key][str(ID)]): # There may be the case we don't have some attribute for 1,2 layers if there are 3 layers
                worksheet.write(row,col,1+n//10)
                worksheet.write(row,col+1, element)
                row += 1
        
        # For both side SPC
        worksheet = workbook.add_worksheet(key+'_both_side_SPC')
        worksheet.write(0,0,'SampleID')
        
        if key == 'Weight':
            column_name = 'Tablet Weigth (mg)'
        elif key == 'Thickness':
            column_name = 'Tablet Thickness (mm)'
        else:
            column_name = 'Tablet Hardness (kp)'

        worksheet.write(0,1,column_name)
        row = 1
        col = 0
        
        for n, element in enumerate(data1[key]):
            worksheet.write(row,col,1+n//20)
            worksheet.write(row,col+1, element)
            row += 1
                
    workbook.close()         
    
    return data1, data2, time

def normal_dist(batchNo,SideID,numOfTest,uploaded_file):
    [data1,data2,time] = Exporter(batchNo,SideID,numOfTest,uploaded_file)
    keys = data1.keys()
    for key in keys:  
        # Generate some data for this 
        # demonstration.
        for ID in SideID:
            data = data2[key][str(ID)]
            
            # Fit a normal distribution to
            # the data:
            # mean and standard deviation
            mu, std = norm.fit(data) 

            plt.figure(figsize=(6,4),dpi=200)
            plt.tick_params(labelsize=8,direction='in')

            # Plot the histogram.
            plt.hist(data, bins=25, density=True, alpha=1, color='b',edgecolor='black')
            
            # Plot the PDF.
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)

            if key == 'Weight':
                ylabelName = 'Tablet Weigth (mg)'
            elif key == 'Thickness':
                ylabelName = 'Tablet Thickness (mm)'
            else:
                ylabelName = 'Tablet Hardness (kp)'

            FigTitle = ylabelName+', Side '+str(ID)+', '+batchNo
            plt.plot(x, p, 'r', linewidth=2)
            plt.title(FigTitle)
            plt.xlabel(ylabelName)
            plt.ylabel('Frequency/PDF')
            plt.tight_layout()
            
            plt.savefig(batchNo+'_ND_'+key+'_side_'+str(ID)+'.png')

        data = data1[key]
        # Fit a normal distribution to
        # the data:
        # mean and standard deviation
        mu, std = norm.fit(data) 
        plt.figure(figsize=(6,4),dpi=200)
        plt.tick_params(labelsize=8,direction='in')
        # Plot the histogram.
        plt.hist(data, bins=25, density=True, alpha=1, color='b',edgecolor='black')
        
        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)

        if key == 'Weight':
            ylabelName = 'Tablet Weigth (mg)'
        elif key == 'Thickness':
            ylabelName = 'Tablet Thickness (mm)'
        else:
            ylabelName = 'Tablet Hardness (kp)'

        FigTitle = ylabelName+', Both Side '+batchNo

        plt.plot(x, p, 'r', linewidth=2)
        plt.title(FigTitle)
        plt.xlabel(ylabelName)
        plt.ylabel('Frequency/PDF')
        plt.tight_layout()
        
        plt.savefig(batchNo+'_ND_'+key+'_both_side.png')


def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)

def ECDFgen(batchNo,SideID,numOfTest,uploaded_file):
    [data1,data2,time] = Exporter(batchNo,SideID,numOfTest,uploaded_file)
    keys = data1.keys()
    for key in keys: 
        for ID in SideID:
            data = data2[key][str(ID)]
            mu, std = norm.fit(data) 
            x_fit = np.sort(data)
            y_fit = scs.norm.cdf(x_fit,mu,std)

            x0,y0 = ecdf(data)

            kmf = KaplanMeierFitter()
            kmf.fit(data)
            df = kmf.survival_function_

            df2 = kmf.confidence_interval_survival_function_
            lower = df2['KM_estimate_lower_0.95']
            upper = df2['KM_estimate_upper_0.95']
            data_x = list(df2.index)

            if key == 'Weight':
                ylabelName = 'Tablet Weigth (mg)'
            elif key == 'Thickness':
                ylabelName = 'Tablet Thickness (mm)'
            else:
                ylabelName = 'Tablet Hardness (kp)'

            FigTitle = ylabelName+', Side '+str(ID)+', '+batchNo

            plt.figure(figsize=(6,4),dpi=200)
            plt.tick_params(direction='in',labelsize=10)
            plt.plot(x0,y0,'b',alpha=1,linewidth=1,label='Empirical CDF')
            plt.plot(x_fit,y_fit,'k',linewidth=1,label='Fitted Line')
            plt.step(data_x[1:],1-lower[1:],'b--',lw=0.75,alpha=1,label='Upper Confidence Bound')
            plt.step(data_x[1:],1-upper[1:],'g--',lw=0.75,alpha=1,label='Lower Confidence Bound')
            plt.xlabel(ylabelName)
            plt.title(FigTitle)
            plt.ylabel('Frequency/Empirical CDF')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                        fancybox=True, shadow=True,ncol=2) 
            plt.tight_layout()
            plt.savefig(batchNo+'_ECDF_'+key+'_side_'+str(ID)+'.png')

        data = data1[key]
        # Fit a normal distribution to
        # the data:
        # mean and standard deviation
        mu, std = norm.fit(data) 
        x_fit = np.sort(data)
        
        y_fit = scs.norm.cdf(x_fit,mu,std)

        x0,y0 = ecdf(data)

        kmf = KaplanMeierFitter()
        kmf.fit(data)
        df = kmf.survival_function_

        df2 = kmf.confidence_interval_survival_function_
        lower = df2['KM_estimate_lower_0.95']
        upper = df2['KM_estimate_upper_0.95']
        data_x = list(df2.index)
        

        if key == 'Weight':
            ylabelName = 'Tablet Weigth (mg)'
        elif key == 'Thickness':
            ylabelName = 'Tablet Thickness (mm)'
        else:
            ylabelName = 'Tablet Hardness (kp)'

        FigTitle = ylabelName+', Both Side, '+batchNo

        plt.figure(figsize=(6,4),dpi=200)
        plt.tick_params(direction='in',labelsize=10)
        plt.plot(x0,y0,'b',alpha=1,linewidth=1,label='Empirical CDF')
        plt.plot(x_fit,y_fit,'k',linewidth=1,label='Fitted Line')
        plt.step(data_x[1:],1-lower[1:],'b--',lw=0.75,alpha=1,label='Upper Confidence Bound')
        plt.step(data_x[1:],1-upper[1:],'g--',lw=0.75,alpha=1,label='Lower Confidence Bound')
        plt.xlabel(ylabelName)
        plt.title(FigTitle)
        plt.ylabel('Frequency/Empirical CDF')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                    fancybox=True, shadow=True,ncol=2) 
        plt.tight_layout()
        plt.savefig(batchNo+'_ECDF_'+key+'_both_side.png')


def IChart(batchNo,SideID,numOfTest,LimitsTarget,uploaded_file):
    [data1,data2,time] = Exporter(batchNo,SideID,numOfTest,uploaded_file)

    keys = data1.keys()
    
    for key in keys: 
        data = data1[key]
        means = stat.mean(data)
        [target,USL,LSL] = LimitsTarget[key] 

        fig, ax = plt.subplots(figsize=(6,4),dpi=200)
        ax.tick_params(direction='in',labelsize=10)

        if key == 'Weight':
            ylabelName = 'Tablet Weigth (mg)'
        elif key == 'Thickness':
            ylabelName = 'Tablet Thickness (mm)'
        else:
            ylabelName = 'Tablet Hardness (kp)'

        FigTitle = ylabelName+', '+batchNo
        # x chart
        ax.plot(data, linestyle='-', lw=0.75, marker='o',markersize=3, color='blue',label='Data')
        ax.axhline(target, color='green',label='Target')
        ax.axhline(USL, color = 'red', linestyle = 'dashed',label='USL/LSL')
        ax.axhline(LSL, color = 'red', linestyle = 'dashed')
        ax.set_title(FigTitle)
        ax.set_xlabel('Observation',fontsize=12)
        ax.set_ylabel(ylabelName,fontsize=12)

        i = 0
        j = 0
        control = True
        for unit in data:
            if unit > USL or unit < LSL:
                print('Unit', i, 'out of cotrol limits!')
                if j == 0:
                    ax.plot(i,data[i],marker = 'o',markersize=10,lw=0,
                    fillstyle='none',color='red',label='Violation')
                else:
                    ax.plot(i,data[i],marker = 'o',markersize=10,lw=0,
                    fillstyle='none',color='red')
                control = False
                n_col = 4
                j += 1
            i += 1
        if control == True:
            print('All points within control limits.')
            n_col = 3

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=True, ncol=n_col,fontsize=10)   
        plt.tight_layout()
        plt.savefig(batchNo+'_IChart_'+key+'.png')


def AChart(batchNo,SideID,numOfTest,LimitsTarget,uploaded_file):
    [data1,data2,time] = Exporter(batchNo,SideID,numOfTest,uploaded_file)

    keys = data1.keys()
    
    for key in keys: 
        data = []
        sem = []
        for i in np.arange(len(data1[key])/numOfTest):
            n = int(i)
            avg = stat.mean(data1[key][n*numOfTest:(n+1)*numOfTest])
            sem.append(scs.sem(data1[key][n*numOfTest:(n+1)*numOfTest]))
            data.append(avg)

        fig, ax = plt.subplots(figsize=(6,4),dpi=200)
        ax.tick_params(direction='in',labelsize=10)

        [target, USL, LSL] = LimitsTarget[key]
        if key == 'Weight':
            ylabelName = 'Tablet Weigth (mg)'
        elif key == 'Thickness':
            ylabelName = 'Tablet Thickness (mm)'
        else:
            ylabelName = 'Tablet Hardness (kp)'

        FigTitle = ylabelName+', '+batchNo
        # x chart
        ax.plot(data, linestyle='-', lw=0.75, marker='s',fillstyle='none',markersize=3, color='blue',label='Data')
        ax.axhline(target, color='green',label='Target')
        ax.axhline(USL, color = 'red', linestyle = 'dashed',label='USL/LSL')
        ax.axhline(LSL, color = 'red', linestyle = 'dashed')
        ax.errorbar(np.arange(len(data)),data,yerr=sem,linewidth=0.5,capsize=2,color='b')
        ax.set_title(FigTitle)
        ax.set_xlabel('Observation',fontsize=12)
        ax.set_ylabel(ylabelName,fontsize=12)

        i = 0
        j = 0
        control = True
        for unit in data:
            if unit > USL or unit < LSL:
                print('Unit', i, 'out of cotrol limits!')
                if j == 0:
                    ax.plot(i,data[i],marker = 'o',markersize=10,lw=0,
                    fillstyle='none',color='red',label='Violation')
                else:
                    ax.plot(i,data[i],marker = 'o',markersize=10,lw=0,
                    fillstyle='none',color='red')
                control = False
                n_col = 4
                j += 1
            i += 1
        if control == True:
            print('All points within control limits.')
            n_col = 3

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=True, ncol=n_col,fontsize=10)   
        plt.tight_layout()
        plt.savefig(batchNo+'_AChart_'+key+'.png')

def IMChart(batchNo,SideID,numOfTest,uploaded_file):
    [data1,data2,time] = Exporter(batchNo,SideID,numOfTest,uploaded_file)

    keys = data1.keys()
    
    for key in keys: 
        for ID in SideID:
            x = data2[key][str(ID)]
            # Create dummy data

            # Define list variable for moving ranges
            MR = [np.nan]

            # Get and append moving ranges
            i = 1
            for data in range(1, len(x)):
                MR.append(abs(x[i] - x[i-1]))
                i += 1

            # Convert list to pandas Series objects    
            MR = pd.Series(MR)
            x = pd.Series(x)

            # Concatenate mR Series with and rename columns
            data = pd.concat([x,MR], axis=1).rename(columns={0:"x", 1:"mR"})


            if key == 'Weight':
                ylabelName = 'Tablet Weigth (mg)'
            elif key == 'Thickness':
                ylabelName = 'Tablet Thickness (mm)'
            else:
                ylabelName = 'Tablet Hardness (kp)'

            FigTitle = ylabelName+', Side '+str(ID)+', '+batchNo

            # Plot x and mR charts

            UCLx = stat.mean(data['x'])+3*stat.stdev(data['x'])
            LCLx = stat.mean(data['x'])-3*stat.stdev(data['x'])
            UCLmr = stat.mean(data['mR'][1:len(data['mR'])])+3*stat.stdev(data['mR'][1:len(data['mR'])])
            LCLmr = stat.mean(data['mR'][1:len(data['mR'])])-3*stat.stdev(data['mR'][1:len(data['mR'])])

            fig, axs = plt.subplots(2, figsize=(6,6), sharex=True,dpi=200)
            axs[0].tick_params(direction='in',labelsize=10)
            axs[1].tick_params(direction='in',labelsize=10)
            # x chart
            axs[0].plot(data['x'], linestyle='-', marker='o', markersize=3, lw= 0.75, color='blue',label='Data')
            axs[0].axhline(stat.mean(data['x']), color='green',label='Center')
            axs[0].axhline(UCLx, color = 'red', linestyle = 'dashed',label='UCL/LCL')
            axs[0].axhline(LCLx, color = 'red', linestyle = 'dashed')
            if min(data['x']) < LCLx:
                ylimbottom = min(data['x']) - stat.stdev(data['x'])
            else:
                ylimbottom = LCLx-stat.stdev(data['x'])
            
            if max(data['x']) > UCLx:
                ylimtop = max(data['x']) + stat.stdev(data['x'])
            else:
                ylimtop = UCLx+stat.stdev(data['x'])
    
            axs[0].set_ylim([ylimbottom,ylimtop])
            axs[0].set_title('Control Chart')
            axs[0].set_ylabel('I')

            # mR chart
            axs[1].plot(data['mR'], linestyle='-', marker='o', markersize=3, lw= 0.75, color='blue',label='Data')
            axs[1].axhline(stat.mean(data['mR'][1:len(data['mR'])]), color='green',label='Center')
            axs[1].axhline(UCLmr, color='red', linestyle ='dashed',label='UCL/LCL')
            axs[1].axhline(0, color='red', linestyle ='dashed')
            axs[1].set_ylim(bottom=-UCLmr/10)
            axs[1].set_title(FigTitle)
            axs[1].set_xlabel('Observation',fontsize=12)
            axs[1].set_ylabel('MR')


            # Validate points out of control limits for x chart
            i = 0
            count=0
            control = True
            n_col1 = 3
            
            for unit in data['x']:
                if unit > UCLx or unit < LCLx :
                    print('Unit', i, 'out of cotrol limits!')
                    count += 1
                    if count == 1:
                        axs[0].plot(i, data['x'][i], linewidth=0, marker='o', markersize=10, 
                        fillstyle='none', color='red', label='Violation')
                        
                    else:
                        axs[0].plot(i, data['x'][i], linewidth=0, marker='o', markersize=10, 
                        fillstyle='none', color='red')
                    control = False
                    n_col1 = 4
                i += 1
            if control == True:
                print('All points within control limits.')
                
            # Validate points out of control limits for mR chart
            i = 0
            count = 0
            control = True
            n_col2 = 3
            
            for unit in data['mR']:
                if unit > UCLmr or unit < LCLmr :
                    print('Unit', i, 'out of control limits!')
                    count += 1
                    if count == 1:
                        axs[1].plot(i, data['mR'][i], linewidth=0, marker='o', markersize=10, 
                        fillstyle='none', color='red', label='Violation')
                    else:
                        axs[1].plot(i, data['mR'][i], linewidth=0, marker='o', markersize=10, 
                        fillstyle='none', color='red')
                    control = False
                    n_col2 = 4
                i += 1
            if control == True:
                print('All points within control limits.')

            axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.055), 
            fancybox=True, shadow=True, ncol=n_col1, fontsize=10)   
            axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.225), 
            fancybox=True, shadow=True, ncol=n_col2, fontsize=10)
            plt.tight_layout()
            plt.savefig(batchNo+'_I-MR_'+key+'_side_'+str(ID)+'.png')

        x = data1[key]
        # Create dummy data
        # Define list variable for moving ranges
        MR = [np.nan]
        # Get and append moving ranges
        i = 1
        for data in range(1, len(x)):
            MR.append(abs(x[i] - x[i-1]))
            i += 1
        # Convert list to pandas Series objects    
        MR = pd.Series(MR)
        x = pd.Series(x)
        # Concatenate mR Series with and rename columns
        data = pd.concat([x,MR], axis=1).rename(columns={0:"x", 1:"mR"})
        if key == 'Weight':
            ylabelName = 'Tablet Weigth (mg)'
        elif key == 'Thickness':
            ylabelName = 'Tablet Thickness (mm)'
        else:
            ylabelName = 'Tablet Hardness (kp)'

        FigTitle = ylabelName+', Both Side, '+batchNo
        # Plot x and mR charts
        fig, axs = plt.subplots(2, figsize=(6,6), sharex=True,dpi=200)
        axs[0].tick_params(direction='in',labelsize=10)
        axs[1].tick_params(direction='in',labelsize=10)
        # x chart
        axs[0].plot(data['x'], linestyle='-', marker='o', markersize=3, lw= 0.75, color='blue',label='Data')
        axs[0].axhline(stat.mean(data['x']), color='green',label='Center')
        axs[0].axhline(UCLx, color = 'red', linestyle = 'dashed',label='UCL/LCL')
        axs[0].axhline(LCLx, color = 'red', linestyle = 'dashed')
        
        if min(data['x']) < LCLx:
            ylimbottom = min(data['x']) - stat.stdev(data['x'])
        else:
            ylimbottom = LCLx-stat.stdev(data['x'])
        
        if max(data['x']) > UCLx:
            ylimtop = max(data['x']) + stat.stdev(data['x'])
        else:
            ylimtop = UCLx+stat.stdev(data['x'])

        axs[0].set_ylim([ylimbottom,ylimtop])
        axs[0].set_title('Control Chart')
        axs[0].set_ylabel('I')
 
        # mR chart
        axs[1].plot(data['mR'], linestyle='-', marker='o', markersize=3, lw= 0.75, color='blue',label='Data')
        axs[1].axhline(stat.mean(data['mR'][1:len(data['mR'])]), color='green',label='Center')
        axs[1].axhline(stat.mean(data['mR'][1:len(data['mR'])])+3*stat.stdev(data['mR'][1:len(data['mR'])]), color='red', linestyle ='dashed',label='UCL/LCL')
        axs[1].axhline(0, color='red', linestyle ='dashed')
        axs[1].set_ylim(bottom=-UCLmr/10)
        axs[1].set_title(FigTitle)
        axs[1].set_xlabel('Observation',fontsize=12)
        axs[1].set_ylabel('MR')

        # Validate points out of control limits for x chart
        i = 0
        count=0
        control = True
        n_col1 = 3
        UCL = stat.mean(data['x'])+3*stat.stdev(data['x'])
        LCL = stat.mean(data['x'])-3*stat.stdev(data['x'])
        for unit in data['x']:
            if unit > UCL or unit < LCL :
                print('Unit', i, 'out of cotrol limits!')
                count += 1
                if count == 1:
                    axs[0].plot(i, data['x'][i], linewidth=0, marker='o', markersize=10, 
                    fillstyle='none', color='red', label='Violation')
                    
                else:
                    axs[0].plot(i, data['x'][i], linewidth=0, marker='o', markersize=10, 
                    fillstyle='none', color='red')
                control = False
                n_col1 = 4
            i += 1
        if control == True:
            print('All points within control limits.')
            
        # Validate points out of control limits for mR chart
        i = 0
        count = 0
        control = True
        n_col2 = 3
        UCL = stat.mean(data['mR'][1:len(data['mR'])])+3*stat.stdev(data['mR'][1:len(data['mR'])])
        LCL = stat.mean(data['mR'][1:len(data['mR'])])-3*stat.stdev(data['mR'][1:len(data['mR'])])
        for unit in data['mR']:
            if unit > UCL or unit < LCL :
                print('Unit', i, 'out of control limits!')
                count += 1
                if count == 1:
                    axs[1].plot(i, data['mR'][i], linewidth=0, marker='o', markersize=10, 
                    fillstyle='none', color='red', label='Violation')
                else:
                    axs[1].plot(i, data['mR'][i], linewidth=0, marker='o', markersize=10, 
                    fillstyle='none', color='red')
                control = False
                n_col2 = 4
            i += 1
        if control == True:
            print('All points within control limits.')
        axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.055), fancybox=True, shadow=True, ncol=n_col1, fontsize=10)   
        axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.225), fancybox=True, shadow=True, ncol=n_col2, fontsize=10)
        plt.tight_layout()
        plt.savefig(batchNo+'_I-MR_'+key+'_both_side.png')


def DSCA(batchNo,SideID,numOfTest,LimitsTarget,uploaded_file):
    [data1,data2,time] = Exporter(batchNo,SideID,numOfTest,uploaded_file)

    keys = data1.keys()
    for key in keys:
        [target, USL, LSL] = LimitsTarget[key]
        if key == 'Weight':
            ylabelName = 'Tablet Weigth (mg)'
        elif key == 'Thickness':
            ylabelName = 'Tablet Thickness (mm)'
        else:
            ylabelName = 'Tablet Hardness (kp)'

        for ID in SideID:
            
            data = data2[key][str(ID)]
            mu = np.mean(data)
            std = np.std(data,ddof=1)

            Cpl = (mu - LSL)/(3*std)
            Cpu = (USL - mu)/(3*std)

            if Cpu > Cpl:
                Cpk = Cpl
            else:
                Cpk = Cpu 

            zscore_low2 = scs.norm.ppf(0.025)
            zscore_high2 = scs.norm.ppf(0.975)

            CI = [mu + zscore_low2*std/np.sqrt(len(data)), mu + zscore_high2*std/np.sqrt(len(data))]

            RSD = std/mu

            # calculating the expected value based on z value 
            '''
            z = (Value of Interest - mean)/(standard deviation)
            '''

            zscore_low = scs.norm.ppf(0.0015)
            zscore_high = scs.norm.ppf(0.9985)

            LowExptValue = zscore_low*std + mu
            HighExptValue = zscore_high*std + mu 

            xmin, xmax = [LSL-(USL-LSL)/2, USL+(USL-LSL)/2]
            x = np.linspace(xmin, xmax, 1000)
            p = norm.pdf(x, mu, std)


            # Descriptive Stats
            DS = np.array([[''],[len(data)],['%.3f'%(mu)],[[float('%.3f'%(CI[0])),float('%.3f'%(CI[1]))]],
                        ['%.3f'%(std)],['%.3f'%(RSD)],[min(data)],[max(data)],['%.3f'%(LowExptValue)],
                        ['%.3f'%(HighExptValue)]])

            # Capability Analysis
            CA = np.array([[''],[LSL],[target],[USL],['%.3f'%((Cpu+Cpl)/2)],['%.3f'%(Cpl)],['%.3f'%(Cpu)],['%.3f'%(Cpk)]])

            # Descriptive Stats and Capability Analysis Combined 
            St = np.vstack([DS,CA])

            rowNames = ('$\\bf{Descriptive Data}$','N','Mean','CI(95%) of Mean','StDev','%RSD','Min','Max','X(0.0015)','X(0.9985)', # Descriptive Data 
                        '$\\bf{Capability Analysis}$','LSL','Target','USL','$C_p$','$C_{pl}$','$C_{pu}$','$C_{pk}$')                # Capability Analysis 

            fig, ax = plt.subplots(1,2,figsize=(8,4),dpi=200,gridspec_kw={'width_ratios': [1, 3]})
            # Stat Table 
            ax[0].axis('tight')
            ax[0].axis('off')
            
            the_table = ax[0].table(cellText=St,rowLabels=rowNames,loc='best',colLoc='center',edges='open',colWidths=[0.75,0.75],cellLoc='left')
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(8)
            # Histogram 
            ax[1].tick_params(labelsize=10, direction='in')
            ax[1].hist(data, bins='auto', alpha=1, color='b',edgecolor='black',label='Data',density=True)
            ax[1].plot(x, p, 'r', linewidth=2,label='Normal Distribution')
            ax[1].axvline(USL,linewidth=0.75,linestyle="--",color='red',label = 'USL/LSL')
            ax[1].axvline(LSL,linewidth=0.75,linestyle="--",color='red')
            ax[1].axvline(LowExptValue,linewidth=0.75,linestyle=":",color='blue',label = 'X(0.0015)/X(0.9985)')
            ax[1].axvline(HighExptValue,linewidth=0.75,linestyle=":",color='blue')
            ax[1].axvline(target,linewidth=2,linestyle="-",color='green',label='Target')
            ax[1].axvline(mu,linewidth=2,linestyle="--",color='k',label='Mean')
            ax[1].set_xlabel(ylabelName)
            ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                    fancybox=True, shadow=True,ncol=2,fontsize=8) 
            ax[1].set_ylabel('Probability/PDF')
            ax[1].set_xlim([LSL-(USL-LSL)/2,USL+(USL-LSL)/2])
            plt.tight_layout()

            plt.savefig(batchNo+'_StatSummary_'+key+'_side_'+str(ID)+'.png')


        data = data1[key]
        mu = np.mean(data)
        std = np.std(data,ddof=1)
        Cpl = (mu - LSL)/(3*std)
        Cpu = (USL - mu)/(3*std)
        if Cpu > Cpl:
            Cpk = Cpl
        else:
            Cpk = Cpu 
        zscore_low2 = scs.norm.ppf(0.025)
        zscore_high2 = scs.norm.ppf(0.975)
        CI = [mu + zscore_low2*std/np.sqrt(len(data)), mu + zscore_high2*std/np.sqrt(len(data))]
        RSD = std/mu
        # calculating the expected value based on z value 
        '''
        z = (Value of Interest - mean)/(standard deviation)
        '''
        zscore_low = scs.norm.ppf(0.0015)
        zscore_high = scs.norm.ppf(0.9985)
        LowExptValue = zscore_low*std + mu
        HighExptValue = zscore_high*std + mu 
        xmin, xmax = [LSL-(USL-LSL)/2, USL+(USL-LSL)/2]
        x = np.linspace(xmin, xmax, 1000)
        p = norm.pdf(x, mu, std)
        # Descriptive Stats
        DS = np.array([[''],[len(data)],['%.3f'%(mu)],[[float('%.3f'%(CI[0])),float('%.3f'%(CI[1]))]],
                    ['%.3f'%(std)],['%.3f'%(RSD)],[min(data)],[max(data)],['%.3f'%(LowExptValue)],
                    ['%.3f'%(HighExptValue)]])
        # Capability Analysis
        CA = np.array([[''],[LSL],[target],[USL],['%.3f'%((Cpu+Cpl)/2)],['%.3f'%(Cpl)],['%.3f'%(Cpu)],['%.3f'%(Cpk)]])
        # Descriptive Stats and Capability Analysis Combined 
        St = np.vstack([DS,CA])
        rowNames = ('$\\bf{Descriptive Data}$','N','Mean','CI(95%) of Mean','StDev','%RSD','Min','Max','X(0.0015)','X(0.9985)', # Descriptive Data 
                    '$\\bf{Capability Analysis}$','LSL','Target','USL','$C_p$','$C_{pl}$','$C_{pu}$','$C_{pk}$')                # Capability Analysis 
        
        fig, ax = plt.subplots(1,2,figsize=(8,4),dpi=200,gridspec_kw={'width_ratios': [1, 3]})
        # Stat Table 
        ax[0].axis('tight')
        ax[0].axis('off')
        ax[0].table(cellText=St,rowLabels=rowNames,loc='best',colLoc='center',edges='open',colWidths=[0.75,0.75],fontsize=8,cellLoc='left')
        # Histogram 
        ax[1].tick_params(labelsize=10, direction='in')
        ax[1].hist(data, bins='auto', alpha=1, color='b',edgecolor='black',label='Data',density=True)
        ax[1].plot(x, p, 'r', linewidth=2,label='Normal Distribution')
        ax[1].axvline(USL,linewidth=0.75,linestyle="--",color='red',label = 'USL/LSL')
        ax[1].axvline(LSL,linewidth=0.75,linestyle="--",color='red')
        ax[1].axvline(LowExptValue,linewidth=0.75,linestyle=":",color='blue',label = 'X(0.0015)/X(0.9985)')
        ax[1].axvline(HighExptValue,linewidth=0.75,linestyle=":",color='blue')
        ax[1].axvline(target,linewidth=2,linestyle="-",color='green',label='Target')
        ax[1].axvline(mu,linewidth=2,linestyle="--",color='k',label='Mean')
        ax[1].set_xlabel(ylabelName)
        ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                fancybox=True, shadow=True,ncol=2,fontsize=8) 
        ax[1].set_ylabel('Probability/PDF')
        ax[1].set_xlim([LSL-(USL-LSL)/2,USL+(USL-LSL)/2])
        plt.tight_layout()
        plt.savefig(batchNo+'_StatSummary_'+key+'_both_side.png')