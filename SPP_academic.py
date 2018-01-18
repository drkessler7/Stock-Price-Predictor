#!/usr/bin/env python2.7

'''
This stock price predictor uses simple machine learning models to predict future adjusted closing prices.
This script produces a GUI with which the user may interact.
The predictor also displays historical stock price data, and other fundamental company data.
The predictor pulls necessary data automatically from Yahoo! Finance through its API.
This project is purely academic, and not intended for commercial purposes.
'''

from PyQt4 import QtCore, QtGui
from PyQt4.Qt import *
from yahoo_finance import Share
import datetime
import os
import pandas as pd
import numpy as np
import pandas_datareader.data as web 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import webbrowser

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, make_scorer     
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)
        
        
class PricePredictor:  
    '''Predicts future closing prices of a stock using Machine Learning'''
    def __init__(self, start_date, end_date, last_train_date, future_dates, future_num_days, pred_ticker):
        self.start_date = start_date
        self.end_date = end_date
        self.last_train_date = last_train_date
        self.future_dates = future_dates
        self.future_num_days = future_num_days
        self.pred_ticker = pred_ticker 
        self.results_df = pd.DataFrame({'Ticker' : [], 'Pred_ret' : [], 'Predicted_Date' : [], 'Days_Later': [], 'Act_ret' : [],
            'Act_Adj_Close' : [], 'Pred_Adj_Close' : [], 'Pct_Err' : [], 'Bench_PAC' : [], 'Bench_Pct_Err' : []})
        
    def make_dir(self):
        '''Creates folder in same directory on user's computer to hold data automatically downloaded from Yahoo'''
        self.DSP_downloads_folder = os.getcwd()+"\Daily_SP_Downloads"
        if not os.path.exists(self.DSP_downloads_folder):
            os.makedirs(self.DSP_downloads_folder)
        
    def get_yahoo_df(self):
        '''Obtains data from Yahoo and builds data frame to be used by Predictor'''
        self.yahoo_df = web.DataReader(self.pred_ticker, 'yahoo', self.start_date, self.end_date)
        self.yahoo_df.to_csv(self.DSP_downloads_folder +'\\'+ self.pred_ticker + '.csv')
        self.yahoo_df = self.yahoo_df.reset_index()
        return self.yahoo_df
       
    def calc_dttf(self):
        '''Calculates the number of trading days from last training date to requested future date'''
        dttf = 0    #dttf = (number of) days train to future
        if self.future_num_days[self.pred_loop_index] == 0: #If the user has specified number of days, use that number instead
            for index, row in self.yahoo_df.iterrows():
                temp_date_str = str(self.yahoo_df['Date'][index]).split(' ')[0]
                if temp_date_str > self.last_train_date.strftime('%Y-%m-%d') and temp_date_str <= self.future_date.strftime('%Y-%m-%d'): 
                    dttf+=1 
        else:
            dttf = self.future_num_days[self.pred_loop_index]
        return dttf
        
    def calc_ntd(self):
        '''Calculates the number of trading days from beginning of dataset to the last training date'''
        ntd = 0 #ntd = num training days
        for index, row in self.yahoo_df.iterrows():
            temp_date_str = str(self.yahoo_df['Date'][index]).split(' ')[0]
            if temp_date_str > self.start_date.strftime('%Y-%m-%d') and temp_date_str <= self.last_train_date.strftime('%Y-%m-%d'): 
                ntd+=1 
        return ntd
        
    def build_df(self): 
        '''Reads in the historical stock price dataframe and constructs features'''
        self.full_df = self.yahoo_df.copy()
        #self.window = self.dttf #MAY need to investigate cases where anomalies in data leads to problems with window and dttf
        self.full_df['{} Return'.format(self.dttf)] = self.full_df['Adj Close'].shift(-self.dttf)/self.full_df['Adj Close']      
        self.full_df['Roll {} ADV'.format(self.window)] = self.full_df['Volume'].rolling(window=self.window).mean()
        self.full_df['V Div {} ADV'.format(self.window)] = self.full_df['Volume']/self.full_df['Roll {} ADV'.format(self.window)]
        self.full_df['SMA {}'.format(self.window)] = self.full_df['Adj Close'].rolling(window=self.window).mean()
        self.full_df['Bench_PAC'] = self.full_df['Adj Close'].shift(1).rolling(window=self.dttf).mean()
        self.full_df['Bench_Pct_Err'] = ((self.full_df['Bench_PAC']/self.full_df['Adj Close']) - 1) * 100
        return self.full_df  
		
    def get_features_labels(self):  
        '''Reads in full dataframe and extracts features and labels'''
        self.X_all = self.full_df[['Adj Close', 'Roll {} ADV'.format(self.window), 'V Div {} ADV'.format(self.window), 
                        'SMA {}'.format(self.window)]][self.window-1:-1]      
        self.X_all = self.X_all.dropna(axis = 'columns', how = 'all')
        self.X_all = self.X_all.reset_index()
        self.X_all = self.X_all[list(self.X_all.columns[1:])]
        self.y_all = self.full_df['{} Return'.format(self.dttf)][self.window-1:-1]
        self.y_all = self.y_all.reset_index()
        self.y_all = self.y_all[list(self.y_all.columns[1:])]
        return self.X_all, self.y_all   
	
    def split_train_test(self):  
        '''Splits data points into training and testing points'''
        self.X_train = self.X_all[:(self.num_training_days)]
        self.y_train = self.y_all[:(self.num_training_days)]
        self.X_test = self.X_all[(self.num_training_days):(self.num_training_days + self.dttf)] 
        self.y_test = self.y_all[(self.num_training_days):(self.num_training_days + self.dttf)] 
        return self.X_train, self.X_test, self.y_train, self.y_test     
        
    def train_classifier(self):
        '''Trains classifier on training data'''
        #self.clf = KNeighborsRegressor()
        self.clf = SVR(cache_size = 600)
        #self.clf = RandomForestRegressor(random_state = 5)  
        
        #vvvv----THIS SECTION PURELY FOR PARAMETER TUNING----vvvv
        #These parameters are for a random forest regressor
        '''self.parameters = {'n_estimators' : [2, 3, 4, 5, 16],
                  'min_samples_split' : [2, 3, 4, 5]}
                  #'oob_score' : [True, False]}
                  #'max_depth' : [None, 1, 2]}
                  #'max_features' : ['auto', 'log2', 'sqrt']}
                  #'n_jobs' : [-1]}'''
                      
        #These parameters are for a SVR
        self.parameters = {'C' : np.arange(0.01, 0.08, 0.005),
                      #'kernel' : ['rbf'],#, 'sigmoid'],
                      #'degree' : [2, 3, 4],
                      #'epsilon': [0.01, 0.05, 0.1, 0.2],
                      'gamma' : [0.00001, 0.000025, 0.00005, 0.000075, 0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.01, 0.1]}
                      #'n_jobs' : [-1]}
					  					  
        '''#These parameters are for a KNeighborsRegressor
        self.parameters = {'n_neighbors' : [3, 5, 7, 9], 
                      'leaf_size' : [10, 20, 30, 40],#, 'sigmoid'],
                      'weights' : ['uniform', 'distance']},
                      #'epsilon': [0.01, 0.1, 0.2],
                      #'gamma' : [0.0001, 0.001, 0.01]}#, 0.05, 0.1, 0.2]}
                      #'n_jobs' : [-1]}'''
        #^^^^----THIS SECTION PURELY FOR PARAMETER TUNING----^^^^
        
        self.scorer = make_scorer(r2_score) #making scorer
        self.grid_obj = GridSearchCV(self.clf, self.parameters, scoring = None, n_jobs=-1)     #tuning parameters using GridSearchCV
        self.grid_fit = self.grid_obj.fit(self.X_train, self.y_train.values.ravel())    #fitting grid search object to training data
        self.clf = self.grid_fit.best_estimator_  #using the estimator with the best set of parameters
        return self.clf

    def test_classifier(self):  
        '''Classifier produces predictions for the required date'''
        if self.dttf != 0:
            self.future_line = self.X_test.iloc[self.dttf - 1]
            self.future_line = self.future_line.values.reshape(1, -1)
            self.pred = self.clf.predict(self.future_line)
        else:
            self.pred = [[0]]
        return self.pred

    def score_classifier(self):
        '''Calculates accuracy of prediction by comparing the predicted price to the actual price '''
        self.score_df = pd.DataFrame(columns=['Pred_ret'], data=self.pred)
        self.score_df['Pred_ret'] = round(self.score_df['Pred_ret'], 4)
        self.score_df['Ticker'] = self.pred_ticker
        self.score_df['Days_Later'] = self.dttf
        self.fut_date_index = self.num_training_days + self.dttf - 1# + self.window - 2 #changed from 69SPI
        self.score_df['Predicted_Date'] = datetime.datetime.strptime(str(self.full_df['Date'][self.fut_date_index]),'%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        self.last_train_index = self.num_training_days - 1
        self.score_df['Act_ret'] = round(self.full_df.at[self.last_train_index, '{} Return'.format(self.dttf)], 4) 
        self.score_df['Act_Adj_Close'] = round(self.full_df.at[self.fut_date_index, 'Adj Close'], 2)
        self.score_df['Pred_Adj_Close'] = round(self.full_df.at[self.last_train_index, 'Adj Close'] * self.score_df['Pred_ret'], 2)
        self.score_df['Pct_Err'] = abs(round(((self.score_df['Pred_Adj_Close']/self.score_df['Act_Adj_Close']) - 1) * 100, 2))
        self.score_df['Bench_PAC'] = abs(round(self.full_df.at[self.fut_date_index, 'Bench_PAC'], 2))
        self.score_df['Bench_Pct_Err'] = abs(round(self.full_df.at[self.fut_date_index, 'Bench_Pct_Err'], 2))
        self.score_df = self.score_df[['Ticker', 'Predicted_Date', 'Days_Later', 'Pred_ret', 
                                'Act_ret', 'Act_Adj_Close', 'Pred_Adj_Close', 'Pct_Err', 'Bench_PAC', 'Bench_Pct_Err']]
        return self.score_df

    def make_predictions(self):
        '''Makes predictions for each date entered in GUI'''
        self.make_dir()
        
        for self.pred_loop_index in range(4): #---------------return this back to range(4) when done tuning-----------------
            self.yahoo_df = self.get_yahoo_df()
            self.num_training_days = self.calc_ntd()
            self.future_date = self.future_dates[self.pred_loop_index]
            self.dttf = self.calc_dttf() 
            self.window = self.dttf  
            self.full_df = self.build_df()
            self.X_all, self.y_all = self.get_features_labels() 
            self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test() 
            self.X_train = self.X_train.fillna(method = 'backfill')
            self.Y_train = self.y_train.fillna(method = 'backfill')
            self.clf = self.train_classifier()
            self.pred = self.test_classifier()
            self.score_df = self.score_classifier()
            self.results_df = self.results_df.append(self.score_df)
        
        self.results_df = self.results_df[['Ticker', 'Predicted_Date', 'Days_Later', 'Pred_ret', 'Act_ret', 
                                        'Act_Adj_Close', 'Pred_Adj_Close', 'Pct_Err', 'Bench_PAC', 'Bench_Pct_Err']]
        self.results_df = self.results_df.reset_index(drop = True)
        return self.results_df
        
        
class FundamentalWidget(QtGui.QWidget):
    '''Creates popup widget to display fundamental data obtained from Yahoo'''
    def __init__(self, fundamental_df, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self._fundamental_df = fundamental_df
        self.initUI()
        
    def initUI(self):
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8("DK_icon.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        QWidget.setWindowIcon(self, icon)
        self.setGeometry(350, 250, 750, 200)    
        self.setWindowTitle('Fundamental Stock Data')
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        table = QtGui.QTableWidget(self)
        table.setRowCount(5)
        table.setColumnCount(9)
        col_headers = ['Ticker', 'Price', 'Year High', 'Year Low', 'Market Cap.',
                    'Avg. Daily Volume', '50 Day Moving Avg.', '200 Day Moving Avg.', 'P/E Ratio']

        for i in range(len(self._fundamental_df.index)):
            for j in range(len(self._fundamental_df.columns)):
                table.setItem(i,j,QtGui.QTableWidgetItem(str(self._fundamental_df.iat[i, j])))
                
        table.setHorizontalHeaderLabels(col_headers)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        grid.addWidget(table, 0, 0)
        
        self.show()        

        
class PredictionWidget(QtGui.QWidget):
    '''Creates popup widget to display table of results produced by the price Predictor'''
    def __init__(self, results_df, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self._results_df = results_df
        self.initUI()
        
    def initUI(self):
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8("DK_icon.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        QWidget.setWindowIcon(self, icon)
        self.setGeometry(250, 250, 903, 200)    
        self.setWindowTitle('Predicted Stock Prices')
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        table = QtGui.QTableWidget(self)
        table.setRowCount(20)
        table.setColumnCount(10)
        col_headers = ['Ticker', 'Predicted Date', 'Days Later', 'Predicted Return', 'Actual Return',
                    'Actual Adj. Close', 'Predicted Adj. Close', 'Percent Error', 'Bench PAC', 'Bench Pct Err']

        for i in range(len(self._results_df.index)):
            for j in range(len(self._results_df.columns)):
                table.setItem(i,j,QtGui.QTableWidgetItem(str(self._results_df.iat[i, j])))
                
        table.setHorizontalHeaderLabels(col_headers)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        grid.addWidget(table, 0, 0)
        self.show() 
        
        
class PlotWidget(QtGui.QWidget):
    '''Creates popup widget to display plots of stock data obtained from Yahoo'''
    def __init__(self, yahoo_df, plot_ticker, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self._yahoo_df = yahoo_df
        self._plot_ticker = plot_ticker
        self.initUI()
        
    def initUI(self):
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8("DK_icon.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        QWidget.setWindowIcon(self, icon)
        plt.style.use('ggplot') 
        fig, ax = plt.subplots()
        ax.plot(pd.to_datetime(self._yahoo_df.index), self._yahoo_df['Adj Close'], 'g-', label='Adjusted Close Price', zorder=1)
        ax.set_ylabel('Close Price', fontsize = 14)
        plt.setp(ax.get_xticklabels(), rotation=15, fontsize=13)
        plt.setp(ax.get_yticklabels(), rotation=15, fontsize=13)
        ax2 = ax.twinx()
        ax2.plot(pd.to_datetime(self._yahoo_df.index), self._yahoo_df['Volume'], 'b', label='Volume', alpha = 0.2, zorder=2)
        ax2.fill_between(self._yahoo_df.index, self._yahoo_df['Volume'], 0, alpha = 0.2)
        ax2.set_ylabel('Volume', fontsize = 14)
        plt.setp(ax2.get_yticklabels(), rotation=15, fontsize=13)
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], loc='upper left')
        ax2.format_xdata = mdates.DateFormatter('%m-%d-%Y')
        ax.set_title(self._plot_ticker + ' Historical Stock Data')
        plt.show()
        
        
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        '''Creates basic geometry for GUI'''
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.setMinimumSize(QtCore.QSize(490, 400))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8("DK_icon.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setTabShape(QtGui.QTabWidget.Rounded)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.btnPredictPrices = QtGui.QPushButton(self.centralwidget)
        self.btnPredictPrices.setGeometry(QtCore.QRect(300, 330, 161, 23))
        self.btnPredictPrices.setObjectName(_fromUtf8("btnPredictPrices"))
        self.btnPredictPrices.clicked.connect(self.show_predictions)
        self.btnPlotSymbol = QtGui.QPushButton(self.centralwidget)
        self.btnPlotSymbol.setGeometry(QtCore.QRect(300, 90, 161, 23))
        self.btnPlotSymbol.setObjectName(_fromUtf8("btnPlotSymbol"))
        self.btnPlotSymbol.clicked.connect(self.display_plots)
        self.leditEnterTickers = QtGui.QLineEdit(self.centralwidget)
        self.leditEnterTickers.setGeometry(QtCore.QRect(30, 330, 241, 21))
        self.leditEnterTickers.setObjectName(_fromUtf8("leditEnterTickers"))
        self.deditStartDate = QtGui.QDateEdit(self.centralwidget)
        self.deditStartDate.setGeometry(QtCore.QRect(30, 30, 110, 22))
        self.deditStartDate.setAlignment(QtCore.Qt.AlignCenter)
        self.deditStartDate.setDate(QtCore.QDate(2013, 1, 1))
        self.deditStartDate.setCalendarPopup(True)
        self.deditStartDate.setObjectName(_fromUtf8("deditStartDate"))
        self.deditEndDate = QtGui.QDateEdit(self.centralwidget)
        self.deditEndDate.setGeometry(QtCore.QRect(290, 30, 110, 22))
        self.deditEndDate.setAlignment(QtCore.Qt.AlignCenter)
        self.deditEndDate.setDate(QtCore.QDate(2017, 1, 1))
        self.deditEndDate.setCalendarPopup(True)
        self.deditEndDate.setObjectName(_fromUtf8("deditEndDate"))
        self.labelStartDate = QtGui.QLabel(self.centralwidget)
        self.labelStartDate.setGeometry(QtCore.QRect(150, 30, 111, 21))
        self.labelStartDate.setObjectName(_fromUtf8("labelStartDate"))
        self.labelEndDate = QtGui.QLabel(self.centralwidget)
        self.labelEndDate.setGeometry(QtCore.QRect(410, 30, 51, 21))
        self.labelEndDate.setObjectName(_fromUtf8("labelEndDate"))
        self.leditPlotSymb = QtGui.QLineEdit(self.centralwidget)
        self.leditPlotSymb.setGeometry(QtCore.QRect(30, 90, 241, 21))
        self.leditPlotSymb.setObjectName(_fromUtf8("leditPlotSymb"))
        self.dedit1stPDate = QtGui.QDateEdit(self.centralwidget)
        self.dedit1stPDate.setGeometry(QtCore.QRect(30, 210, 110, 22))
        self.dedit1stPDate.setAlignment(QtCore.Qt.AlignCenter)
        self.dedit1stPDate.setDate(QtCore.QDate(2016, 12, 6))
        self.dedit1stPDate.setCalendarPopup(True)
        self.dedit1stPDate.setObjectName(_fromUtf8("dedit1stPDate"))
        self.dedit2ndPDate = QtGui.QDateEdit(self.centralwidget)
        self.dedit2ndPDate.setGeometry(QtCore.QRect(30, 240, 110, 22))
        self.dedit2ndPDate.setAlignment(QtCore.Qt.AlignCenter)
        self.dedit2ndPDate.setDate(QtCore.QDate(2016, 12, 7))
        self.dedit2ndPDate.setCalendarPopup(True)
        self.dedit2ndPDate.setObjectName(_fromUtf8("dedit2ndPDate"))
        self.dedit3rdPDate = QtGui.QDateEdit(self.centralwidget)
        self.dedit3rdPDate.setGeometry(QtCore.QRect(30, 270, 110, 22))
        self.dedit3rdPDate.setAlignment(QtCore.Qt.AlignCenter)
        self.dedit3rdPDate.setDate(QtCore.QDate(2016, 12, 8))
        self.dedit3rdPDate.setCalendarPopup(True)
        self.dedit3rdPDate.setObjectName(_fromUtf8("dedit3rdPDate"))
        self.label1stPDate = QtGui.QLabel(self.centralwidget)
        self.label1stPDate.setGeometry(QtCore.QRect(150, 210, 131, 21))
        self.label1stPDate.setObjectName(_fromUtf8("label1stPDate"))
        self.label3rdPDate = QtGui.QLabel(self.centralwidget)
        self.label3rdPDate.setGeometry(QtCore.QRect(150, 270, 121, 21))
        self.label3rdPDate.setObjectName(_fromUtf8("label3rdPDate"))
        self.label2ndPDate = QtGui.QLabel(self.centralwidget)
        self.label2ndPDate.setGeometry(QtCore.QRect(150, 240, 131, 21))
        self.label2ndPDate.setObjectName(_fromUtf8("label2ndPDate"))
        self.btnFundData = QtGui.QPushButton(self.centralwidget)
        self.btnFundData.setGeometry(QtCore.QRect(300, 120, 161, 23))
        self.btnFundData.setObjectName(_fromUtf8("btnFundData"))
        self.btnFundData.clicked.connect(self.display_fund_data)
        self.deditLastTDate = QtGui.QDateEdit(self.centralwidget)
        self.deditLastTDate.setGeometry(QtCore.QRect(30, 180, 110, 22))
        self.deditLastTDate.setAlignment(QtCore.Qt.AlignCenter)
        self.deditLastTDate.setDate(QtCore.QDate(2016, 12, 5))
        self.deditLastTDate.setCalendarPopup(True)
        self.deditLastTDate.setObjectName(_fromUtf8("deditLastTDate"))
        self.labelStartDate_6 = QtGui.QLabel(self.centralwidget)
        self.labelStartDate_6.setGeometry(QtCore.QRect(150, 180, 91, 21))
        self.labelStartDate_6.setObjectName(_fromUtf8("labelStartDate_6"))
        self.sbox1stPDate = QtGui.QSpinBox(self.centralwidget)
        self.sbox1stPDate.setGeometry(QtCore.QRect(290, 210, 42, 22))
        self.sbox1stPDate.setAlignment(QtCore.Qt.AlignCenter)
        self.sbox1stPDate.setMinimum(0)
        self.sbox1stPDate.setProperty("value", 0)
        self.sbox1stPDate.setObjectName(_fromUtf8("sbox1stPDate"))
        self.dedit4thPDate = QtGui.QDateEdit(self.centralwidget)
        self.dedit4thPDate.setGeometry(QtCore.QRect(30, 300, 110, 22))
        self.dedit4thPDate.setAlignment(QtCore.Qt.AlignCenter)
        self.dedit4thPDate.setDate(QtCore.QDate(2016, 12, 9))
        self.dedit4thPDate.setCalendarPopup(True)
        self.dedit4thPDate.setObjectName(_fromUtf8("dedit4thPDate"))
        self.label4thPDate = QtGui.QLabel(self.centralwidget)
        self.label4thPDate.setGeometry(QtCore.QRect(150, 300, 131, 21))
        self.label4thPDate.setFrameShape(QtGui.QFrame.NoFrame)
        self.label4thPDate.setObjectName(_fromUtf8("label4thPDate"))
        self.label1stPDate_2 = QtGui.QLabel(self.centralwidget)
        self.label1stPDate_2.setGeometry(QtCore.QRect(340, 210, 121, 21))
        self.label1stPDate_2.setObjectName(_fromUtf8("label1stPDate_2"))
        self.label2ndPDate_2 = QtGui.QLabel(self.centralwidget)
        self.label2ndPDate_2.setGeometry(QtCore.QRect(340, 240, 121, 21))
        self.label2ndPDate_2.setObjectName(_fromUtf8("label2ndPDate_2"))
        self.sbox2ndPDate = QtGui.QSpinBox(self.centralwidget)
        self.sbox2ndPDate.setGeometry(QtCore.QRect(290, 240, 42, 22))
        self.sbox2ndPDate.setAlignment(QtCore.Qt.AlignCenter)
        self.sbox2ndPDate.setMinimum(0)
        self.sbox2ndPDate.setProperty("value", 0)
        self.sbox2ndPDate.setObjectName(_fromUtf8("sbox2ndPDate"))
        self.label3rdPDate_2 = QtGui.QLabel(self.centralwidget)
        self.label3rdPDate_2.setGeometry(QtCore.QRect(340, 270, 121, 21))
        self.label3rdPDate_2.setObjectName(_fromUtf8("label3rdPDate_2"))
        self.sbox3rdPDate = QtGui.QSpinBox(self.centralwidget)
        self.sbox3rdPDate.setGeometry(QtCore.QRect(290, 270, 42, 22))
        self.sbox3rdPDate.setAlignment(QtCore.Qt.AlignCenter)
        self.sbox3rdPDate.setMinimum(0)
        self.sbox3rdPDate.setProperty("value", 0)
        self.sbox3rdPDate.setObjectName(_fromUtf8("sbox3rdPDate"))
        self.label4thPDate_2 = QtGui.QLabel(self.centralwidget)
        self.label4thPDate_2.setGeometry(QtCore.QRect(340, 300, 121, 21))
        self.label4thPDate_2.setObjectName(_fromUtf8("label4thPDate_2"))
        self.sbox4thPDate = QtGui.QSpinBox(self.centralwidget)
        self.sbox4thPDate.setGeometry(QtCore.QRect(290, 300, 42, 22))
        self.sbox4thPDate.setAlignment(QtCore.Qt.AlignCenter)
        self.sbox4thPDate.setMinimum(0)
        self.sbox4thPDate.setProperty("value", 0)
        self.sbox4thPDate.setObjectName(_fromUtf8("sbox4thPDate"))
        self.btnLookupSymbol = QtGui.QPushButton(self.centralwidget)
        self.btnLookupSymbol.setGeometry(QtCore.QRect(30, 120, 161, 23))
        self.btnLookupSymbol.setObjectName(_fromUtf8("btnLookupSymbol"))
        self.btnLookupSymbol.clicked.connect(self.lookup_symbol)
        self.line2ndHorizontal = QtGui.QFrame(self.centralwidget)
        self.line2ndHorizontal.setGeometry(QtCore.QRect(30, 140, 431, 20))
        self.line2ndHorizontal.setFrameShape(QtGui.QFrame.HLine)
        self.line2ndHorizontal.setFrameShadow(QtGui.QFrame.Sunken)
        self.line2ndHorizontal.setObjectName(_fromUtf8("line2ndHorizontal"))
        self.labelPricePredictionDates = QtGui.QLabel(self.centralwidget)
        self.labelPricePredictionDates.setGeometry(QtCore.QRect(30, 160, 151, 20))
        self.labelPricePredictionDates.setObjectName(_fromUtf8("labelPricePredictionDates"))
        self.line1stHorizontal = QtGui.QFrame(self.centralwidget)
        self.line1stHorizontal.setGeometry(QtCore.QRect(30, 50, 431, 20))
        self.line1stHorizontal.setFrameShape(QtGui.QFrame.HLine)
        self.line1stHorizontal.setFrameShadow(QtGui.QFrame.Sunken)
        self.line1stHorizontal.setObjectName(_fromUtf8("line1stHorizontal"))
        self.labelHistoricalData = QtGui.QLabel(self.centralwidget)
        self.labelHistoricalData.setGeometry(QtCore.QRect(30, 70, 151, 20))
        self.labelHistoricalData.setObjectName(_fromUtf8("labelHistoricalData"))
        self.labelDateRange = QtGui.QLabel(self.centralwidget)
        self.labelDateRange.setGeometry(QtCore.QRect(30, 10, 261, 20))
        self.labelDateRange.setObjectName(_fromUtf8("labelDateRange"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 490, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionQuit = QtGui.QAction(MainWindow)
        self.actionQuit.triggered.connect(QtGui.qApp.quit)
        self.actionQuit.setObjectName(_fromUtf8("actionQuit"))
        self.actionTo_Find_Stock_Symbol = QtGui.QAction(MainWindow)
        self.actionTo_Find_Stock_Symbol.setObjectName(_fromUtf8("actionTo_Find_Stock_Symbol"))
        self.actionPortfolio_Folder = QtGui.QAction(MainWindow)
        self.actionPortfolio_Folder.setObjectName(_fromUtf8("actionPortfolio_Folder"))
        self.menuFile.addAction(self.actionQuit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        '''Adds extra features to GUI's buttons and fields'''
        MainWindow.setWindowTitle(_translate("MainWindow", "Stock Price Predictor", None))
        self.btnPredictPrices.setToolTip(_translate("MainWindow", "<html><head/><body><p>Displays the predicted closing prices for the stock symbols and days entered. Also compares the predicted price to the actual price if possible.</p></body></html>", None))
        self.btnPredictPrices.setText(_translate("MainWindow", "Predict Future Prices", None))
        self.btnPlotSymbol.setToolTip(_translate("MainWindow", "Plots the daily historical stock price and volume information for a given stock symbol.", None))
        self.btnPlotSymbol.setText(_translate("MainWindow", "Plot Historical Data", None))
        self.leditEnterTickers.setToolTip(_translate("MainWindow", "(MAX 5) Enter stock ticker symbols for price prediction, separated by commas.", None))
        self.leditEnterTickers.setPlaceholderText(_translate("MainWindow", "Enter Stock Ticker Symbols for Price Prediction", None))
        self.deditStartDate.setToolTip(_translate("MainWindow", "The beginning of the range of dates to be used for data download and/or stock price prediction.", None))
        self.deditEndDate.setToolTip(_translate("MainWindow", "The end of the range of dates to be used for data download and/or stock price prediction.", None))
        self.labelStartDate.setToolTip(_translate("MainWindow", "The beginning of the range of dates to be used for data download and/or stock price prediction.", None))
        self.labelStartDate.setText(_translate("MainWindow", "Start Date            TO", None))
        self.labelEndDate.setToolTip(_translate("MainWindow", "The end of the range of dates to be used for data download and/or stock price prediction.", None))
        self.labelEndDate.setText(_translate("MainWindow", " End Date", None))
        self.leditPlotSymb.setToolTip(_translate("MainWindow", "(MAX 5) Enter stock ticker symbols to plot, separated by commas.", None))
        self.leditPlotSymb.setPlaceholderText(_translate("MainWindow", "Enter Stock Ticker Symbols to Plot", None))
        self.dedit1stPDate.setToolTip(_translate("MainWindow", "First day for which the closing price will be predicted. Must come after the Last Training Date.", None))
        self.dedit2ndPDate.setToolTip(_translate("MainWindow", "(Optional) Second day for which the closing price will be predicted. Must come after the Last Training Date.", None))
        self.dedit3rdPDate.setToolTip(_translate("MainWindow", "(Optional) Third day for which the closing price will be predicted. Must come after the Last Training Date.", None))
        self.label1stPDate.setToolTip(_translate("MainWindow", "First day for which the closing price will be predicted. Must come after the Last Training Date.", None))
        self.label1stPDate.setText(_translate("MainWindow", "1st Predicted Date      OR", None))
        self.label3rdPDate.setToolTip(_translate("MainWindow", "(Optional) Third day for which the closing price will be predicted. Must come after the Last Training Date.", None))
        self.label3rdPDate.setText(_translate("MainWindow", "3rd Predicted Date      OR", None))
        self.label2ndPDate.setToolTip(_translate("MainWindow", "(Optional) Second day for which the closing price will be predicted. Must come after the Last Training Date.", None))
        self.label2ndPDate.setText(_translate("MainWindow", "2nd Predicted Date     OR", None))
        self.btnFundData.setToolTip(_translate("MainWindow", "Plots the fundamental data for a given stock symbol.", None))
        self.btnFundData.setText(_translate("MainWindow", "Show Fundamental Data", None))
        self.labelStartDate_6.setToolTip(_translate("MainWindow", "Last day used for training the price prediction model.", None))
        self.labelStartDate_6.setText(_translate("MainWindow", "Last Training Date", None))
        self.sbox1stPDate.setToolTip(_translate("MainWindow", "First day for which the closing price will be predicted. Must come after the Last Training Date.", None))
        self.dedit4thPDate.setToolTip(_translate("MainWindow", "(Optional) Fourth day for which the closing price will be predicted. Must come after the Last Training Date.", None))
        self.label4thPDate.setToolTip(_translate("MainWindow", "(Optional) Fourth day for which the closing price will be predicted. Must come after the Last Training Date.", None))
        self.label4thPDate.setText(_translate("MainWindow", "4th Predicted Date      OR", None))
        self.label1stPDate_2.setToolTip(_translate("MainWindow", "First day for which the closing price will be predicted. Must come after the Last Training Date.", None))
        self.label1stPDate_2.setText(_translate("MainWindow", "Trading Days into Future", None))
        self.label2ndPDate_2.setToolTip(_translate("MainWindow", "(Optional) Second day for which the closing price will be predicted. Must come after the Last Training Date.", None))
        self.label2ndPDate_2.setText(_translate("MainWindow", "Trading Days into Future", None))
        self.sbox2ndPDate.setToolTip(_translate("MainWindow", "(Optional) Second day for which the closing price will be predicted. Must come after the Last Training Date.", None))
        self.label3rdPDate_2.setToolTip(_translate("MainWindow", "(Optional) Third day for which the closing price will be predicted. Must come after the Last Training Date.", None))
        self.label3rdPDate_2.setText(_translate("MainWindow", "Trading Days into Future", None))
        self.sbox3rdPDate.setToolTip(_translate("MainWindow", "(Optional) Third day for which the closing price will be predicted. Must come after the Last Training Date.", None))
        self.label4thPDate_2.setToolTip(_translate("MainWindow", "(Optional) Fourth day for which the closing price will be predicted. Must come after the Last Training Date.", None))
        self.label4thPDate_2.setText(_translate("MainWindow", "Trading Days into Future", None))
        self.sbox4thPDate.setToolTip(_translate("MainWindow", "(Optional) Fourth day for which the closing price will be predicted. Must come after the Last Training Date.", None))
        self.btnLookupSymbol.setToolTip(_translate("MainWindow", "Opens web page to help find a stock\'s ticker symbol.", None))
        self.btnLookupSymbol.setText(_translate("MainWindow", "Lookup Symbol", None))
        self.labelPricePredictionDates.setText(_translate("MainWindow", "Stock Price Prediction", None))
        self.labelHistoricalData.setText(_translate("MainWindow", "Historical Stock Data", None))
        self.labelDateRange.setText(_translate("MainWindow", "Date Range for Data Download and Price Prediction", None))
        self.menuFile.setTitle(_translate("MainWindow", "File", None))
        self.actionQuit.setText(_translate("MainWindow", "Quit", None))
        self.actionTo_Find_Stock_Symbol.setText(_translate("MainWindow", "Look Up Stock Symbols", None))
        self.actionPortfolio_Folder.setText(_translate("MainWindow", "Portfolio Folder", None))
        self.actionPortfolio_Folder.setToolTip(_translate("MainWindow", "Location for saved portfolios", None))
        
    def lookup_symbol(self):
        '''Opens web page in browser to help user research stock ticker symbols'''
        webbrowser.open('http://finance.yahoo.com/')
        
    def get_fund_data(self, fund_ticker):
        '''Obtains and displays basic stock information from Yahoo! Finance for each of the tickers'''
        self.yahoo_request = Share(self.fund_ticker)
        self.ADV = self.yahoo_request.get_avg_daily_volume()
        self.market_cap = self.yahoo_request.get_market_cap()
        self.mov_avg50 = self.yahoo_request.get_50day_moving_avg()
        self.mov_avg200 = self.yahoo_request.get_200day_moving_avg()
        self.pe_ratio = self.yahoo_request.get_price_earnings_ratio()
        self.price = self.yahoo_request.get_price()
        self.year_high = self.yahoo_request.get_year_high()
        self.year_low = self.yahoo_request.get_year_low()     
        self.data = {'Ticker': self.fund_ticker, 'Price' : self.price, 'Year High' : self.year_high, 'Year Low' : self.year_low,
                'Market Cap.' : self.market_cap, 'Avg. Daily Volume' : self.ADV,  
                '50 Day Moving Avg.': self.mov_avg50, '200 Day Moving Avg.': self.mov_avg200, 'P/E Ratio' : self.pe_ratio,
                }  
        self.temp_df = pd.DataFrame(data = self.data, index=[0])
        self.temp_df = self.temp_df[['Ticker', 'Price', 'Year High', 'Year Low', 'Market Cap.',
                            'Avg. Daily Volume', '50 Day Moving Avg.', '200 Day Moving Avg.', 'P/E Ratio']]
        return self.temp_df
        
        
    def display_fund_data(self):
        '''Reads ticker symbols entered into GUI's plotting line edit, obtains fundamental data from Yahoo, displays data in FundamentalWidget''' 
        fund_ticker_text = str(self.leditPlotSymb.text())
        fund_tickers = fund_ticker_text.split(',')
        self.fundamental_df = pd.DataFrame()
        
        for self.fund_ticker in fund_tickers:
            self.fund_ticker = self.fund_ticker.strip().upper()
            self.temp_df = self.get_fund_data(self.fund_ticker) 
            self.fundamental_df = self.fundamental_df.append(self.temp_df)
        
        self.fund_window = FundamentalWidget(self.fundamental_df)
        self.fund_window.show()
   
        
    def show_predictions(self):
        '''Reads ticker symbols and dates entered into GUI's fields, makes Predictor object, displays results in PredictionWidget'''
        self.start_date = self.deditStartDate.date().toPyDate() 
        self.end_date = self.deditEndDate.date().toPyDate()
        self.last_train_date = self.deditLastTDate.date().toPyDate()
        
        future_date1 = self.dedit1stPDate.date().toPyDate()
        future_date2 = self.dedit2ndPDate.date().toPyDate()
        future_date3 = self.dedit3rdPDate.date().toPyDate()
        future_date4 = self.dedit4thPDate.date().toPyDate()
        self.future_dates = [future_date1, future_date2, future_date3, future_date4]
        
        future_num_day1 = self.sbox1stPDate.value()
        future_num_day2 = self.sbox2ndPDate.value()
        future_num_day3 = self.sbox3rdPDate.value()
        future_num_day4 = self.sbox4thPDate.value()
        self.future_num_days = [future_num_day1, future_num_day2, future_num_day3, future_num_day4]
        
        pred_ticker_text = str(self.leditEnterTickers.text())
        self.pred_tickers = pred_ticker_text.split(',')
        self.results_df = pd.DataFrame()
        
        for self.pred_ticker in self.pred_tickers:
            self.pred_ticker = self.pred_ticker.strip().upper()
            self.predictor = PricePredictor(self.start_date, self.end_date, self.last_train_date, 
                                        self.future_dates, self.future_num_days, self.pred_ticker)
            self.temp_df = self.predictor.make_predictions()
            self.results_df = self.results_df.append(self.temp_df)
        
        self.results_window = PredictionWidget(self.results_df)
        self.results_window.show()
        
    def display_plots(self):
        '''Reads ticks symbols from GUI's plotting line edit, retrieves data from Yahoo, plots data in PlotWidget'''
        plot_ticker_text = str(self.leditPlotSymb.text())
        plot_tickers = plot_ticker_text.split(',')
        self.start_date = self.deditStartDate.date().toPyDate() 
        self.end_date = self.deditEndDate.date().toPyDate()
        
        for self.plot_ticker in plot_tickers:
            self.plot_ticker = self.plot_ticker.strip().upper()
            self.yahoo_df = web.DataReader(self.plot_ticker, 'yahoo', self.start_date, self.end_date)
            self.plot_window = PlotWidget(self.yahoo_df, self.plot_ticker)
            
      

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

