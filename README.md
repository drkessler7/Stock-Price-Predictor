## Stock Price Predictor
### Overview
Imagine being able to predict future stock prices of a company. What if you could also easily see basic information about that company, as well as plots of its past stock prices? That’s exactly what this predictor does, all underneath a user-friendly Graphical User Interface (GUI). Users can easily request information that is pulled automatically from online sources and displayed in simple tables and graphs. They can also request price predictions that are made using machine learning models. 
### Problem
For many years, the movement of stock prices has been studied by financial institutions and individuals wishing to capitalize on price patterns. When investing in companies listed on New York Stock Exchange (NYSE), NASDAQ, or other exchanges, the basic strategy is to buy stock shares when the price is low, and sell them later when the price is higher. In so doing, one can gain a profit by receiving more money than they initially paid. The motivation for this project is to help users make better decisions when trading stocks.
### Process
The report in this repository discusses all of the steps that were taken in creating this predictor. It also describes the data sources, and what what data looked like. Here I will describe some of the main points that are addressed in the report.
#### Data Source
As with any data science project, one of the earliest steps is acquiring data that is pertinent to the goal. The data used by this predictor comes from online sources, which ensures that users can get the most up-to-date information. Data used in this project comes from Yahoo! Finance, which is one of many online sources that are open to the public for free. 
#### Data Description
By default, the predictor uses three years of daily historical stock price data to make its predictions. This means that there is one row of stock data for each trading day. Each row contains:
* The date in format mm/dd/yyyy
* Its price at the 9:30am EST open of the trading day 
* Its highest price for that trading day
* Its lowest price for that trading day
* Its price at the 4:00pm EST close of the trading day
* Its adjusted price at the close of the trading day (described further in report)
* Its volume (number of shares traded that day)
#### ML Algorithms
I tested three different machine learning algorithms to see which produced the best results. These algorithms were Support Vector Machine (SVM), Random Forest (RF), and K-Nearest Neighbors (KNN). Stock price prediction is a regression problem, meaning that I am trying to predict a numerical value. These three algorithms are simple, but suitable for regression applications.
#### Algorithm Testing
I tested the accuracy of these three algorithms by feeding each of them the exact same data, and comparing their results. Each algorithm took in data for the same stocks over the same period of time. The three trained models produced results that were compared to the actual adjusted closing price on the dates for which the predictions were made. By doing this, the SVM was shown to be the best. 
#### Prediction Improvement
To improve the accuracy of the predictions, the predictor automatically tunes some of the SVM’s parameters. This is like slightly turning the knobs on the radio to hear the station a little bit clearer. 
#### Results
The goal of this project was to have predictions that were more accurate than those made by a Simple Moving Average (SMA). Mathematical explanations of the SMA and the reasons why this was used for comparison can be found in the report within this repository. The tuned SVM model produced slightly more accurate results than the SMA.

#### GUI Construction
The Graphical User Interface (GUI) was designed with the help of QT Designer (https://www.qt.io/). This allowed me to lay out elements such as buttons, text input fields and drop-down menus in a visual way. Once everything was laid out, QT Designer translated the design into Python code. 

I then took this code, and added to it everything that was needed for the GUI to work. Code was needed to dictate what happens when each button is pressed, when text is entered by the user, or drop-down dates are chosen. Beyond that, code was needed to download necessary data, make price predictions based off of it, and display the results to the user.  

#### Predictor Improvements
I have plans to add more features to this predictor. It can be improved to store portfolio information entered by the user. For instance, if the user buys shares of a stock at a particular time, they may want to store that information. Based upon the predictor’s results, the user may want to sell that stock before an predicted loss is incurred, or after a gain is made.

People investing in stocks should not buy stocks in only one company, but multiple companies. This is to protect against the possibility of any one company dropping sharply in value, losing lots of money for the user. Risk/reward features can be built into the predictor to help the user diversify funds safely by investing more in safer companies, and less in riskier ones.

#### More Information
The ‘Capstone Report.pdf’ document in this repository explains the functionality of this predictor in greater detail. It does this by showing an example of price predictions for four stocks. It also compares the results of the three different machine learning models. Greater detail of conclusions and suggested improvements is given as well. 

### Code Availability
The Python code for this predictor can be found in this file as ‘SPP_academic.py’, and is free for anyone to observe, download, modify, or share as they wish. 
## About Files/Folders in this Repository
* SPP_Images: The images within the Stock Price Predictor (SPP) folder are screenshots of various windows and charts created by this script. These will better help the viewer understand the functionality of this predictor.
* SPP_academic.py: This script contains all of the Python code needed to run the predictor. 
* DK_icon.png: The small icon used by the GUI.
* Capstone Report.pdf: This report contains an in-depth description of this predictor’s development.
 
## Requirements 
This project requires **Python 2.7** and the following Python libraries installed:
* [pyqt4](http://pyqt.sourceforge.net/Docs/PyQt4/)
* [yahoo_finance](https://pypi.python.org/pypi/yahoo-finance)
* [numpy](http://www.numpy.org/)
* [pandas](http://pandas.pydata.org)
* [pandas_datareader](https://pandas-datareader.readthedocs.io/en/latest/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [matplotlib](http://matplotlib.org/)
* [webbrowser](https://docs.python.org/2/library/webbrowser.html)
     
