from tkinter import *
from PIL import ImageTk,Image
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import datetime
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from keras.models import model_from_json
import pickle
plt.style.use('fivethirtyeight')


def datesRange(per):
	datesf =[]
	prevdates = pd.date_range(end = date+'/'+month+'/'+year,periods =60 )
	dates = pd.date_range(start=date+'/'+month+'/'+year, periods=per+1)
	dates = dates[-per:]
	for x in range(60):
		datesf.append(prevdates[x])
	for x in range(20):
		datesf.append(dates[x])
	return datesf

x = datetime.datetime.now()
month = x.strftime("%m")
year = x.strftime("%Y")
date = x.strftime("%d")

app = Tk()
app.title('A T J Predictor')
#app.geometry("500x500")
mframe = Frame(app)
mframe.grid(sticky = N+S+E+W)

menubar = Menu(mframe)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="New")
filemenu.add_command(label="Open")
filemenu.add_command(label="Save")
filemenu.add_separator()
filemenu.add_command(label="Exit",command=quit)
menubar.add_cascade(label="File", menu=filemenu)

helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="Help Index")
helpmenu.add_command(label="About...")
menubar.add_cascade(label="Help", menu=helpmenu)
app.config(menu=menubar)
#width = 250,height = 400,
lframe = Frame(mframe,background= '#6666FF')
lframe.grid(column =  0, row = 0,sticky= N+S)

Rframe = Frame(mframe)
Rframe.grid(column =  1, row = 0,sticky=N+S+E+W)


predictedFrame = Frame(Rframe)
predictedFrame.grid(column =  1, row = 0,sticky=N+S+E+W)

aboutUs = Frame(Rframe)
aboutUs.grid(column =  1, row = 0,sticky=N+S+E+W )

liveStock = Frame(Rframe)
liveStock.grid(column =  1, row = 0,sticky=N+S+E+W)

def fb_pred():
	df = web.DataReader('FB', data_source='yahoo', start='2012-01-01', end=date+'-'+month+'-'+year)
	df['Adj High'] = df['High']
	df['Adj Low'] = df['Low']
	df['Adj Open'] = df['Open']
	df['Adj Volume'] = df['Volume']
	df = df[['Adj Open', 'Adj High', 'Adj Low', 'Adj Close', 'Adj Volume']]

	df['HL_PCT'] = (df['Adj High'] - df['Adj Close'])/ df['Adj Close'] * 100
	df['PCT_change'] = (df['Adj Close'] - df['Adj Open'])/ df['Adj Open'] * 100

	df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Adj Volume']]
	prev = df[-60:]
	prev = np.array(prev['Adj Close'])
	prevs = []
	for x in range(60):
		prevs.append(prev[x])
	df = df[-20:]
	adj_cl = np.array(df['Adj Close'])
	hl_pct = np.array(df['HL_PCT'])
	pct_change = np.array(df['PCT_change'])
	adj_vol =	np.array(df['Adj Volume'])

	filename = 'modelut.sav'
	loaded_model = pickle.load(open(filename, 'rb'))
	dates = datesRange(20)
	valu = []
	graph = []
	for x in range(20):
		y = loaded_model.predict([[adj_cl[x],hl_pct[x],pct_change[x],adj_vol[x]]])
		valu.append(y)
		prevs.append(valu[x][0])
	# prevs.append(graph)                        #60 before + 20 after
	fig = Figure(figsize=(10,4), dpi=100)
	plt.title('FB Model')
	plt.xlabel('Date', fontsize=2)
	plt.ylabel('Close Price USD ($)', fontsize=2)

	plt.plot(dates,prevs)
	plt.rc('xtick', labelsize=8)  
	plt.rc('ytick', labelsize=8)  
	# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
	#plt.show()
	#plt.savefig('foo.png')
	# fig.add_subplot(111).plot(dates,prevs)    // latest
	fig.add_subplot(111).plot(dates[:-20],prevs[:-20],color='red')
	fig.add_subplot(111).plot(dates[-21:],prevs[-21:],color= 'green')
	# print("predicted price \n")
	# print(pred_price)


	# apple_quote2 = web.DataReader('FB', data_source='yahoo', start='2020-01-24', end='2020-01-24')
	# print(apple_quote2['Close'])

	canvas = FigureCanvasTkAgg(fig, master=predictedFrame)  # A tk.DrawingArea.
	canvas.draw()
	canvas.get_tk_widget().grid(column = 0,columnspan=5,row = 1)

	rpredictedfb = Label(predictedFrame,text ="           Next 5 days Price           " + str(round(prevs[-20], 2)) + "           " + str(round(prevs[-19], 2)) + "           "+str(round(prevs[-18], 2))+"           "+str(round(prevs[-17], 2)) + "           "+str(round(prevs[-16], 2)) +"           " )
	rpredictedfb.grid(row = 2,columnspan=5)

	#toolbar = NavigationToolbar2Tk(canvas, Rframe)
	#toolbar.update()
	#canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)


def mic_pred():
	df = web.DataReader('MSFT', data_source='yahoo', start='2012-01-01', end=date+'-'+month+'-'+year)
	df['Adj High'] = df['High']
	df['Adj Low'] = df['Low']
	df['Adj Open'] = df['Open']
	df['Adj Volume'] = df['Volume']
	df = df[['Adj Open', 'Adj High', 'Adj Low', 'Adj Close', 'Adj Volume']]

	df['HL_PCT'] = (df['Adj High'] - df['Adj Close'])/ df['Adj Close'] * 100
	df['PCT_change'] = (df['Adj Close'] - df['Adj Open'])/ df['Adj Open'] * 100

	df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Adj Volume']]
	prev = df[-60:]
	prev = np.array(prev['Adj Close'])
	prevs = []
	for x in range(60):
		prevs.append(prev[x])
	df = df[-20:]
	adj_cl = np.array(df['Adj Close'])
	hl_pct = np.array(df['HL_PCT'])
	pct_change = np.array(df['PCT_change'])
	adj_vol =	np.array(df['Adj Volume'])

	filename = 'microsoft_predict.sav'
	loaded_model = pickle.load(open(filename, 'rb'))
	dates = datesRange(20)
	valu = []
	graph = []
	for x in range(20):
		y = loaded_model.predict([[adj_cl[x],hl_pct[x],pct_change[x],adj_vol[x]]])
		valu.append(y)
		prevs.append(valu[x][0])
	# prevs.append(graph)                        #60 before + 20 after
	fig = Figure(figsize=(10,4), dpi=100)
	plt.title('Microsoft Model')
	plt.xlabel('Date', fontsize=2)
	plt.ylabel('Close Price USD ($)', fontsize=2)

	plt.plot(dates,prevs)
	plt.rc('xtick', labelsize=8)  
	plt.rc('ytick', labelsize=8)  
	# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
	#plt.show()
	#plt.savefig('foo.png')
	# fig.add_subplot(111).plot(dates,prevs)    // latest
	fig.add_subplot(111).plot(dates[:-20],prevs[:-20],color='red')
	fig.add_subplot(111).plot(dates[-21:],prevs[-21:],color= 'green')
	# print("predicted price \n")
	# print(pred_price)


	# apple_quote2 = web.DataReader('FB', data_source='yahoo', start='2020-01-24', end='2020-01-24')
	# print(apple_quote2['Close'])

	canvas = FigureCanvasTkAgg(fig, master=predictedFrame)  # A tk.DrawingArea.
	canvas.draw()
	canvas.get_tk_widget().grid(column = 0,columnspan=5,row = 1)

	rpredictedmic = Label(predictedFrame,text ="           Next 5 days Price           " + str(round(prevs[-20], 2)) + "           " + str(round(prevs[-19], 2)) + "           "+str(round(prevs[-18], 2))+"           "+str(round(prevs[-17], 2)) + "           "+str(round(prevs[-16], 2)) +"           " )
	rpredictedmic.grid(row = 2,columnspan=5)

	#toolbar = NavigationToolbar2Tk(canvas, Rframe)
	#toolbar.update()
	#canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)


def amazon_pred():
	df = web.DataReader('AMZN', data_source='yahoo', start='2012-01-01', end=date+'-'+month+'-'+year)
	df['Adj High'] = df['High']
	df['Adj Low'] = df['Low']
	df['Adj Open'] = df['Open']
	df['Adj Volume'] = df['Volume']
	df = df[['Adj Open', 'Adj High', 'Adj Low', 'Adj Close', 'Adj Volume']]

	df['HL_PCT'] = (df['Adj High'] - df['Adj Close'])/ df['Adj Close'] * 100
	df['PCT_change'] = (df['Adj Close'] - df['Adj Open'])/ df['Adj Open'] * 100

	df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Adj Volume']]
	prev = df[-60:]
	prev = np.array(prev['Adj Close'])
	prevs = []
	for x in range(60):
		prevs.append(prev[x])
	df = df[-20:]
	adj_cl = np.array(df['Adj Close'])
	hl_pct = np.array(df['HL_PCT'])
	pct_change = np.array(df['PCT_change'])
	adj_vol =	np.array(df['Adj Volume'])

	filename = 'amazon_predict.sav'
	loaded_model = pickle.load(open(filename, 'rb'))
	dates = datesRange(20)
	valu = []
	graph = []
	for x in range(20):
		y = loaded_model.predict([[adj_cl[x],hl_pct[x],pct_change[x],adj_vol[x]]])
		valu.append(y)
		prevs.append(valu[x][0])
	# prevs.append(graph)                        #60 before + 20 after
	fig = Figure(figsize=(10,4), dpi=100)
	plt.title('Amazon Model')
	plt.xlabel('Date', fontsize=2)
	plt.ylabel('Close Price USD ($)', fontsize=2)

	plt.plot(dates,prevs)
	plt.rc('xtick', labelsize=8)  
	plt.rc('ytick', labelsize=8)  
	# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
	#plt.show()
	#plt.savefig('foo.png')
	# fig.add_subplot(111).plot(dates,prevs)    // latest
	fig.add_subplot(111).plot(dates[:-20],prevs[:-20],color='red')
	fig.add_subplot(111).plot(dates[-21:],prevs[-21:],color= 'green')
	# print("predicted price \n")
	# print(pred_price)


	# apple_quote2 = web.DataReader('FB', data_source='yahoo', start='2020-01-24', end='2020-01-24')
	# print(apple_quote2['Close'])

	canvas = FigureCanvasTkAgg(fig, master=predictedFrame)  # A tk.DrawingArea.
	canvas.draw()
	canvas.get_tk_widget().grid(column = 0,columnspan=5,row = 1)

	rpredictedmic = Label(predictedFrame,text ="           Next 5 days Price           " + str(round(prevs[-20], 2)) + "           " + str(round(prevs[-19], 2)) + "           "+str(round(prevs[-18], 2))+"           "+str(round(prevs[-17], 2)) + "           "+str(round(prevs[-16], 2)) +"           " )
	rpredictedmic.grid(row = 2,columnspan=5)

	#toolbar = NavigationToolbar2Tk(canvas, Rframe)
	#toolbar.update()
	#canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)



def google_pred():
	df = web.DataReader('GOOGL', data_source='yahoo', start='2012-01-01', end=date+'-'+month+'-'+year)
	df['Adj High'] = df['High']
	df['Adj Low'] = df['Low']
	df['Adj Open'] = df['Open']
	df['Adj Volume'] = df['Volume']
	df = df[['Adj Open', 'Adj High', 'Adj Low', 'Adj Close', 'Adj Volume']]

	df['HL_PCT'] = (df['Adj High'] - df['Adj Close'])/ df['Adj Close'] * 100
	df['PCT_change'] = (df['Adj Close'] - df['Adj Open'])/ df['Adj Open'] * 100

	df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Adj Volume']]
	prev = df[-60:]
	prev = np.array(prev['Adj Close'])
	prevs = []
	for x in range(60):
		prevs.append(prev[x])
	df = df[-20:]
	adj_cl = np.array(df['Adj Close'])
	hl_pct = np.array(df['HL_PCT'])
	pct_change = np.array(df['PCT_change'])
	adj_vol =	np.array(df['Adj Volume'])

	filename = 'google_predict.sav'
	loaded_model = pickle.load(open(filename, 'rb'))
	dates = datesRange(20)
	valu = []
	graph = []
	for x in range(20):
		y = loaded_model.predict([[adj_cl[x],hl_pct[x],pct_change[x],adj_vol[x]]])
		valu.append(y)
		prevs.append(valu[x][0])
	# prevs.append(graph)                        #60 before + 20 after
	fig = Figure(figsize=(10,4), dpi=100)
	plt.title('Google Model')
	plt.xlabel('Date', fontsize=2)
	plt.ylabel('Close Price USD ($)', fontsize=2)

	plt.plot(dates,prevs)
	plt.rc('xtick', labelsize=8)  
	plt.rc('ytick', labelsize=8)  
	# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
	#plt.show()
	#plt.savefig('foo.png')
	# fig.add_subplot(111).plot(dates,prevs)    // latest
	fig.add_subplot(111).plot(dates[:-20],prevs[:-20],color='red')
	fig.add_subplot(111).plot(dates[-21:],prevs[-21:],color= 'green')
	# print("predicted price \n")
	# print(pred_price)


	# apple_quote2 = web.DataReader('FB', data_source='yahoo', start='2020-01-24', end='2020-01-24')
	# print(apple_quote2['Close'])

	canvas = FigureCanvasTkAgg(fig, master=predictedFrame)  # A tk.DrawingArea.
	canvas.draw()
	canvas.get_tk_widget().grid(column = 0,columnspan=5,row = 1)

	rpredictedmic = Label(predictedFrame,text ="           Next 5 days Price           " + str(round(prevs[-20], 2)) + "           " + str(round(prevs[-19], 2)) + "           "+str(round(prevs[-18], 2))+"           "+str(round(prevs[-17], 2)) + "           "+str(round(prevs[-16], 2)) +"           " )
	rpredictedmic.grid(row = 2,columnspan=5)

	#toolbar = NavigationToolbar2Tk(canvas, Rframe)
	#toolbar.update()
	#canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)



def app_pred():
	df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end=date+'-'+month+'-'+year)
	df['Adj High'] = df['High']
	df['Adj Low'] = df['Low']
	df['Adj Open'] = df['Open']
	df['Adj Volume'] = df['Volume']
	df = df[['Adj Open', 'Adj High', 'Adj Low', 'Adj Close', 'Adj Volume']]

	df['HL_PCT'] = (df['Adj High'] - df['Adj Close'])/ df['Adj Close'] * 100
	df['PCT_change'] = (df['Adj Close'] - df['Adj Open'])/ df['Adj Open'] * 100

	df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Adj Volume']]
	prev = df[-60:]
	prev = np.array(prev['Adj Close'])
	prevs = []
	for x in range(60):
		prevs.append(prev[x])
	df = df[-20:]
	adj_cl = np.array(df['Adj Close'])
	hl_pct = np.array(df['HL_PCT'])
	pct_change = np.array(df['PCT_change'])
	adj_vol =	np.array(df['Adj Volume'])

	filename = 'apple_predict.sav'
	loaded_model = pickle.load(open(filename, 'rb'))
	dates = datesRange(20)
	valu = []
	graph = []
	for x in range(20):
		y = loaded_model.predict([[adj_cl[x],hl_pct[x],pct_change[x],adj_vol[x]]])
		valu.append(y)
		prevs.append(valu[x][0])
	# prevs.append(graph)                        #60 before + 20 after
	fig = Figure(figsize=(10,4), dpi=100)
	plt.title('Apple Model')
	plt.xlabel('Date', fontsize=2)
	plt.ylabel('Close Price USD ($)', fontsize=2)

	plt.plot(dates[:-20],prevs[:-20],color='red')
	plt.plot(dates[-20:],prevs[-20:],color= 'green')
	plt.rc('xtick', labelsize=8)  
	plt.rc('ytick', labelsize=8)  
	# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
	#plt.show()
	#plt.savefig('foo.png')
	fig.add_subplot(111).plot(dates[:-20],prevs[:-20],color='red')
	fig.add_subplot(111).plot(dates[-21:],prevs[-21:],color= 'green')

	# print("predicted price \n")
	# print(pred_price)


	# apple_quote2 = web.DataReader('FB', data_source='yahoo', start='2020-01-24', end='2020-01-24')
	# print(apple_quote2['Close'])

	canvas = FigureCanvasTkAgg(fig, master=predictedFrame)  # A tk.DrawingArea.
	canvas.draw()
	canvas.get_tk_widget().grid(column = 0,columnspan=5,row = 1)
	rpredictedapp = Label(predictedFrame,text ="           Last 5 days Price           " + str(round(prevs[-20], 2)) + "           " + str(round(prevs[-19], 2)) + "           "+str(round(prevs[-18], 2))+"           "+str(round(prevs[-17], 2)) + "           "+str(round(prevs[-16], 2)) +"           " )
	rpredictedapp.grid(row = 2,columnspan=5)

	#toolbar = NavigationToolbar2Tk(canvas, Rframe)
	#toolbar.update()
	#canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
def raise_frame(frame,id):
    frame.tkraise()
    # if(id == "1"):
    # 	button2['background'] = '#1C1C9F'
    # # 	# button4.config(state = ACTIVE)
    # # 	# button4['bg'] = '#6666FF'
    # # 	# button.config(state = ACTIVE)
    # # 	# button['bg'] = '#6666FF'
    # # 	# crt("1")
    # #elif (id == "2"):
    # 	#button4['background'] = '#1C1C9F'
    # # 	# button2.config(state = ACTIVE,highlightbackground = '#6666FF')
    # # 	# # button2.config(bg = '#6666FF')
    # # 	# button.config(state = ACTIVE,highlightbackground = '#6666FF')
    # # 	# # button.config(bg = '#6666FF')
    # # 	# crt("2")
    # elif (id == "3"):
    # 	button['background'] = '#1C1C9F'
    # # 	# button2.config(state = ACTIVE,highlightbackground = '#6666FF')
    # # 	# # button2.config(bg = '#6666FF')
    # # 	# button4.config(state = ACTIVE,highlightbackground = '#6666FF')
    # # 	# # button4.config(bg = '#6666FF')
    # # 	# crt("3")

    # # else:
    # # 	print("nA")


rlabel1 = Label(aboutUs,text ="About the Company")
rlabel1.grid(row = 1)
rlabel = Label(aboutUs,text ="\t\t\tStock Forecasting Int. (Chicago, USA) provides innovative artificial price-prediction technology for active Day Traders, Short- and Long-term Investors.\nWe develop advanced web-based software for stock market forecasting and analysis. \n Our high-quality solutions are based on research done by world-renowned scientists.\nThe innovative technology and mathematical theory were presented at 58th International Atlantic Economic Conference (Chicago, USA, October 2004), \n60th International Atlantic Economic Conference (New York, USA, October 2005) and \n 98th Symposium of the Mathematical Theory of Network and System (Padova, Italy, July 1998).\nOur team is highly educated in associated fields, and we are committed to delivering the \n most accurate predictions for financial professionals, active traders, and investors.")
rlabel.grid(row = 2)
rlabel2 = Label(aboutUs,text = "About our Service")
rlabel2.grid(row = 3)
rlabel3 = Label(aboutUs,text = "\t\t\tThe artificial intelligence Stock-Forecasting software is based on neural network technology, advanced statistical methods and \n non-periodic stock price wave analysis.The Stock-Forecasting software predicts stock prices, generates trading 'Buy-Hold-Sell signals', \n computes the most profitable company to invest in and analyzes the accuracy of predictions.")
rlabel3.grid(row = 4)
clicked = IntVar()
clicked.set("1")
Radiobutton(predictedFrame,text = "Facebook",value = 1, variable = clicked,anchor = CENTER,command = fb_pred).grid(column = 0,row = 0)
Radiobutton(predictedFrame,text = "Apple",value = 2, variable = clicked,anchor = CENTER,command = app_pred).grid(column = 1,row = 0)
Radiobutton(predictedFrame,text = "Microsoft",value = 3, variable = clicked,anchor = CENTER,command = mic_pred).grid(column = 2,row = 0)
Radiobutton(predictedFrame,text = "Amazon",value = 4, variable = clicked,anchor = CENTER,command = amazon_pred).grid(column = 3,row = 0)
Radiobutton(predictedFrame,text = "Google",value = 5, variable = clicked,anchor = CENTER,command = google_pred).grid(column = 4,row = 0)
fb_pred()
# rpredict = Label(predictedFrame,text ="predict")
# rpredict.grid(column = 0,row = 0)
# rpredict = Label(liveStock,text ="liveStock")
# rpredict.grid()

#,width = 250,height = 100
lframe1 = Frame(lframe,background = '#6666FF')
lframe1.grid(column =  0 , row=0)

lframe2 = Frame(lframe,background = '#6666FF')
lframe2.grid(column =  0  ,row = 1)

lframe3 = Frame(lframe,background = '#6666FF')
lframe3.grid(column =  0  ,row = 2)

lframe4 = Frame(lframe,background = '#6666FF')
lframe4.grid(column =  0  ,row = 3)

lframe5 = Frame(lframe,background = '#6666FF')
lframe5.grid(column =  0 , row=4)

def on_enter(e):
    button2['background'] = '#1C1C9F'

def on_leave(e):
    button2['background'] = '#6666FF'

def on_enterr(e):
    button['background'] = '#1C1C9F'

def on_leaver(e):
    button['background'] = '#6666FF'

def on_entere(e):
    button4['background'] = '#1C1C9F'

def on_leavee(e):
    button4['background'] = '#6666FF'

# def crt(x):
# 	if (x == "1"):
# 		button2 = Button(lframe3, text="Live Stock Graph",fg = "#ffffff",background = '#6666FF',borderwidth=0,command = lambda:raise_frame(liveStock,"1"))
# 		button2.config(width =20 ,height =5)
# 		button2.bind("<Enter>", on_enter)
# 		button2.bind("<Leave>", on_leave)
# 		button2.grid(row = 1,column= 1)
# 	elif (x == "2"):
# 		button4 = Button(lframe4, text="Predict Stock Prices",fg = "#ffffff",background = '#6666FF',borderwidth=0,command =lambda:raise_frame( predictedFrame,"2"))
# 		button4.config(width =20 ,height =5  )
# 		button4.bind("<Enter>", on_entere)
# 		button4.bind("<Leave>", on_leavee)
# 		button4.grid(row = 2,column= 1)

# 	elif(x == "3"):
# 		button = Button(lframe5, text="ABOUT US",fg = "#ffffff",background = '#6666FF',image= filename,compound = TOP,borderwidth=0,padx = 35,pady = 5,command = lambda:raise_frame(aboutUs,"3"))
# 		button.config(width = 74,height = 50)
# 		button.bind("<Enter>", on_enterr)
# 		button.bind("<Leave>", on_leaver)
# 		button.grid(row = 3,column= 1)



photo = PhotoImage(file = "b.png")
button1 = Button(lframe1, image = photo,  background = '#6666FF',fg = "#ffffff",borderwidth=0)
button1.config(width =100 ,height =100 )
button1.grid(sticky=W+E+N+S)

# button3 = Button(lframe2, text="Click me!",fg = "#ffffff",background = '#6666FF',borderwidth=0)
# button3.config(width =20 ,height =5 )
# button3.pack()

button2 = Button(lframe3, text="Live Stock Graph",fg = "#ffffff",background = '#6666FF',borderwidth=0,command = lambda:raise_frame(liveStock,"1"))
button2.config(width =20 ,height =5)
button2.bind("<Enter>", on_enter)
button2.bind("<Leave>", on_leave)
button2.grid(row = 1,column= 1)

button4 = Button(lframe4, text="Predict Stock Prices",fg = "#ffffff",background = '#6666FF',borderwidth=0,command =lambda:raise_frame( predictedFrame,"2"))
button4.config(width =20 ,height =5  )
button4.bind("<Enter>", on_entere)
button4.bind("<Leave>", on_leavee)
button4.grid(row = 2,column= 1)

filename = PhotoImage(file = "icons8-person-16.png")
button = Button(lframe5, text="ABOUT US",fg = "#ffffff",background = '#6666FF',image= filename,compound = TOP,borderwidth=0,padx = 35,pady = 5,command = lambda:raise_frame(aboutUs,"3"))
button.config(width = 74,height = 50)
button.bind("<Enter>", on_enterr)
button.bind("<Leave>", on_leaver)
button.grid(row = 3,column= 1)

status = Label(mframe,text = "copyright by thapar",fg = "#ffffff",background = '#6666FF', anchor= E)
status.grid(row = 1,columnspan=2,sticky = W+E)


    
raise_frame(liveStock,0)
# aboutUsLabel = Label(aboutUs,text = "this is the about us section blablablabla")
# aboutUsLabel.grid()

apple_quote_live = web.DataReader('AAPL', data_source='yahoo', start='2019-10-01', end=date+'-'+month+'-'+year)
	#Create a new dataframe
new_df_live_apple = apple_quote_live.filter(['Close'])
lastval = new_df_live_apple.get('Close')
def live_apple():


	figlive = Figure(figsize=(10,4), dpi=100)
	plt.title('Model')
	plt.xlabel('Date', fontsize=2)
	plt.ylabel('Close Price USD ($)', fontsize=2)

	# plt.plot(new_df_live)
	plt.rc('xtick', labelsize=8)  
	plt.rc('ytick', labelsize=8)  
	# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
	#plt.show()
	#plt.savefig('foo.png')
	figlive.add_subplot(111).plot(new_df_live_apple)

	live1 = FigureCanvasTkAgg(figlive, master=liveStock)  # A tk.DrawingArea.
	live1.draw()
	# live1.show()
	live1.get_tk_widget().grid(column = 0,columnspan=5,row = 1)
	rlive1 = Label(liveStock,text ="           Last 5 days Price           " + str(round(lastval[-5], 2)) + "           " + str(round(lastval[-4], 2)) + "           "+str(round(lastval[-3], 2))+"           "+str(round(lastval[-2], 2)) + "           "+str(round(lastval[-1], 2)) +"           " )
	rlive1.grid(row = 2,columnspan=5)
	print("apple stck")


microsoft_quote_live = web.DataReader('MSFT', data_source='yahoo', start='2019-10-01', end=date+'-'+month+'-'+year)
	#Create a new dataframe
new_df_live_microsoft = microsoft_quote_live.filter(['Close'])
lastvalmicrosoft = new_df_live_microsoft.get('Close')
def live_microsoft():


	figlive = Figure(figsize=(10,4), dpi=100)
	plt.title('Model')
	plt.xlabel('Date', fontsize=2)
	plt.ylabel('Close Price USD ($)', fontsize=2)

	# plt.plot(new_df_live)
	plt.rc('xtick', labelsize=8)  
	plt.rc('ytick', labelsize=8)  
	# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
	#plt.show()
	#plt.savefig('foo.png')
	figlive.add_subplot(111).plot(new_df_live_microsoft)

	live1 = FigureCanvasTkAgg(figlive, master=liveStock)  # A tk.DrawingArea.
	live1.draw()
	# live1.show()
	live1.get_tk_widget().grid(column = 0,columnspan=5,row = 1)
	rlive1 = Label(liveStock,text ="           Last 5 days Price           " + str(round(lastvalmicrosoft[-5], 2)) + "           " + str(round(lastvalmicrosoft[-4], 2)) + "           "+str(round(lastvalmicrosoft[-3], 2))+"           "+str(round(lastvalmicrosoft[-2], 2)) + "           "+str(round(lastvalmicrosoft[-1], 2)) +"           " )
	rlive1.grid(row = 2,columnspan=5)
	print("microsoft stck")


amazon_quote_live = web.DataReader('AMZN', data_source='yahoo', start='2019-10-01', end=date+'-'+month+'-'+year)
	#Create a new dataframe
new_df_live_amazon = amazon_quote_live.filter(['Close'])
lastvalamazon = new_df_live_amazon.get('Close')
def live_amazon():


	figlive = Figure(figsize=(10,4), dpi=100)
	plt.title('Model')
	plt.xlabel('Date', fontsize=2)
	plt.ylabel('Close Price USD ($)', fontsize=2)

	# plt.plot(new_df_live)
	plt.rc('xtick', labelsize=8)  
	plt.rc('ytick', labelsize=8)  
	# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
	#plt.show()
	#plt.savefig('foo.png')
	figlive.add_subplot(111).plot(new_df_live_amazon)

	live1 = FigureCanvasTkAgg(figlive, master=liveStock)  # A tk.DrawingArea.
	live1.draw()
	# live1.show()
	live1.get_tk_widget().grid(column = 0,columnspan=5,row = 1)
	rlive1 = Label(liveStock,text ="           Last 5 days Price           " + str(round(lastvalamazon[-5], 2)) + "           " + str(round(lastvalamazon[-4], 2)) + "           "+str(round(lastvalamazon[-3], 2))+"           "+str(round(lastvalamazon[-2], 2)) + "           "+str(round(lastvalamazon[-1], 2)) +"           " )
	rlive1.grid(row = 2,columnspan=5)
	print("Amazon stck")


google_quote_live = web.DataReader('GOOGL', data_source='yahoo', start='2019-10-01', end=date+'-'+month+'-'+year)
	#Create a new dataframe
new_df_live_google = google_quote_live.filter(['Close'])
lastvalgoogle = new_df_live_google.get('Close')
def live_google():


	figlive = Figure(figsize=(10,4), dpi=100)
	plt.title('Model')
	plt.xlabel('Date', fontsize=2)
	plt.ylabel('Close Price USD ($)', fontsize=2)

	# plt.plot(new_df_live)
	plt.rc('xtick', labelsize=8)  
	plt.rc('ytick', labelsize=8)  
	# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
	#plt.show()
	#plt.savefig('foo.png')
	figlive.add_subplot(111).plot(new_df_live_google)

	live1 = FigureCanvasTkAgg(figlive, master=liveStock)  # A tk.DrawingArea.
	live1.draw()
	# live1.show()
	live1.get_tk_widget().grid(column = 0,columnspan=5,row = 1)
	rlive1 = Label(liveStock,text ="           Last 5 days Price           " + str(round(lastvalgoogle[-5], 2)) + "           " + str(round(lastvalgoogle[-4], 2)) + "           "+str(round(lastvalgoogle[-3], 2))+"           "+str(round(lastvalgoogle[-2], 2)) + "           "+str(round(lastvalgoogle[-1], 2)) +"           " )
	rlive1.grid(row = 2,columnspan=5)
	print("Google stck")



fb_quote_live = web.DataReader('FB', data_source='yahoo', start='2019-10-01', end=date+'-'+month+'-'+year)
new_df_live_fb = fb_quote_live.filter(['Close'])
lastvalfb = new_df_live_fb.get('Close')
def fb_live():
	

	figlive = Figure(figsize=(10,4), dpi=100)
	plt.title('Model')
	plt.xlabel('Date', fontsize=2)
	plt.ylabel('Close Price USD ($)', fontsize=2)

	# plt.plot(new_df_live)
	plt.rc('xtick', labelsize=8)  
	plt.rc('ytick', labelsize=8)  
	# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
	#plt.show()
	#plt.savefig('foo.png')
	figlive.add_subplot(111).plot(new_df_live_fb)
	# try:
	# 	live1.destroy()
	# except Exception as e:
	# 	raise
	# else:
	# 	pass
	# finally:
	# 	pass

	live2 = FigureCanvasTkAgg(figlive, master=liveStock)  # A tk.DrawingArea.
	live2.draw()
	# live2.show()
	live2.get_tk_widget().grid(column = 0,columnspan=5,row = 1)
	print("fb stck")
	rlive1 = Label(liveStock,text ="           Last 5 days Price           " + str(round(lastvalfb[-5], 2)) + "           " + str(round(lastvalfb[-4], 2)) + "           "+str(round(lastvalfb[-3], 2))+"           "+str(round(lastvalfb[-2], 2)) + "           " +str(round(lastvalfb[-1], 2))+"           ")
	rlive1.grid(row = 2,columnspan=5)
	



clickedlive = StringVar()
clickedlive.set("1")
Radiobutton(liveStock,text = "Apple",value = "1", variable = clickedlive,anchor = CENTER, command=live_apple ).grid(column = 0,row = 0)
Radiobutton(liveStock,text = "Facebook",value = "2", variable = clickedlive,anchor = CENTER,command = fb_live).grid(column = 1,row = 0)
Radiobutton(liveStock,text = "Microsoft",value = "3", variable = clickedlive,anchor = CENTER,command = live_microsoft).grid(column = 2,row = 0)
Radiobutton(liveStock,text = "Amazon",value = "4", variable = clickedlive,anchor = CENTER,command = live_amazon).grid(column = 3,row = 0)
Radiobutton(liveStock,text = "Google",value = "5", variable = clickedlive,anchor = CENTER,command = live_google).grid(column = 4,row = 0)
live_apple()



app.mainloop()