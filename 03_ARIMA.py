import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from datetime import datetime
import time

from itertools import combinations
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA as ARIMA
import statsmodels.api as sm
import statsmodels.tsa.api as smt
pd.options.display.float_format = '{:.2f}'.format

import pickle
import streamlit as st
from sklearn import metrics

#--------------
# GUI
st.title("Data Science Project")
st.header("USA’s Avocado AveragePrice Prediction")
st.markdown("Business Objective")

st.subheader('Học viên: Phạm Thuỷ Tú - Nguyễn Thị Trần Lộc')       
st.write("Hiện tại: Công ty kinh doanh quả bơ ở rất nhiều vùng của nước Mỹ với 2 loại bơ là bơ thường và bơ hữu cơ, được đóng gói theo nhiều quy chuẩn (Small/Large/XLarge Bags), và có 3 PLU (Product Look Up) khác nhau (4046, 4225, 4770). Nhưng họ chưa có mô hình để dự đoán giá bơ cho việc mở rộng.")  
st.write("Mục tiêu/ Vấn đề: Xây dựng mô hình dự đoán giá trung bình của bơ “Hass” ở Mỹ. Từ đó xem xét việc mở rộng sản xuất, kinh doanh.",justify="center")    
st.write("#### Sử dụng phương pháp ARIMA dự đoán xu hướng giá trung bình. Chọn vùng dự đoán LosAngeles")
st.image('avocado.jpeg', width=400)

# 1. Read data
data = pd.read_csv("df_remove_region.csv")
data["Date"] = pd.to_datetime(data["Date"])
data=data.drop(['Unnamed: 0'],axis=1)

# 2. Chuẩn bị dữ liệu
# Chọn LosAngeles để dự đoán
st.subheader('Chọn LOSANGELES để dự đoán')
df_LosAngeles = data[data.region=='LosAngeles']
# Tách thành hai tập con theo từng loại Conventional và Organic
df_organic = df_LosAngeles[(df_LosAngeles.type=="organic")].groupby(['Date'])['AveragePrice'].mean()
df_organic = df_organic.resample('W').mean()
df_organic = df_organic.to_frame(name='AveragePrice')

df_conventional = df_LosAngeles[(df_LosAngeles.type=="conventional")].groupby(['Date'])['AveragePrice'].mean()
df_conventional = df_conventional.resample('W').mean()
df_conventional = df_conventional.to_frame(name='AveragePrice')


@st.cache_data
def visual_plot(df):
    df=df.reset_index()
    df.sort_values(by=['Date'])
    fig2 = plt.figure(figsize = (27, 12))
    ax = plt.axes()
    #set ticks every month
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    #set major ticks format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.plot(df['Date'],df['AveragePrice'],color='b', linewidth=1)
    plt.xlabel("2015-2018")
    plt.ylabel("Avocado Price USD")
    plt.legend()
    st.pyplot(fig2)

@st.cache_data
def df_diff(df):
    df=df['AveragePrice'].diff()
    df=df.to_frame()
    df=df.dropna()
    # Tính toán phức tạp
    return df

# Xử lý tính dừng cho bộ dữ liệu
df_organic_diff= df_diff(df_organic)

df_conventional_diff= df_diff(df_conventional)

# Splitting data into training and test set
train_size = int(len(df_organic_diff) * 0.75)
train_organic, test_organic = df_organic_diff[0:train_size], df_organic_diff[train_size:]

train_size = int(len(df_conventional_diff) * 0.75)
train_conventional, test_conventional = df_conventional_diff[0:train_size], df_conventional_diff[train_size:]

# 3. Build model
@st.cache_data
def build_model(df):
    model_sarimax = sm.tsa.statespace.SARIMAX(df,order = (1,1,0),seasonal_order = (1,1,1,52))
    model_fit_sarimax = model_sarimax.fit()
    return model_fit_sarimax

start_time=time.time()
model_fit_sarimax_organic=build_model(train_organic)
end_time=time.time()
total_time_organic=end_time-start_time

start_time=time.time()
model_fit_sarimax_conventional=build_model(train_conventional)
end_time=time.time()
total_time_conventional=end_time-start_time

#4. Evaluate model and prediction

def predict_func(model_fit,start,end):
    pred_test = model_fit.predict(start = start, end =end,dynamic=True)
    pred_test

    pred_range = model_fit.get_prediction(start = start, end = end,dynamic=True)
    pred_ci_range = pred_range.conf_int()
    pred_ci_range

    fig, ax = plt.subplots()       
    ax.plot(pred_test,color='red',label = 'Predicted Values')
    ax.plot(model_fit.fittedvalues, color='blue',label = 'Real Values');
    plt.legend(loc="upper left")
    st.pyplot(fig)

# GUI
menu = ["Business Objective", 'Build Project', 'New Prediction']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Business Objective':    

    st.markdown('Thông tin tổng quan về bộ dữ liệu')
    st.write("##### 1. Dữ liệu ban đầu")
    st.dataframe(data.head(3))
    st.dataframe(data.tail(3))

    st.subheader('Thông tin tổng quan về bộ dữ liệu LOSANGLES')

    st.dataframe(df_LosAngeles.head(3))
    st.dataframe(df_LosAngeles.tail(3))  

    import matplotlib.dates as mdates

    st.write('Data chuỗi thời gian LOSANGELES loại Organic ban đầu: ')
    visual_plot(df_organic)

    st.write('Data chuỗi thời gian LOSANGELES loại Conventional ban đầu: ')
    visual_plot(df_conventional)

    st.subheader('Data chuỗi thời gian LOSANGELES sau khi thực hiện sai phân: ')
    st.write('Data chuỗi thời gian LOSANGELES loại Organic sau khi thực hiện sai phân: ')

    visual_plot(df_organic_diff)

    st.write('Data chuỗi thời gian LOSANGELES loại Conventional sau khi loại bỏ tính dừng: ')
    
    visual_plot(df_conventional_diff)

elif choice == 'Build Project':

    st.subheader("Build Project")
    st.write("##### 1. Dữ liệu ban đầu")
    st.dataframe(data.head(3))
    st.dataframe(data.tail(3))  

    st.write("##### 2. Dữ liệu LosAngeles")
    st.dataframe(df_LosAngeles.head(3))
    st.dataframe(df_LosAngeles.tail(3))

    st.write("##### 3.1. Dữ liệu LosAngeles loại Organic")
    st.dataframe(df_organic.head(3))
    st.write("##### 3.2. Dữ liệu LosAngeles loại Conventional")
    st.dataframe(df_conventional.head(3))

    # Xem thông tin tổng quan
    st.write('Thống kê mô tả của dữ liệu Bơ Organic tại TP LosAngeles')
    st.write(df_organic.describe().T)
    st.write('Thống kê mô tả của dữ liệu Bơ Conventional tại TP LosAngeles')
    st.write(df_organic.describe().T)

    st.write("##### 4. Build model...")
    st.code('Thời gian thực hiện model trên bơ Organic tại LosAngeles '+str(total_time_organic)+' giây')
    st.write('Tham số tối ưu trong mô hình dự đoán trên bơ Organic tại LosAngeles')
    st.write(model_fit_sarimax_organic.summary())
    st.code('Thời gian thực hiện model trên bơ Conventional tại LosAngeles '+str(total_time_conventional)+' giây')

    st.write("##### 5. Evaluation")
    pred_organic = model_fit_sarimax_organic.predict()
    pred_conventional = model_fit_sarimax_conventional.predict()
    
    st.write('Độ chính xác dự đoán trên dữ liệu Bơ Organic tại LosAngeles') 
    st.code('RMSE :'+ str( np.sqrt(mean_squared_error(train_organic, pred_organic))))
    st.code('MSE :'+ str( mean_squared_error(train_organic, pred_organic)))
    mae_p = mean_absolute_error(train_organic, pred_organic)
    st.code('MAE: %.3f' % mae_p)

    st.write('Độ chính xác dự đoán trên dữ liệu Bơ Conventional tại LosAngeles')
    st.code('RMSE :'+ str( np.sqrt(mean_squared_error(train_conventional, pred_conventional))))
    st.code('MSE :'+ str( mean_squared_error(train_conventional, pred_conventional)))
    mae_p = mean_absolute_error(train_conventional, pred_conventional)
    st.code('MAE: %.3f' % mae_p)

    fig, ax = plt.subplots()       
    ax.plot(train_organic)
    ax.plot(model_fit_sarimax_organic.fittedvalues, color='red');
    st.pyplot(fig)

    st.write("##### 5. Summary: This model is good enough for AVOCADO in LosAngeles")

elif choice=='New Prediction':

    st.subheader('Kết quả dự đoán trong 20 tuần trong tương lai: ')

    predict_func(model_fit_sarimax_organic,135,185)
    predict_func(model_fit_sarimax_conventional,135,185)