import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import time

from itertools import combinations
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm
import statsmodels.tsa.api as smt
pd.options.display.float_format = '{:.2f}'.format

import streamlit as st
from sklearn import metrics

from prophet import Prophet 
from prophet.plot import add_changepoints_to_plot

import warnings
warnings.filterwarnings('ignore')

# 1. Read data
data = pd.read_csv("df_remove_region.csv")
data["Date"] = pd.to_datetime(data["Date"])
data=data.drop(['Unnamed: 0'],axis=1)

# 2. Chuẩn bị dữ liệu
# Chọn 20 TP có tổng khối lượng bán cao nhất để dự đoán
region = data.groupby(['region'])['Total Volume'].sum().reset_index()
top_region = region.nlargest(20,'Total Volume')
top_regions=top_region['region'].tolist()

# Tạo 20 dataframe chứa dữ liệu của 20 thành phố có total volume lớn nhất
dfs = {}
for i in range(20):
    df_i = data[data['region']==top_regions[i]]
    dfs[f"{top_regions[i]}"] = df_i

# Duyệt 20 TP, tách dữ liệu mỗi thành phố thành hai loại organic và conventional
def split_df(df):
    df_organic = df[(df.type=="organic")].groupby(['Date'])['AveragePrice'].mean()
    df_organic = df_organic.to_frame(name='AveragePrice')
    df_conventional = df[(df.type=="conventional")].groupby(['Date'])['AveragePrice'].mean()
    df_conventional = df_conventional.to_frame(name='AveragePrice')
    return df_organic,df_conventional

# Đặt tên cho từng dataframe tương ứng: vd df_LosAngeles, df_LosAngeles_organic, df_LosAngeles_conventional
dfs_organic=[]
dfs_conventional=[]

dfs_organic_name=[]
dfs_conventional_name=[]

for name,df in dfs.items():
    name_ct=name
    df_ct=df
    globals()[name_ct] = df.copy()
    organic,conventional=split_df(df_ct)
    # Đặt tên cho DataFrame
    prefix = name_ct
    value1='organic'
    value2='conventional'
    df_name1 = f"{prefix}_{value1}"
    globals()[df_name1] = organic.copy()
    dfs_organic.append(globals()[df_name1])
    dfs_organic_name.append(df_name1)

    df_name2 = f"{prefix}_{value2}"
    globals()[df_name2] = conventional.copy()
    dfs_conventional.append(globals()[df_name2])
    dfs_conventional_name.append(df_name2)

# ORGANIC PREDICTION
# Xử lý tất cả các thuộc tính không ổn định (không dừng)
dfs_organic_stationary=dfs_organic
for i in range(len(dfs_organic_stationary)):
    df1=pd.DataFrame(dfs_organic_stationary[i])
    df1_log_diff = df1['AveragePrice'].diff()
    df1_log_diff = df1_log_diff.dropna()
    dfs_organic_stationary[i]=df1_log_diff
    dfs_organic_stationary[i] = pd.DataFrame(dfs_organic_stationary[i], columns=['Date', 'AveragePrice'])
    dfs_organic_stationary[i].set_index('Date')

# Tạo ra các list sẽ lần lượt chứa thông tin sau mỗi lần fit model
total_time_organic=[]
MSE_organic=[]
MAE_organic=[]
RMSE_organic=[]
Mean_df_organic=[]
Mean_test_organic=[]
Std_test_organic=[]

df_organic = pd.DataFrame(dfs_organic_name, columns=['data_region'])

# Định dạng về dạng dataframe xếp theo thứ tự tăng dần của cột Date
def weekly_average_prices(df):
    w_df = df.reset_index().dropna()
    w_df.sort_values(by=['Date'])
    return w_df

for i in range(len(dfs_organic_stationary)):
    dfs_organic_stationary[i].drop(dfs_organic_stationary[i].columns[0], axis=1, inplace=True)
    w_df=weekly_average_prices(dfs_organic_stationary[i])

# Xây dựng hàm dự báo fbprophet
def fbProphet_prediction(w_df):
    w_df=weekly_average_prices(w_df)
    w_df.columns=["ds", "y"]
    Mean_df_organic.append(w_df.y.mean())
    # train vs test: train 10 years, test 2 years
    train = w_df.drop(w_df.index[-34:])
    test = w_df.drop(w_df.index[0:-34])

    start_time=time.time()
    model=Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, changepoint_range=1) 
    #interval_width sets the uncertainty interval to produce a confidence interval around the forecast
    model.add_seasonality(name='weekly', period=7, fourier_order=5, prior_scale=0.02)
    model.fit(train)

    end_time=time.time()
    total_time=end_time-start_time
    total_time_organic.append(total_time)

    # calculate MAE/RMSE between expected and predicted values for december
    future = model.make_future_dataframe(freq='W', periods=7)  # Let's predict the next month's average prices
    forecast=model.predict(future)
    y_test = test['y'].values
    y_pred = forecast['yhat'].values[:34]

    mae_p = mean_absolute_error(y_test, y_pred)
    MAE_organic.append(mae_p)

    rmse_p = np.sqrt(mean_squared_error(y_test, y_pred))
    RMSE_organic.append(rmse_p)

    Mean_test_organic.append(test.y.mean())
    Std_test_organic.append(test.y.std())

# Duyệt qua từng df của organic và lưu lại kết quả đánh giá của từng df
for i in range(len(dfs_organic_stationary)):
    fbProphet_prediction(dfs_organic_stationary[i])

df_organic['Time']=total_time_organic
df_organic['MAE']=MAE_organic
df_organic['RMSE']=RMSE_organic
df_organic['mean_df']=Mean_df_organic
df_organic['mean_test']=Mean_test_organic
df_organic['std_text']=Std_test_organic
df_organic['RMSE']=RMSE_organic

df_organic['Eval']=df_organic.apply(lambda x: 'Good' if (x['MAE'] <x['std_text']) and (x['RMSE'] <x['std_text']) else 'Not_Good', axis=1)

df_organic.insert(0,'Type','organic')

# CONVENTIONAL PREDICTION
dfs_conventional_stationary=dfs_conventional
for i in range(len(dfs_conventional_stationary)):
    df1=pd.DataFrame(dfs_conventional_stationary[i])
    df1['Log_AveragePrice']=np.log(df1['AveragePrice'])
    df1_log_diff = df1['AveragePrice'].diff()
    df1_log_diff = df1_log_diff.dropna()
    dfs_conventional_stationary[i]=df1_log_diff
    dfs_conventional_stationary[i] = pd.DataFrame(dfs_conventional_stationary[i], columns=['Date', 'AveragePrice'])
    dfs_conventional_stationary[i].set_index('Date')

# Tạo ra các list sẽ lần lượt chứa thông tin sau mỗi lần fit model
total_time_conventional=[]
MSE_conventional=[]
MAE_conventional=[]
RMSE_conventional=[]
Mean_df_conventional=[]
Mean_test_conventional=[]
Std_test_conventional=[]
Class_conventional=[]

df_conventional = pd.DataFrame(dfs_conventional_name, columns=['data_region'])

for i in range(len(dfs_conventional_stationary)):
    dfs_conventional_stationary[i].drop(dfs_conventional_stationary[i].columns[0], axis=1, inplace=True)
    w_df=weekly_average_prices(dfs_conventional_stationary[i])

def fbProphet_prediction(w_df):
    w_df=weekly_average_prices(w_df)
    w_df = w_df.dropna()
    w_df.columns=["ds", "y"]
    Mean_df_conventional.append(w_df.y.mean())

    # train vs test:
    train = w_df.drop(w_df.index[-34:])
    test = w_df.drop(w_df.index[0:-34])

    start_time=time.time()
    model=Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, changepoint_range=1)  
    #interval_width sets the uncertainty interval to produce a confidence interval around the forecast
    model.add_seasonality(name='weekly', period=52, fourier_order=5, prior_scale=0.02)
    model.fit(train)

    end_time=time.time()
    total_time=end_time-start_time
    total_time_conventional.append(total_time)

    # calculate MAE/RMSE between expected and predicted values
    future = model.make_future_dataframe(freq='W', periods=7)  # Let's predict the next week's average prices
    forecast=model.predict(future)
    y_test = test['y'].values
    y_pred = forecast['yhat'].values[:34]

    mae_p = mean_absolute_error(y_test, y_pred)
    MAE_conventional.append(mae_p)

    rmse_p = np.sqrt(mean_squared_error(y_test, y_pred))
    RMSE_conventional.append(rmse_p)

    Mean_test_conventional.append(test.y.mean())
    Std_test_conventional.append(test.y.std())

for i in range(len(dfs_conventional_stationary)):
    fbProphet_prediction(dfs_conventional_stationary[i])

df_conventional['Time']=total_time_conventional
df_conventional['MAE']=MAE_conventional
df_conventional['RMSE']=RMSE_conventional
df_conventional['mean_df']=Mean_df_conventional
df_conventional['mean_test']=Mean_test_conventional
df_conventional['std_text']=Std_test_conventional
df_conventional['RMSE']=RMSE_conventional

df_conventional.insert(0,'Type','conventional')
df_conventional['Eval']=df_conventional.apply(lambda x: 'Good' if (x['MAE'] <x['std_text']) and (x['RMSE'] <x['std_text']) else 'Not_Good', axis=1)

df_eval_full=pd.concat([df_conventional,df_organic],axis=0)
df_eval_full.reset_index(drop=True, inplace=True)
df_eval_full_good=df_eval_full[df_eval_full['Eval']=='Good']

#--------------
# GUI
st.title("Data Science Project")
st.write("## USA’s Avocado AveragePrice Prediction")

menu = ["Business Objective", 'Build Project', 'New Prediction']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Business Objective':    
    st.markdown("# Business Objective")

    st.subheader('Học viên: Phạm Thuỷ Tú - Nguyễn Thị Trần Lộc')       
    st.write("Hiện tại: Công ty kinh doanh quả bơ ở rất nhiều vùng của nước Mỹ với 2 loại bơ là bơ thường và bơ hữu cơ, được đóng gói theo nhiều quy chuẩn (Small/Large/XLarge Bags), và có 3 PLU (Product Look Up) khác nhau (4046, 4225, 4770). Nhưng họ chưa có mô hình để dự đoán giá bơ cho việc mở rộng.")  
    st.write("Mục tiêu/ Vấn đề: Xây dựng mô hình dự đoán giá trung bình của bơ “Hass” ở Mỹ. Từ đó xem xét việc mở rộng sản xuất, kinh doanh.",justify="center")    
    st.subheader("Sử dụng phương pháp FBPROPHET dự đoán xu hướng giá trung bình. Tuỳ chọn vùng dự đoán")
    st.image('avocado.jpeg', width=400)

elif choice == 'Build Project':

    st.subheader("Build Project")
    st.write("##### 1. Dữ liệu ban đầu")
    st.dataframe(data.head(3))
    st.dataframe(data.tail(3))  

    # Xem thông tin tổng quan
    st.write('Thống kê mô tả của toàn bộ dữ liệu')
    st.write(data.describe().T)

    st.write('Chọn 20 TP có tổng khối lượng bán cao nhất để dự đoán')
    
    fig=plt.figure(figsize=(12,5));
    ax = sns.barplot(x="region", y="Total Volume", data=top_region,color="b")
    fig.suptitle('20 thành phố có Total Volume cao nhất', fontsize=15)
    st.pyplot(fig)

    st.write('Danh sách các region có tổng lượng bán cao nhất: ',top_regions)
        
    st.write("##### 4. Build model and Evaluation")
    st.write('Kết quả dự đoán trên dữ liệu bơ Organic của 20 vùng có tổng lượng bán ra cao nhất')
    st.dataframe(df_organic.head(10))
    st.write('Kết quả dự đoán trên dữ liệu bơ Convetional của 20 vùng có tổng lượng bán ra cao nhất')
    st.dataframe(df_conventional.head(10))
    st.write('Kết quả dự đoán trên dữ liệu cả hai loại bơ Organic và Conventional của 20 vùng có tổng lượng bán ra cao nhất')
    st.dataframe(df_eval_full)
    st.write('Kết quả dự đoán tốt trên dữ liệu bơ Organic và Conventional của 20 vùng có tổng lượng bán ra cao nhất')
    st.dataframe(df_eval_full_good)

    st.write("##### 5. Summary: This model is good enough for AVOCADO")

elif choice=='New Prediction':
    option = st.selectbox('Chọn thành phố',('LosAngeles','DallasFtWorth','Houston','PhoenixTucson','WestTexNewMexico',
 'Denver','SanFrancisco','BaltimoreWashington','Chicago','Portland','Seattle','MiamiFtLauderdale','Boston',
 'SanDiego','Atlanta','Sacramento','Philadelphia','NorthernNewEngland','Tampa','Detroit'))

    text_select=option
    number = st.number_input('Hãy nhập số tuần muốn dự đoán:',step=1, format="%d")

    st.write('Thành phố đã chọn: ',text_select)
    st.write('Số tuần muốn dự đoán: ', number)

    st.write("##### 1. Dữ liệu ban đầu về ",text_select)
    df_selected = data[data.region==text_select]
    st.write(df_selected.info())
    
    st.dataframe(df_selected.head(3))
    st.dataframe(df_selected.tail(3))

    st.write('Thống kê mô tả về dữ liệu của ',text_select)
    st.write(df_selected.describe().T)

    

    df_selected_organic,df_selected_conventional=split_df(df_selected)

    df_selected_organic['Log_AveragePrice']=np.log(df_selected_organic['AveragePrice'])
    df_selected_organic_diff = df_selected_organic['AveragePrice'].diff()
    df_selected_organic_diff = df_selected_organic_diff.dropna()

    df_selected_conventional['Log_AveragePrice']=np.log(df_selected_conventional['AveragePrice'])
    df_selected_conventional_diff = df_selected_conventional['AveragePrice'].diff()
    df_selected_conventional_diff = df_selected_conventional_diff.dropna()

    w_df_organic=weekly_average_prices(df_selected_organic_diff)
    w_df_conventional=weekly_average_prices(df_selected_conventional_diff)    

    w_df_organic.columns=["ds", "y"]
    w_df_conventional.columns=["ds", "y"]

    len_train_organic=int(len(df_selected_organic_diff)*0.7)
    len_test_organic=int(len(df_selected_organic_diff)*0.3)

    len_train_conventional=int(len(df_selected_conventional_diff)*0.7)
    len_test_conventional=int(len(df_selected_conventional_diff)*0.3)

    # train vs test organic
    train_organic_df_selected = w_df_organic.drop(w_df_organic.index[-len_test_organic:])
    test_organic_df_selected = w_df_organic.drop(w_df_organic.index[0:-len_test_organic])

    # train vs test conventional
    train_conventional_df_selected = w_df_organic.drop(w_df_organic.index[-len_test_conventional:])
    test_conventional_df_selected = w_df_organic.drop(w_df_organic.index[0:-len_test_conventional])

    st.header('ORGANIC PREDICTION')

    model_organic=Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, changepoint_range=1)
    model_organic.add_seasonality(name='weekly', period=52, fourier_order=5, prior_scale=0.02)
    model_organic.fit(w_df_organic)

    future_organic = model_organic.make_future_dataframe(freq='W', periods=number)  # Let's predict the next month's average prices
    forecast_organic=model_organic.predict(future_organic)
    st.dataframe(forecast_organic[['ds','yhat']].tail(number))
    st.dataframe(forecast_organic[['ds','yhat']].tail(number))
    st.dataframe(forecast_organic[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']].tail(number))

    y_test = test_organic_df_selected['y'].values
    y_pred = forecast_organic['yhat'].values[:len_test_organic]

    y_test_value = pd.DataFrame(y_test, index = pd.to_datetime(test_organic_df_selected['ds']),columns=['Actual'])
    y_pred_value = pd.DataFrame(y_pred, index = pd.to_datetime(test_organic_df_selected['ds']),columns=['Prediction'])

    # Visulaize the result
    fig1 =plt.figure(figsize=(12,6))
    plt.plot(y_test_value, label='Real AveragePrice')
    plt.plot(y_pred_value, label='Prediction AveragePrice')
    plt.xticks(rotation='vertical')
    plt.legend()
    st.pyplot(fig1)

    fig2 = model_organic.plot(forecast_organic)
    a = add_changepoints_to_plot(fig2.gca(), model_organic, forecast_organic)
    st.pyplot(fig2)
    
    fig3= plt.figure(figsize=(15,8))
    plt.plot(w_df_organic['y'], label='AveragePrice')
    plt.plot(forecast_organic['yhat'], label='AveragePrice with next number months prediction', 
         color='red')
    plt.xticks(rotation='vertical')
    plt.legend()
    st.pyplot(fig3)

    st.header('CONVENTIONAL PREDICTION')

    model_conventional=Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, changepoint_range=1)
    model_conventional.add_seasonality(name='weekly', period=52, fourier_order=5, prior_scale=0.02)
    model_conventional.fit(w_df_conventional)

    future_conventional = model_conventional.make_future_dataframe(freq='W', periods=number)  # Let's predict the next month's average prices
    forecast_conventional=model_conventional.predict(future_conventional)
    st.dataframe(forecast_conventional[['ds','yhat']].tail(number))
    st.dataframe(forecast_conventional[['ds','yhat']].tail(number))
    st.dataframe(forecast_conventional[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']].tail(number))

    y_test = test_conventional_df_selected['y'].values
    y_pred = forecast_conventional['yhat'].values[:len_test_organic]

    y_test_value = pd.DataFrame(y_test, index = pd.to_datetime(test_conventional_df_selected['ds']),columns=['Actual'])
    y_pred_value = pd.DataFrame(y_pred, index = pd.to_datetime(test_conventional_df_selected['ds']),columns=['Prediction'])

    # Visulaize the result
    fig1 =plt.figure(figsize=(12,6))
    plt.plot(y_test_value, label='Real AveragePrice')
    plt.plot(y_pred_value, label='Prediction AveragePrice')
    plt.xticks(rotation='vertical')
    plt.legend()
    st.pyplot(fig1)

    fig2 = model_conventional.plot(forecast_conventional)
    a = add_changepoints_to_plot(fig2.gca(), model_conventional, forecast_conventional)
    st.pyplot(fig2)
    
    fig3= plt.figure(figsize=(15,8))
    plt.plot(w_df_conventional['y'], label='AveragePrice')
    plt.plot(forecast_conventional['yhat'], label='AveragePrice with next number months prediction', 
         color='red')
    plt.xticks(rotation='vertical')
    plt.legend()
    st.pyplot(fig3)