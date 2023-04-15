import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn. metrics import classification_report, roc_auc_score, roc_curve

import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.metrics import max_error
import xgboost as xgb

# 1. Read data
data = pd.read_csv("data_dropDate.csv")
data=data.drop(['Unnamed: 0'],axis=1)

#--------------
# GUI
st.title("Data Science Project")
st.write("## USA’s Avocado AveragePrice Prediction")

# Hiển thị dữ liệu sau tiền xử lý
df_cleaned=pd.read_csv("df_remove_region_encode_dummy.csv")

data_new=df_cleaned.drop(['Unnamed: 0'],axis=1)
X=data_new.drop(['AveragePrice','Date','type'],axis=1)
y=data_new.AveragePrice

# 3. Build model

# Splitting data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)

xgb_model = xgb.XGBRegressor(learning_rate = 0.2946386696029385,max_depth= 10,n_estimators =571,reg_alpha =  0.8663005771251282)
xgb_model.fit(X_train, y_train)

#4. Evaluate model
score_train=xgb_model.score(X_train, y_train)
score_test=xgb_model.score(X_test, y_test)

# GUI
menu = ["Business Objective", 'Build Project', 'New Prediction']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Business Objective':    
    st.markdown("# Business Objective")

    st.subheader('Học viên: Phạm Thuỷ Tú - Nguyễn Thị Trần Lộc')       
    st.write("Hiện tại: Công ty kinh doanh quả bơ ở rất nhiều vùng của nước Mỹ với 2 loại bơ là bơ thường và bơ hữu cơ, được đóng gói theo nhiều quy chuẩn (Small/Large/XLarge Bags), và có 3 PLU (Product Look Up) khác nhau (4046, 4225, 4770). Nhưng họ chưa có mô hình để dự đoán giá bơ cho việc mở rộng.")    
    st.write("Mục tiêu/ Vấn đề: Sử dụng các kỹ thuật trong Machine Learning, xây dựng và lựa chọn mô hình dự đoán giá trung bình của bơ “Hass” ở Mỹ",justify="center")
    st.image('avocado.jpeg', width=400)

    st.write("##### 1. Dữ liệu ban đầu")
    st.dataframe(data.head(5))
    st.dataframe(data.tail(5))

    df_continuous=data.drop(['year','Region', 'Type'],axis=1)

    st.write('Mối tương quan giữa các biến liên tục')
    f, ax = plt.subplots(1, 1, figsize=(10, 10))

    mask = np.triu(np.ones_like(df_continuous.corr()))
    ax.text(2.5, -0.1, 'Correlation Matrix', fontsize=18, fontweight='bold', fontfamily='serif')
    sns.heatmap(df_continuous.corr(), annot=True, fmt='.2f', cmap='RdBu', 
                square=True, mask=mask, linewidth=0.7, ax=ax)
    st.pyplot(f)

    #import pandas_profiling
    #profiling=pandas_profiling.ProfileReport(df_cleaned,title='Pandas Profiling Report - AVOCADO',explorative=True)
    #st.write(profiling.to_html(), unsafe_allow_html=True)
elif choice == 'Build Project':
    st.markdown("# Build Project")
    st.write("##### 2. Dữ liệu sau xử lý - đã scaled và encoded")
    st.dataframe(data_new.head(5))
    st.dataframe(data_new.tail(5))
    st.write('Tổng hợp kết quả của các models')
    df_result=pd.read_csv('result_MLmodels.csv')
    df_result=df_result.drop(['Unnamed: 0'],axis=1)
    st.dataframe(df_result)

    fig = plt.figure(figsize=(15, 4))
    gs = fig.add_gridspec(1, 2)
    gs.update(wspace=0.2)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # Title
    ax0.text(0.5, 0.5, 'Score of Models\n ___________',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=18, fontfamily='serif', fontweight='bold')
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])
    ax0.tick_params(left=False, bottom=False)
    ax0.spines['left'].set_visible(False)

    background_color = '#F8EDF4'
    color_palette = ['#F78904', '#00C73C', '#D2125E', '#693AF9', '#B20600', '#007CDE', '#994936', '#886A00', '#39BBC2']
    # Graph
    ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1,5))
    sns.barplot(x='Score', y='Model', data=df_result, palette=color_palette, ax=ax1)
    ax1.set_xlabel('')
    ax1.set_ylabel('')

    fig.patch.set_facecolor(background_color)
    axes = [ax0, ax1]

    for ax in axes:
        ax.set_facecolor(background_color)
        for s in ['top', 'right', 'bottom']:
            ax.spines[s].set_visible(False)
    st.pyplot(fig)

    st.markdown('Chọn phương pháp XGB REGRESSOR')
    st.write("##### 4. Evaluation")
    st.code("Score train:"+ str(round(score_train,2)) + " vs Score test:" + str(round(score_test,2)))
        
    y_train_pred = xgb_model.predict(X_train)
    y_test_pred = xgb_model.predict(X_test)
        
    st.code('Train Set:')
    st.code('RMSE :'+ str( mean_squared_error(y_train, y_train_pred, squared = False)))
    st.code('Max Error: '+ str(max_error(y_train, y_train_pred)))
    st.code('R2-score: '+ str(r2_score(y_train, y_train_pred)))

    st.code('Test Set:')
    st.code('RMSE :'+ str(mean_squared_error(y_test, y_test_pred, squared = False)))
    st.code('Max Error: '+ str( max_error(y_test, y_test_pred)))
    st.code('R2-score: '+ str(r2_score(y_test, y_test_pred)))

    st.write("##### 5. Summary: This model is good enough for AVOCADO classification.")
    
elif choice=='New Prediction':
    st.subheader("Select data")
        
    lines = None
    st.write("##### Hãy upload file cần dự báo giá")
    # Upload file
    uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
    if uploaded_file_1 is not None:
        lines = pd.read_csv(uploaded_file_1, header=None)
        st.dataframe(lines)
        lines = lines[0]     

    st.write("Content:")
    y_pred_new = xgb_model.predict(X_test)

    y_pred_new=pd.DataFrame(y_pred_new, columns=['y_pred_new'])
    df_result=(pd.concat([y_test.reset_index(),y_pred_new],axis=1))
    st.dataframe(df_result[['AveragePrice','y_pred_new']])