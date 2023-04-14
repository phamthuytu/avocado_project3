import numpy as np
import pandas as pd

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
from sklearn.ensemble import RandomForestRegressor

import streamlit as st

st.markdown("# Business Objective")

st.write('#### Học viên: Phạm Thuỷ Tú - Nguyễn Thị Trần Lộc')
    
st.write("""Hiện tại: Công ty kinh doanh quả bơ ở rất nhiều vùng của nước Mỹ với 2 loại bơ là bơ thường và bơ hữu cơ, được đóng gói theo nhiều quy chuẩn (Small/Large/XLarge Bags), và có 3 PLU (Product Look Up) khác nhau (4046, 4225, 4770). Nhưng họ chưa có mô hình để dự đoán giá bơ cho việc mở rộng.""")  
st.write("""Mục tiêu/ Vấn đề: Xây dựng mô hình dự đoán giá trung bình của bơ “Hass” ở Mỹ. Từ đó xem xét việc mở rộng sản xuất, kinh doanh.""",justify="center")
st.image('avocado.jpeg', width=400)

st.write('Thông tin tổng quan về bộ dữ liệu AVOCADO')

st.write("""Toàn bộ dữ liệu được đổ ra và lưu trữ trong tập tin avocado.csv với 18249 record, bao gồm 11 cột như sau:

* Date - ngày ghi nhận

* AveragePrice – giá trung bình của một quả bơ

* Type - conventional / organic – loại: thông thường/ hữu cơ

* Region – vùng được bán

* Total Volume – tổng số bơ đã bán

* 4046 – tổng số bơ có mã PLU 4046 đã bán

* 4225 - tổng số bơ có mã PLU 4225 đã bán

* 4770 - tổng số bơ có mã PLU 4770 đã bán

* Total Bags – tổng số túi đã bán

* Small/Large/XLarge Bags – tổng số túi đã bán theo size""")

# 1. Read data
data = pd.read_csv("avocado.csv")
data=data.drop(['Unnamed: 0'],axis=1)

st.dataframe(data.head(10))
st.dataframe(data.tail(10))

st.write(data.describe().T)

st.write("Vì có sự chồng chéo dữ liệu giữa các vùng lớn và thành phố con nên đã loại đi bớt dữ liệu")
regionsToRemove = ['California', 'GreatLakes', 'Midsouth', 'NewYork', 'Northeast', 'SouthCarolina', 'Plains', 'SouthCentral', 'Southeast', 'TotalUS', 'West']
df = data[~data.region.isin(regionsToRemove)]
len(df.region.unique())
st.write('Các vùng đã bị loại khỏi bộ dữ liệu nghiên cứu',regionsToRemove)

st.write('Bộ dữ liệu dùng dự đoán', df['region'].unique())
st.write(df.describe().T)

background_color = '#F8EDF4'
color_palette = ['#F78904', '#00C73C', '#D2125E', '#693AF9', '#B20600', '#007CDE', '#994936', '#886A00', '#39BBC2']

fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(3, 2)
gs.update(hspace=0.2, wspace=0.3)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, 0])
ax5 = fig.add_subplot(gs[2, 1])
fig.patch.set_facecolor(background_color)

axes = [ax0, ax1, ax2, ax3, ax4, ax5]


# Title1
ax0.text(0.5, 0.5, 'Distribution of AveragePrice\n____________________',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=18, fontweight='bold', fontfamily='serif')

# Graph1
sns.kdeplot(x='AveragePrice', data=df, fill=True, ax=ax1, color=color_palette[0])



# Title2
ax2.text(0.5, 0.5, 'Distribution of AveragePrice\nby Type\n____________________',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=18, fontweight='bold', fontfamily='serif')

# Graph2
sns.kdeplot(x='AveragePrice', data=df, fill=True, hue='type', ax=ax3, palette=color_palette[:2])



# Title3
ax4.text(0.5, 0.5, 'Distribution of AveragePrice\nby Year\n____________________',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=18, fontweight='bold', fontfamily='serif')

# Graph3
sns.kdeplot(x='AveragePrice', data=df, fill=True, hue='year', ax=ax5, palette=color_palette[:4])



# Settings
for ax in axes:
    ax.set_facecolor(background_color)
    for s in ['top', 'right', 'left']:
        ax.spines[s].set_visible(False)

for ax in [ax0, ax2, ax4]:
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(left=False, bottom=False)
    ax.spines[['bottom']].set_visible(False)
        
for ax in [ax1, ax3, ax5]:
    ax.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1,5))
    ax.set_xlabel('')
    ax.set_ylabel('')

st.pyplot(fig)

fig2= plt.figure(figsize=(10,11))
plt.title("Avg.Price of Avocado by Region")
Av= sns.barplot(x="AveragePrice",y="region",data= df)
st.pyplot(fig2)

df_organic=df[df['type']=='organic']
df_conventional=df[df['type']=='conventional']

fig3= plt.figure(figsize=(10,11))
plt.title("Avg.Price of Organic Hass")
Av= sns.barplot(x="AveragePrice",y="region",data= df_organic)
st.pyplot(fig3)

fig4= plt.figure(figsize=(10,11))
plt.title("Avg.Price of Conventional Hass")
Av= sns.barplot(x="AveragePrice",y="region",data= df_conventional)
st.pyplot(fig4)

fig5=plt.figure(figsize=(6,3))
plt.title("Avg.Price of Avocados by Type")
Av= sns.barplot(x="type",y="AveragePrice",data= df)
st.pyplot(fig5)

categorical_features=['type','year']
numerical_features=['Total Volume', '4046', '4225', '4770',
       'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']

color1 = ['#296C92','#3EB489']
fig6, ax = plt.subplots(nrows = 1,ncols = 2,figsize = (10,5))
for i in range(len(categorical_features)):
    
    plt.subplot(1,2,i+1)
    sns.barplot(x = categorical_features[i],y = 'AveragePrice',data = df,palette = color1,edgecolor = 'black')
    title = categorical_features[i] + ' vs AveragePrice'
    plt.title(title)
st.pyplot(fig6)