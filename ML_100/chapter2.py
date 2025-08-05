import pandas as pd
import matplotlib.pyplot as plt
import os 

import warnings
warnings.filterwarnings('ignore')

#한글 폰트 처리 
if os.name == 'nt' :  # windows
    plt.rc('font', family='Malgun Gothic')
elif os.name == 'posix' : # macos
    plt.rc('font', family = 'AllieGothic')

order_data = pd.read_csv("./ML_100/data/order_data.csv")
#print(order_data.head())

# status가 1 or 2인 경우만 추출
order_data = order_data.loc[(order_data['status'] == 1) | (order_data['status'] == 2)]
#print(order_data.columns)

# 사용할 컬럼만 추출
analyze_data = order_data[['store_id', 'customer_id', 'coupon_cd','order_accept_date', 'delivered_date','total_amount','store_name','wide_area', 'narrow_area','takeout_name', 'status_name']]

# 형변환
analyze_data[['store_id','coupon_cd']] = analyze_data[['store_id','coupon_cd']].astype(str)

#print(analyze_data.dtypes)


#월별 매출 집계하기 
analyze_data['order_accept_date'] = pd.to_datetime(analyze_data['order_accept_date'])
analyze_data['order_accept_month'] = analyze_data['order_accept_date'].dt.strftime('%Y%m')
#print(analyze_data[['order_accept_date','order_accept_month']].head())

analyze_data['delivered_date'] = pd.to_datetime(analyze_data['delivered_date'])
analyze_data['delivered_month'] = analyze_data['delivered_date'].dt.strftime('%Y%m')
#print(analyze_data[['delivered_date','delivered_month']].head())

#print(analyze_data.dtypes)

# 날짜로 그룹핑 하기 
month_data = analyze_data.groupby('order_accept_month')
#print(month_data.describe())
#print(month_data.sum(numeric_only=True)) # object형 타입 컬럼을 무시하고, int64, 

# 그래프 그리기
#plt.rc('axes', unicode_minus=False) # minus font setting
#month_data.sum(numeric_only=True).plot()

plt.hist(analyze_data['total_amount'], bins=21)
#plt.show()

#시,군,군,구별 매출 집계
pre_data = pd.pivot_table(
    analyze_data, index='order_accept_month',
    columns='narrow_area', values = 'total_amount',
    aggfunc='mean'
)

#print(pre_data)

# x축 값으로 사용할 인덱스를 리스트 형대로 변환
# y축 값으로 사용할 데이터

# plt.plot(list(pre_data.index), pre_data['서울'], label='서울')
# plt.plot(list(pre_data.index), pre_data['부산'], label='부산')
# plt.plot(list(pre_data.index), pre_data['대전'], label='대전')
# plt.plot(list(pre_data.index), pre_data['광주'], label='광주')
# plt.plot(list(pre_data.index), pre_data['세종'], label='세종')
# plt.plot(list(pre_data.index), pre_data['경기남부'], label='경기남부')
# plt.plot(list(pre_data.index), pre_data['경기북부'], label='경기북부')
# plt.legend()

#plt.show()

# agg : 그룹별로 합계, 평균, 개수, 최대값 등 다양한 통게 계산을 한 번에 처리할 수 있게 해주는 메서드
# reset_index(drop=True) : 인덱스 초기화 
store_clustering = analyze_data.groupby('store_id')
store_clustering_= store_clustering['total_amount'].agg(['mean', 'median', 'max', 'min', 'size']).reset_index(drop=True)
print(store_clustering_)

import seaborn as sns
sns.jointplot(x="mean", y="size", data=store_clustering_, kind='hex')

# 클러스터링 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# _winapi.CreateProcess 내부에서 병렬 처리를 위해 코어 수를 확인하려고 사용하려는데 내부적으로 호출할때 문제 발생시 나옴 
sc = StandardScaler()
store_clustering_sc = sc.fit_transform(store_clustering_)

kmeans = KMeans(n_clusters=4, random_state=0, n_init='auto')
clusters = kmeans.fit(store_clustering_sc)
store_clustering_['cluster'] = clusters.labels_
print(store_clustering_['cluster'].unique())
print(store_clustering_.head())