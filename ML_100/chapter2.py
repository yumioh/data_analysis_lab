import pandas as pd
import warnings

warnings.filterwarnings('ignore')

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

print(analyze_data.dtypes)

# 날짜로 그룹핑 하기 
month_data = analyze_data.groupby('order_accept_month')
print(month_data.describe())
print(month_data.sum())

# p36