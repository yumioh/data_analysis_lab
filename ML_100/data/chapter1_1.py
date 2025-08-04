import pandas as pd
import os
import glob

# 파일 로딩
m_store = pd.read_csv("./ML_100/data/m_store.csv")
m_area = pd.read_csv("./ML_100/data/m_area.csv")

# 주문 데이터 로딩
current_dir = os.getcwd()
tbl_order_file = os.path.join(current_dir+'\\ML_100\\data\\', 'tbl_order_*.csv')
tbl_order_files = glob.glob(tbl_order_file)
order_all = pd.DataFrame()

for file_ in tbl_order_files:
    order_data = pd.read_csv(file_)
    order_all = pd.concat([order_all, order_data], ignore_index=True)

print(order_all.head())
# 불필요한 데이터 제거 
order_data = order_all.loc[order_all['store_id'] != 999]

# 마스터 데이터 결합
order_data = pd.merge(order_data, m_store, on='store_id', how='left')
order_data = pd.merge(order_data, m_area, on="area_cd", how='left')

# 이름설정 
order_data.loc[order_data['takeout_flag'] == 0, 'takeout_name'] = 'delivery'
order_data.loc[order_data['takeout_flag'] == 1, 'takeout_name'] = 'takeout'

# 이름 설정(주문상태)
order_data.loc[order_data['status'] == 0, 'status_name'] = '주문접수'
order_data.loc[order_data['status'] == 1, 'status_name'] = '결제완료'
order_data.loc[order_data['status'] == 2, 'status_name'] = '배달완료'
order_data.loc[order_data['status'] == 3, 'status_name'] = '주문취소'

# 파일에 저장
# output_dir = os.path.join(currend_dir, 'output_data')
# os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(current_dir+'\\ML_100\\data\\', 'order_data.csv')
order_data.to_csv(output_file, index=False)