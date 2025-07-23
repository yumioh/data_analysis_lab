import pandas as pd
import os
import glob

m_store = pd.read_csv("./ML_100/data/m_store.csv")
m_area = pd.read_csv("./ML_100/data/m_area.csv")

tbl_order_4=pd.read_csv("./ML_100/data/tbl_order_202104.csv")
tbl_order_5=pd.read_csv("./ML_100/data/tbl_order_202105.csv")
tbl_order_6=pd.read_csv("./ML_100/data/tbl_order_202106.csv")

# ignore_index = 인덱스 0부터 다시 시작
order_all = pd.concat([tbl_order_4, tbl_order_5], ignore_index=True)

current_dir = os.getcwd()

tbl_order_file = os.path.join(current_dir+'\\ML_100\\data\\', "tbl_order_*.csv")
tbl_order_files = glob.glob(tbl_order_file)

order_all = pd.DataFrame()

for file_ in tbl_order_files :
    order_data = pd.read_csv(file_)
    #print(f'{file_}:len{order_data}')
    order_all = pd.concat([order_all, order_data], ignore_index=True)

# 결손값 확인
#print(order_all.isnull().sum())

order_data = order_all.loc[order_all['store_id'] != 999] 
#print(order_data)

# left join
order_data = pd.merge(order_data, m_store, on="store_id", how="left")

order_data = pd.merge(order_data, m_area, on="area_cd", how="left")

order_data.loc[order_data['takeout_flag'] == 0, 'takeout_name'] = 'delivery'
order_data.loc[order_data['takeout_flag'] == 1, 'takeout_name'] = 'takeout'
print(order_data)

order_data.loc[order_data['status'] == 0, 'status_name'] = '주문접수'
order_data.loc[order_data['status'] == 1, 'status_name'] = '지불완료'
order_data.loc[order_data['status'] == 2, 'status_name'] = '배달완료'
order_data.loc[order_data['status'] == 3, 'status_name'] = '주문취소'
print(order_data)

output_dir = os.path.join(current_dir, 'output_data')
os.makedirs(output_dir, exist_ok=True) # exist_ok=True 폴더가 이미 존재해도 그냥 넘어감

output_file = os.path.join(output_dir, 'order_data.csv')
order_data.to_csv(output_file, index=False)