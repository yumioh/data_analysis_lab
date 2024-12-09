import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
acd_api_key = os.getenv('MONGO_DB')

csi_file = './mongoDB/data/csi_accident_cases_240915.xlsx'

db = MongoClient('mongodb://192.168.150.79:27017/')

#디비 생성 => 데이터를 삽인 전까지 데이터베이스나 컬렉션 생성이 안됨 
# 다만 명시적으로 컬렉션 생성 가능 (create_collection)
csi_collection = db['accident_data']

#테이블 생성
csi_document = csi_collection['csi_accidents']

#파일 읽기 
read_raw_data = pd.read_excel(csi_file)

#json 행태로 변환
convert_raw_data = read_raw_data.to_json(force_ascii=False, orient = 'records', indent=4)

#json 문자열을 python 객체로 변환
#mongoDB는 json문자열을 직접 이해하지 못함 python 딕셔너리와 같은 객체를 사용하여 데이터 전달 해야함
json_raw_data = json.loads(convert_raw_data)

if isinstance(json_raw_data, list):
    csi_document.insert_many(json_raw_data) # 배열형태의 다수의 데이터를 삽입
else:
    csi_document.insert_one(json_raw_data) # 단일 document 

db.close()
sys.exit()
