from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import numpy as np
from zlib import crc32

import matplotlib.pyplot as plt



def load_housing_data():
    # 기준 경로 설정 (여기서 모든 작업 수행)
    base_path = Path("handson_ML/datasets")
    base_path.mkdir(parents=True, exist_ok=True)  # 폴더 없으면 생성

    tarball_path = base_path / "housing.tgz"
    extract_path = base_path  # 압축 해제할 위치

    # 파일이 없으면 다운로드 및 압축 해제
    if not tarball_path.is_file():
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        print("Downloading:", url)
        urllib.request.urlretrieve(url, tarball_path)

        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path=extract_path)

    # CSV 경로 지정
    csv_path = extract_path / "housing/housing.csv"
    return pd.read_csv(csv_path)
housing = load_housing_data()

print(housing) 
# 어떤 카테고리가 있고 각 카테고리마다 얼마나 많은 구역이 있는지 확인
print(housing["ocean_proximity"].value_counts())

housing.hist(bins=50, figsize=(12,8))
plt.show()

# 테스트 세트 생성
def shuffle_and_split_data(data, test_ratio):
    shuffle_indices = np.random.permutation(len(data)) # 난수 생성
    test_set_size = int(len(data) * test_ratio) 
    test_indices = shuffle_indices[:test_set_size]
    train_indices = shuffle_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = shuffle_and_split_data(housing, 0.2)
print(len(train_set))
print(len(test_set))

def is_id_in_test_set(identifier, test_ratio):
    # crc32 해시 함수로 적용
    # 그 해시 값이 전체 가능한 해시값(2^32) 중 test ratio 비율 안에 들어오면 테스트 세트에 포함 => 항상 같은 결과 값을 줌
    return crc32(np.int64(identifier)) < test_ratio * 2**32 

def split_data_with_id_hash(data, test_ratio, id_column) :
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_ : is_id_in_test_set(id, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]