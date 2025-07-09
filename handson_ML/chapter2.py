from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import numpy as np
from zlib import crc32

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

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
#plt.show()

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
    in_test_set = ids.apply(lambda id_ : is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# index행 사용
housing_with_id = housing.reset_index()
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")

#print(housing_with_id.head())

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# 카테고리 특성 만들기 
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1,2,3,4,5])

housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("소득 카테고리")
plt.ylabel("구역 개수")
#plt.show()

# 한 데이터셋으로 각각 다른 10개의 계층 분할
splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

strat_splits = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

# 첫번째 분할
print(strat_splits[0])
strat_train_set, strat_test_set = strat_splits[0]
#print(strat_train_set, strat_test_set)

# 하나의 분할이 필요한 경우 train_test_split() 함수와 stratify 매개변수 사용
# stratify : 비율이 train/test 세트 모두에서 원본과 똑같이 유지되도록 분할
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42
)

# 소득구간의 분포 알아보기
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

# income 특성을 다시 사용하지 않으므로 열 삭제
# axis=1 : 열
for set_ in (strat_train_set, strat_test_set) :
    set_.drop("income_cat", axis=1, inplace=True)