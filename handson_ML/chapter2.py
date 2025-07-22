from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import numpy as np
from zlib import crc32

from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor

from sklearn.preprocessing import FunctionTransformer

from sklearn.impute import SimpleImputer

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
    

housing = strat_train_set.copy()

# 지리적 데이터 시각화하기
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
plt.xlabel("경도")
plt.ylabel("위도")
#plt.show()

# 주택가격 알아보기 
# cmap : 색상 맵: 파란색(낮음) → 빨간색(높음)으로 색 표시
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, 
             s=housing["population"]/100, label="인구",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, figsize=(10,7))
cax = plt.gcf().get_axes()[1]
cax.set_ylabel("중간 주택 가격")
plt.xlabel("경도")
plt.ylabel("위도")
#plt.show()

# 상관관계 조사 1 : 데이터셋이 크지 않아 모든 특성 간의 피어슨 상관계수 계산
# corr(numeric_only=True) : 모든 숫자형 열끼리의 상관계쑤
corr_matrix = housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False) # 내림차순으로 정리

# 상관관계 조사 2 :  scatter_matrix 함수를 사용

atrributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]

# 산점도 그래프 그리기 
scatter_matrix(housing[atrributes], figsize=(12,8))
#plt.show()

# 그래프를 보면 median_house_value 값이 가장 유의미한 변수로 보임
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1, grid=True)
plt.xlabel("중간 소득")
plt.ylabel("중간 주택 가격")
#plt.show()

# 특정 조합으로 실험하기 
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_house"] = housing["population"] / housing["households"]

corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# 입력데이터와 예측값에 같은 변형을 적용하지 않기 위해 예측변수와 레이블 분리 
housing = strat_train_set.drop("median_house_value", axis=1) #열 제거 => 예측변수만 남김 
housing_lables = strat_train_set["median_house_value"].copy() # 열만 따로 복사하여 저장 => 출력데이터만 들어가 있음

# 데이터 정제 
# 1. 해당 구역을 제거
# 2. 전체 특성을 삭제
# 3. 누락된 값을 대체값으로 채움

# # 방법 1
# housing.dropna(subset=["total_bedrooms"], inplace=True)

# # 방법 2
# housing.drop("total_bedrooms", axis=1, inplace=True)

# 방법 3
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True) # 결측값이 존재하면, mdeian 값으로 대체 

# 여러 열에 대해 한꺼번에 결측치 채우기 
imputer = SimpleImputer(strategy="median")

housing_num = housing.select_dtypes(include=[np.number]) # 숫자형 열만 선택ㅇ하여 새로운 데이터 프레임 생성
imputer.fit(housing_num)

# print(imputer.statistics_)
# print(housing_num.median().values)

X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(8))

ordinal_encoder = OrdinalEncoder()
housing_cat_encold = ordinal_encoder.fit_transform(housing_cat)

print(ordinal_encoder.categories_)


cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# get_dummies() : 변주형 데이터를 숫자형 더미로 변환
df_test = pd.DataFrame({"ocean_proximity" : ["INLAND", "NEAR BAY"]})
pd.get_dummies(df_test)

# 특성 스케일과 변환
min_max_scaler = MinMaxScaler(feature_range=(-1,1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)
print(housing_num_min_max_scaled)

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)
print(housing_num_std_scaled)

# 주택 연령이 35와 유사도 계산
# 거리가 가까울 수록 유사도는 1에 가깝고 멀어질수도록 0에 가까움
# gamma=0.1 : 유사도에 영향을 주는 파라미터 (클수록 급격히 감소)
age_simli_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)

# 레이블 스케일링 후 간단한 선형회귀 모델을 훈련하고 새로운 데이터에서 예측 
# 변환기 메서드를 사용해 원본 스케일로 되돌림

target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_lables.to_frame())
model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)
some_new_data = housing[["median_income"]].iloc[:5] # 5개 데이터만 들고 옴
scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)

# 더 간단한 방법
# 출력을 변환한 상태로 회귀 모델을 훈련하고 예측하는 예제
# 회귀모델을 만들고 타깃값에만 스케이링 변환를 적용뒤 예측 결과를 다시 원래 단위로 되돌려주는 Regressor를 사용

model = TransformedTargetRegressor(LinearRegression(), 
                                   transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_lables)
predictions = model.predict(some_new_data)
#print(predictions)

# 로그 변환과 역변환을 자동으로 할 수 있도록 설정 한 후, 데이터에 로그 변환을 적용 
log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])

# 주택 연령(housing_median_age) 데이터에 대해, "35와 얼마나 비슷한가?"를 유사도로 변환하는 자동 변환기를 만든 것
# Y: 비교기준, GAMMAR : 감쇠값
rbf_transformer = FunctionTransformer(rbf_kernel, 
                                      kw_args=dict(Y=[[35,]], gamma=0.1))
# RBF 유사도 진행 : 각 데이터가 35와 얼마나 유사한지 0~1 사이로 표현한 값
age_simli_35 = rbf_transformer.transform(housing[["housing_median_age"]])

sf_coords = 37.7749, -122.41
sf_transformer = FunctionTransformer(rbf_kernel, 
                                     kw_args=dict(Y=[sf_coords], gamma=0.1))
sf_simil = sf_transformer.transform(housing[["latitude","longitude"]])
print(sf_simil)

# 첫번째 특성과 두번째 특성 사이의 비율 계산
ratio_transformer = FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]])
ratio_transformer.transform(np.array([[1., 2], [3.,4.]]))

# 사이킷런은 덕 타이핑에 의존하기 때문에 이 클래스가 특정 클래스를 상속할 필요가 없음
# 정해진 클래스에 속하지 않아도 필요한 기능만 있으면 쓸수 있다 

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

# StandardScsaler를 흉내 낸 사용자 정의 변환기 클래스
class StandardScalerClone(BaseEstimator, TransformerMixin) :
    def __init__(self, with_mean=True): # 평균을 뺴는 동작을 할지 말지 결정
        self.with_mean = with_mean
    
    def fit(self, X, y=None) :
        X = check_array(X) # 입력이 올바른 배열인지 검사
        self.mean = X.mean(axis=0) # 각 특성의 평균을 계산하여 저장
        self.n_features_in_ = X.shape[1] # 입력 데이터의 특성 개수를 기록. scikit-learn에서는 변환기나 모델이 훈련된 후 몇 개 특성으로 훈련됐는지 확인할 때 이 속성을 사용
        return self
    
    def transform(self, X) :
        check_is_fitted(self) 
        X = check_array(X) # X가 적절한 형태의 배열인지 확인하고 변환
        assert self.n_features_in_ == X.shape[1] # 훈련 데이터와 같은 특성 수인지 확인
        # assert 조건이 참인지 확인
        if self.with_mean: 
            X = X - self.mean # 각 특성에서 평균을 빼는 작업(중심화)
            return X/self.scale_ # 표준화
        
from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator, TransformerMixin) :
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None): # 초기화
        self.n_clusters = n_clusters 
        self.gamma = gamma
        self.random_state = random_state
        
    def fit(self, X, y=None, sample_weight=None) : # KMeans를 데이터 X에 학습시켜서 클러스터 중심을 찾음
        self.kmeans_= KMeans(self.n_clusters, random_state=self.random_state) # sample_weight가 있으면 가중치를 적용 가능
        self.kmeans_.fit(X, sample_weight=sample_weight) #학습된 KMeans 객체가 저장
        return self
    
    def transform(self, X) :
        # 데이터 X를 클러스터 중심과 비교하여 RBF 커널 유사도를 계산
        # 각 클러스터와의 유사도 벡터
        # 이 값들을 새로운 특성으로 사용 가능
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None) :
        return[f"클러스터 {i} 유사도" for i in range(self.n_clusters)]
    

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1, random_state=42)
similarties = cluster_simil.fit_transform(housing[["latitude", "longitude"]],
                                          sample_weight = housing_lables)

#print(similarties[:3].round(2))

# 변환 파이프라인 

from sklearn.pipeline import Pipeline

# pipeline을 이용해서 숫자형 데이터 전처리 단계를 체인처럼 연결
# 수치형 데이터에 대해 결측값 처리 + 표준화를 순서대로 자동으로 처리하는 파이프라인

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")), # 결측값을 특성별 중앙값으로 채움
    ("standardize", StandardScaler()), # 데이터 스케일
])

from sklearn.pipeline import make_pipeline

num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

housing_num_prepared = num_pipeline.fit_transform(housing_num) #결측치를 채우고, 스케일링 실행
#print(housing_num_prepared[:2].round(2))

# 결과를 데이터프레임으로 재구성
df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, columns=num_pipeline.get_feature_names_out(), 
    index=housing_num.index)
print(df_housing_num_prepared)

from sklearn.compose import ColumnTransformer

# 수치형 특성과 범주형 특성 다른 전처리기로 처리 
# SimpleImputer : 결측값을 가장 많이 등장하는 값으로 채움
num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households",  "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore") # 훈련에 없는 값이 들어오면 무시 
)

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object))
)

housing_prepared = preprocessing.fit_transform(housing)


# 기존 데이터에 비율 계산, 로그 변환, 클러스터 유사도, 범주형 인코딩, 기본 스케일링을 조합하여 
#  머신러닝 모델에 최적화된 특성으로 변환하기 위한 자동화된 전처리기

def column_ratio(X) :
    return X[:, [0]] / X[:, [1]] # 두열의 비율 계산

def ratio_name(function_transformer, feature_name_in) :
    return ["ratio"]

def ratio_pipeline() : # 스케일링 전처리
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler()
    )
    
log_pipeline = make_pipeline( # 로그 변환
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler()
)

# RBF 기반 클러스터 유사도 변환기
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1, random_state=42)

# 기본 수치형 데이터용 파이프라인
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())

preprocessing = ColumnTransformer([
    ("bedroms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]), # 비율변환
    ("rooms_per_house", ratio_pipeline(), ["total_rooms","households"]), # 비율변환
    ("people_per_house", ratio_pipeline(), ["population", "households"]), # 비율변환
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",  # 로그변환
                           "households", "median_income"]), 
    ("geo", cluster_simil, ["latitude","longitude"]), # 클라스터 유사도
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)), # 범주형 인코딩
    ],
    remainder=default_num_pipeline) # 기본 수치형 변환

# 모델 선택과 훈련

from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_lables)

housing_predictions = lin_reg.predict(housing)
print(housing_predictions[:5].round(-2))
print(housing_lables.iloc[:5].values)

from sklearn.metrics import root_mean_squared_error

lin_rmse = root_mean_squared_error(housing_lables, housing_predictions)

from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_lables)

housing_predictions = tree_reg.predict(housing)

tree_rmse = root_mean_squared_error(housing_lables, housing_predictions)
print(tree_rmse)

# 교차검증으로 평가하기 

from sklearn.model_selection import cross_val_score

# tree_reg : 훈련시킬 결정 트리 회귀 모델
# housing : 입력데이터
# housing_labels : 타깃값
# scoring="neg_root_mean_squared_error" : 평가지표(평균 제곱근 오차)
# 10겹 교차 검증
tree_rmses = -cross_val_score(tree_reg, housing, housing_lables, scoring="neg_root_mean_squared_error", cv=10)

print(pd.Series(tree_rmses).describe())


from sklearn.ensemble import RandomForestRegressor

forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
forest_rmses = -cross_val_score(tree_reg, housing, housing_lables, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(tree_rmses).describe())

# 모델 미세 튜닝

# 그리드 서치 
from sklearn.model_selection import GridSearchCV

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])

param_grid = [
    {'preprocessing__geo__n_clusters' : [5,8,10],
     'random_forest__max_features' :[4,6,8]},
    { 'preprocessing__geo__n_clusters' : [10,15],
     'random_forest__max_features' :[6,6,10]}
]

grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
                           scoring="neg_root_mean_squared_error")
grid_search.fit(housing, housing_lables)

cv_res = pd.DataFrame(grid_search.cv_results_)
print(cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True))
print(cv_res.head())

# 랜덤서치 
# 하이퍼파라미터 값이 연속적이면 랜던 서치를 1000번 실행했을때 각 하이퍼파라미터마다 1000개 다른 값을 탐색. 반면 그리드 서치는 하이퍼파라미터에 대해 나열한 몇 개의 값을 탐색

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# full_pipelines이라는 파이프라인의 하이퍼파라미터를 튜닝. housing 데이터를 기반으로 housing_labels을 예측

# radint는 scipy.stats의 함수로 랜덤하게 정수를 샘플링
# preprocessing__geo__n_clusters : geo 단계의 클라수터 수 
# random_forest__max_features : 각 노드에서 고려할 특성의 수 
param_distribs = {'preprocessing__geo__n_clusters' : randint(low=3, high=50),
                  'random_forest__max_features' : randint(low=2, high=20)}

# 전치리, 예측 모델이 합쳐진 파이프라인 객체 
rnd_search = RandomizedSearchCV(
    full_pipeline, param_distributions=param_distribs, #
    n_iter=10, #10개의 다른 조합을 랜덤
    cv=3, # 3겹 교차 검증 수행
    scoring='neg_root_mean_squared_error', #음수 RMSE 사용 (클수록 좋음)
    random_state=42) # 재현성을 위한 난수 시드 고정

rnd_search.fit(housing, housing_lables)
print(rnd_search.best_params_)

# 앙상블 방법

final_model = rnd_search.best_estimator_ # 전처리 포함
feature_importances = final_model["random_forest"].feature_importances_
print(feature_importances.round(2))

sorted(zip(feature_importances,
           final_model["preprocessing"].get_feature_names_out()),
       reverse=True)

# 테스트 세트로 시스템 평가하기 

# 테스트 데이터셋에 대한 예측 결과를 기반으로 RMSE를 계산하는 부분
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)

final_rmse = root_mean_squared_error(y_test, final_predictions)
print(final_rmse)

from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y.test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors)-1,
                         loc=squared_errors.mean(),
                         scale= stats.sem(squared_errors)))

import joblib
# 모델 저장
joblib.dump(final_model, "my_california_housing_model.pkl")
