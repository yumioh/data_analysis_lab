import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# 데이터 다운로드
data_root = "https://github.com/ageron/data/raw/main/"
print(data_root)
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
x = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

#데이터 그래프로 나타내기 
lifesat.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 52_500, 4, 9])
for i in range(len(lifesat)):
    gdp = lifesat.loc[i, "GDP per capita (USD)"]
    life = lifesat.loc[i, "Life satisfaction"]
    country = lifesat.loc[i, "Country"]
    plt.text(gdp + 100, life, country, fontsize=9)  # +100은 텍스트가 겹치지 않게 오른쪽으로 살짝 이동

plt.axis([23_500, 52_500, 4, 9])
plt.title("Life Satisfaction vs GDP per Capita")
plt.xlabel("GDP per capita (USD)")
plt.ylabel("Life satisfaction")
plt.show()

# 선형 모델 선택 
model = KNeighborsRegressor(n_neighbors=3)

# 모델을 훈련
model.fit(x,y)

#키프로스에 대해 예측 
X_new = [[37_655.2]] # 2020년 1인당 GDP
print(model.predict(X_new)) # [[6.30165767]]