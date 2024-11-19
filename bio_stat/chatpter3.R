#바이오 관련 데이터 불려오기 
data1 <- read.csv("./bio_stat/data/biostat_ex_data.csv")

#table 함수는 각 값이 몇 번 나왔는지를 계산
table(data1$sex)


#정규근사 방식(prop.test) : 표본의 크기가 충분히 크다면 이항분포가 정규 분포로 근사 될 수 있다
#다만 표본의 크기가 작거나 관측된 비율이 극단적일때 근사가 부정확해 질 수있음

#Exact 방식(binom.test) : 표본의 크기가 작거나 비율이 극단적일때 정확한 확률을 계산하기 위해 이항 분포를 그대로 사용하는 방식
# 다만 큰 표본인 경우 계산이 복잡해져 비효율적임

#정규근사 방식
#남성 비율의 점추정(비율의 추정값)과 구간추정(신뢰 구간)을 제공
#sum(data1$sex==1) : 남성 수, nrow(data1) : 전체 데이터의 행 수 
prop.test(sum(data1$sex==1), nrow(data1))

#Exact 이항분포 방식 
#남성 비율에 대한 추정값과 신뢰구간을 구함
# 이항 분포를 기반으로 계산하며, 특히 표본 크기가 작거나 비율이 극단적인 경우 유용
binom.test(sum(data1$sex==1), nrow(data1))

#3. 가설검정 : 모집단으로 부터 추출한 표본 데이터를 이용하여 모수에 대한 가설에 대해 결론을 도출하는 것

#귀무가설 vs 대립가설
# 귀무가설(H0) : 이미 받아 들이고 있는 기존의 통념
# 대립가설(H1) : 귀무가설과 반대로 기존의 통념과 다른 새로운 주장 
# 1종 오류  : 귀무가설이 참인데 귀무가설을 기각하는 경우
# 2종 오류  : 대립가설이 참이나 귀무가설을 기각하지 못하는 경우 
# 1종 오류의 최대 한계를 유의수준
# 검정력 : 대립가설이 참일때 올바른 결론을 내릴 확률 (1-b)

# 유의확률 : 데이터와 귀무가설 간의 거리를 재는 측도 
# 유의 확률이 작다는 것은 귀무가설이 참일때 지금과 같은 결과가 나올 확률
# 유의확률이 유의 수준보다 작다면 귀무가설 기각

# 3.1 모평균 가설검정
# 일표본 t-검정 :  모평균 u가 u0와 같다고 할 수있는지 검정 

# mu : 평균값
# 대립가설 방향  alternative = "two.sided"(양측)  alternative = "greater" (u>u0)  alternative = "less" (u < u0)
t.test(data1$weight, mu=68, alternative = "greater")

#3.2 모비율 가설검정 : 모집단의 어떤 비율 p을 특정값 p0와 비굑하는 가설검정
# prop.test(성공횟수, 시행횟수, 검정비율)
prop.test(sum(data1$sex==1), nrow(data1), p=0.6)
#Exact 방법으로 검정 : 
binom.test(sum(data1$sex==1), nrow(data1), p=0.6)
binom.test(sum(data1$sex==1), nrow(data1), p=0.6, alternative = "greater")