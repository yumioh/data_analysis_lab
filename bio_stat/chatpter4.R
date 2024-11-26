data1 <- read.csv("./bio_stat/data/biostat_ex_data.csv")
install.packages("dplyr")
install.packages("ggplot2")
library("dplyr")

# mutate_at() : 특정 열에만 변환 작업 적용 mutate_at(.vars, funds)
# vars : 변환하고자 하는 변수들 .funs: 적용할 함수
# as.factor : 범주형 변수로 변환 => 회귀분석, 카이제곱 검정
data <- data1 %>% mutate_at(vars(sex, Recur, stage, smoking, obesity, 
                            Recur_1y, post.CA19.9.binary, post.CA19.9.3grp),
                            as.factor)
library("ggplot2")


ggplot(data)+geom_histogram(aes(x=weight), color="black",fill = "skyblue", breaks=seq(40,100,10))

#이표본 이분산 : 두 그룹간 평균을 비교할때 사용
# var.equal = FALSE 디폴트 값이 FALSE = > 이표본 이분산
# var.equal = TRUE => 이표본 등분산

#남녀간 몸무게 평균은 다르다 
# 두그룹 : sex / 비교할 평균값 : weight
t.test(weight~sex, data=data)


#월콕순 순위합 검정 : 비모수적 방법
# wilcox.test 

# 몸무게의 분포의 중심 위치가 남녀 간에 다른지 윌콕슨 순위함 검정 수행
wilcox.test(weight~sex, data=data)

#3개 이상 그룹의 비교

#모수적 방법 : ANOVA F-검정과 웰치 F-검정

#ANOVA F-검정 : 각 그룹의 분산이 같다라고 가정함 => 이표본 등분산 t검정을 3개 이상 그룹에 대한 검정으로 확장

#췌장암 환자데이터에서 병기에 따라 CEA의 분포에 차이가 확인
#로그를 취하여 분포의 대칭성을 회복하고, 정규분포에 가까워짐 => ANOVA F-검정실시
ggplot(data) + geom_histogram(aes(x=log(CEA)), color="black", fill="skyblue")

#ANOVA F검정 : AOV()
# aov()함수는 결과물을 다시 summary()함수에 넣어야 유의확률이 출력
fit <- aov(log(CEA)~stage, data=data)
summary(fit)

#웰치스 F-검정
oneway.test(log(CEA)~stage, data=data)

#크루스칼-왈리스 검정 : kruskal.test() 
# CEA 중심위치가 같은지 여부 확인
kruskal.test(CEA~stage, data=data)

#취장암 데이터에서 변수 CEA는 수술 직전에 측정한 CEA값이고, post.CEA는 수술직후 측정한 CEA값이다
#수술 전후에 CEA값에 변화가 있는지 검정
# 정규분포 그래프 확인 => 검정가능
ggplot(data) + geom_histogram(aes(x=log(post.CEA)-log(CEA), color="black", fill="skyblue"))

#대응표본 t-검정 : t.test()
#짝지어진 표본 : paired = TRUE, 독립이표본 : FALSE
#log(data$post.CEA)가 log(data$CEA)값보다 작다
#두 데이터간 로그 변환된 값의 평균 차이가 -0.57~0.27 사이일 것이라고 95%신뢰수준에서 추정
# 자유도는 데이터 쌍의 개수 : n=156 df=n-1 = 155개
# t값 두데이터간의 차이가 얼마나 유의미한지 나타냄 => 0에서 멀어질수록 데이터 간 평균 차이가 유의미하다 것을 의미
t.test(log(data$post.CEA),log(data$CEA), paired=TRUE)

#월콕슨 부호순위 검정
#대립가설이 0이 아닌 기준으로  대칭인 경우 two.sided
#0보다 큰 수를 기준으로 좌우 대칭인 경우 greater
#0보다 작은 수를 기준으로 좌우 대칭인 경우 less
#paired : 짝지어진 표본 TRUE, 독립 이표본이면 FALSE => 디폴트는 FALSE


#관련된 두 데이터 간의 중앙값의 차이가 0인 여부 검정
#데이터를 순위화하고, 부호를 고려하여 순위합을 계산
# 검정통계량(v) :순위의 합 
wilcox.test(log(data$post.CEA),log(data$CEA), paired=TRUE)

#윌콕슨 부호순위 검정 : 비모수적 방법으로 데이터가 정규성을 따르지 않아도 사용할 수 있다
# 대응표본 t-검정 : 데이터가 정규성을 만족한다고 가정하면, 평균의 차이를 검정 가능

