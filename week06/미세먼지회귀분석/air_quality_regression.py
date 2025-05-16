import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print('대기오염 데이터와 미세먼지의 연관성 분석하기')
df = pd.read_csv('./week06/대기오염데이터크롤링/대기오염데이터_종로구_20250101_20250412.csv',
                 header=0, encoding='cp949', engine='python')   # 종로구 대기 오염 데이터 불러오기

print(df.shape)   # 데이터프레임 구조 확인(98개 행과 8개 열로 구성)
print(df.isna().sum())   # 데이터 결측치 확인

# 한글 폰트 설정
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

graph = df.drop(['location'], axis=1, inplace=False)   # 측정소 항목 제외
graph.set_index('day', inplace=True)   # 날짜를 인덱스로 설정
graph.plot(kind='line', figsize=(10, 6))   #  선형 그래프 생성

plt.title('대기 오염 항목 그래프')   # 그래프 제목 설정
plt.xlabel('day')  # x축을 날짜로 설정
plt.legend(loc='upper left')  # 범례 위치 설정
# plt.show()  # 그래프 띄우기

data = df.drop(['location', 'day'], axis=1, inplace=False)  # 측정소, 측정일 제외한 대기 오염 측정치 6개 항목 추출
data_scaled = StandardScaler().fit_transform(data.values)   # 각 특성값을 표준화(평균 0, 분산 1)하여 스케일 맞춤
data = pd.DataFrame(data_scaled)  # 데이터프레임 생성
data.columns = ['so2', 'co', 'o3', 'no2', 'pm10', 'pm25']   # 컬럼명 지정

day = df[['day']].copy()    # 원본 데이터에서 날짜 컬럼 복사
graph2 = pd.concat([day, data], axis=1)  # 데이터 합치기
graph2.set_index('day', inplace=True)   # 날짜를 인덱스로 설정
graph2.plot(kind='line', figsize=(16, 8))   #  선형 그래프 생성

plt.title('대기 오염 항목 그래프2')   # 그래프 제목 설정
plt.xlabel('day')  # x축을 날짜로 설정
plt.legend(loc='upper left')  # 범례 위치 설정
plt.rcParams['axes.unicode_minus'] = False  # 음수 부호 깨지지 않게 설정
# plt.show()  # 그래프 띄우기

# X, Y 분할하기
Y_pm10 = df['pm10']  # 미세먼지 분석을 위한 종속변수
Y_pm25 = df['pm25']  # 초미세먼지 분석을 위한 종속변수
X = df.drop(['location', 'day', 'pm10', 'pm25'], axis=1, inplace=False)  # 독립변수 설정

# 훈련 데이터와 평가 데이터 분할하기
X_train, X_test, Y_pm10_train, Y_pm10_test = train_test_split(X, Y_pm10, test_size=0.3, random_state=156)

lr_pm10 = LinearRegression()  #  선형 회귀 분석 모델 생성
lr_pm10.fit(X_train, Y_pm10_train) # 모델 훈련
Y_pm10_predict = lr_pm10.predict(X_test)  # 평가 데이터(독립 변수)를 모델에 넣어서 예상 미세먼지를 계산

# 성능 평가 지표를 사용하여 모델 평가 수행
mse = mean_squared_error(Y_pm10_test, Y_pm10_predict)
rmse = np.sqrt(mse)
print('MSE: {0:.3f}, RMSE : {1: .3f}'.format(mse, rmse))
print('R^2(Variance score) : {0:.3f}'.format(r2_score(Y_pm10_test, Y_pm10_predict)))
print('Y 절편 값: ', np.round(lr_pm10.intercept_, 2))
print('회귀 계수 값: ', np.round(lr_pm10.coef_, 2))

coef = pd.Series(data=np.round(lr_pm10.coef_,2), index=X.columns)  # 회귀 모델에서 구한 회귀 계수 값과 피처 이름을 묶어서 Series 자료형으로 만들기
print(coef.sort_values(ascending=False))  # 회귀 계수 값을 기준으로 내림차순으로 정렬하여 회귀 계수 값이 큰 항목을 확인

fig, axs = plt.subplots(figsize=(10, 10), ncols=2, nrows=2)    # 10x10 크기의 Figure에 2열 2행으로 총 4개의 서브플롯 생성
x_features = ['so2', 'co', 'o3', 'no2']   # 그래프에 표시할 독립 변수 설정

for i, feature in enumerate(x_features):
    row = int(i/2)  # 서브 플롯의 행 위치 계산
    col = i%2       # 서브 플롯의 열 위치 계산
    # 독립 변수와 미세먼지의 관계를 서브플롯에 산점도 + 회귀선 그래프로 그리기
    sns.regplot(x=feature, y='pm10', data=data, ax=axs[row][col])
plt.show()


print('초미세먼지와 대기오염 항목에 대한 회귀분석 모델 구축 및 평가하기')
# 훈련 데이터와 평가 데이터 분할하기
X_train, X_test, Y_pm25_train, Y_pm25_test = train_test_split(X, Y_pm25, test_size=0.3, random_state=156)

lr_pm25 = LinearRegression()  #  선형 회귀 분석 모델 생성
lr_pm25.fit(X_train, Y_pm25_train) # 모델 훈련
Y_pm25_predict = lr_pm25.predict(X_test)  # 평가 데이터(독립 변수)를 모델에 넣어서 예상 초미세먼지를 계산

# 성능 평가 지표를 사용하여 모델 평가 수행
mse = mean_squared_error(Y_pm25_test, Y_pm25_predict)
rmse = np.sqrt(mse)
print('MSE: {0:.3f}, RMSE : {1: .3f}'.format(mse, rmse))
print('R^2(Variance score) : {0:.3f}'.format(r2_score(Y_pm25_test, Y_pm25_predict)))
print('Y 절편 값: ', np.round(lr_pm25.intercept_, 2))
print('회귀 계수 값: ', np.round(lr_pm25.coef_, 2))

coef = pd.Series(data=np.round(lr_pm25.coef_,2), index=X.columns)  # 회귀 모델에서 구한 회귀 계수 값과 피처 이름을 묶어서 Series 자료형으로 만들기
print(coef.sort_values(ascending=False))  # 회귀 계수 값을 기준으로 내림차순으로 정렬하여 회귀 계수 값이 큰 항목을 확인

fig, axs = plt.subplots(figsize=(10, 10), ncols=2, nrows=2)    # 10x10 크기의 Figure에 2열 2행으로 총 4개의 서브플롯 생성
x_features = ['so2', 'co', 'o3', 'no2']   # 그래프에 표시할 독립 변수 설정

for i, feature in enumerate(x_features):
    row = int(i/2)  # 서브 플롯의 행 위치 계산
    col = i%2       # 서브 플롯의 열 위치 계산
    # 독립 변수와 미세먼지의 관계를 서브플롯에 산점도 + 회귀선 그래프로 그리기
    sns.regplot(x=feature, y='pm25', data=data, ax=axs[row][col])
plt.show()

print('by BD 오후 컴퓨터소트프웨어학과 21101849 이지은')
