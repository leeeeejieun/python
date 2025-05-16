import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print('------------ 시간, 날씨 기반 자전거 대여량 예측하기 ------------')

# 한글 폰트 설정
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
data = pd.read_csv('./week06/자전거대여량예측/bike_data.csv', 
                   header=0, encoding='cp949', engine='python') # 자전거 대여량 데이터 불러오기
data.drop(['instant', 'dteday', 'casual', 'registered', 'workingday', 'atemp'],
           axis=1, inplace=True)  # 분석에 필요없는 컬럼 제거
print(data.info())      # 데이터프레임 기본 정보 확인
print(data.describe())  # 데이터프레임 기본 통계량 확인

# ----  날씨 상태에 따른 자전거 대여수 시각화 ----  
weather_cnt = data.groupby('weathersit')['cnt'].mean().reset_index()   # 날씨 상태별 평균 자전거 대여 수
plt.figure(figsize=(10, 6))   # 그래프 크기 설정
plt.bar(weather_cnt['weathersit'], weather_cnt['cnt'], color='skyblue')  # 막대그래프로 표현
plt.xlabel('날씨 상태')           # x축 라벨 설정
plt.ylabel('평균 자전거 대여 수')  # y축 라벨 설정
plt.title('날씨 상태별 평균 자전거 대여 수')  # 그래프 제목 설정
plt.xticks(ticks=[1,2,3,4], labels=['맑음/부분적 구름', '안개+흐림', '가벼운 비/눈', '폭우/눈/안개'])  # x축 눈금명 설정
plt.show() 

# ----  시간대별 자전거 대여수 변화 시각화 ---- 
hourly_cnt = data.groupby('hr')['cnt'].mean().reset_index()  # 시간대별 평균 대여수 계산
plt.figure(figsize=(12, 6))  # 그래프 크기 설정
plt.plot(hourly_cnt['hr'], hourly_cnt['cnt'], marker='o', color='orange')  # 선 그래프로 표현
plt.xlabel('시간(시)')     # x축 라벨 설정
plt.ylabel('평균 대여수')   # y축 라벨 설정
plt.title('시간대별 자전거 평균 대여수')  # x축 라벨 설정
plt.xticks(range(0, 24))  # x축 눈금: 0시~23시
plt.show()

# ----  선형 회귀 분석 모델링 ---- 
# X, Y 분할하기
Y = data['cnt']  # 대여량을 종속 변수로 설정
X = data.drop(['cnt'], axis=1, inplace=False)  # 대여량을 제외한 모든 항목을 독립 변수로 설정

# 훈련 데이터와 평가 데이터 분할하기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=10)

# 독립변수를 표준화하여 모델이 특정 변수에 치우치지 않도록 함
scaler = StandardScaler()  # StandardScaler 객체를 생성
X_train_scaled = scaler.fit_transform(X_train)   # 훈련 데이터에 대해 표준화를 학습(fit)하고, 변환(transform)까지 동시에 수행
X_test_scaled = scaler.transform(X_test)    # 테스트 데이터는 훈련 데이터의 기준(평균과 표준편차)으로만 변환 (fit은 하지 않음)

lr = LinearRegression()  # 선형 회귀 분석 모델 생성
lr.fit(X_train_scaled, Y_train)         # 모델 학습
Y_predict = lr.predict(X_test_scaled)   # 평가 데이터를 사용해 예측값 계산

# 성능 평가 지표를 사용하여 모델 평가 수행
print('자전거 대여량 예측을 위한 선형 회귀 분석 모델 구축 및 평가하기') 
mse = mean_squared_error(Y_test, Y_predict)
rmse = np.sqrt(mse)
print('MSE: {0:.3f}, RMSE : {1: .3f}'.format(mse, rmse))
print('R^2(Variance score) : {0:.3f}'.format(r2_score(Y_test, Y_predict)))
print('Y 절편 값: ', np.round(lr.intercept_, 2))
print('회귀 계수 값: ', np.round(lr.coef_, 2))

coef = pd.Series(data=np.round(lr.coef_,2), index=X.columns)  # 회귀 모델에서 구한 회귀 계수 값과 피처 이름을 묶어서 Series 자료형으로 만들기
print(coef.sort_values(ascending=False))  # 회귀 계수 값을 기준으로 내림차순으로 정렬하여 회귀 계수 값이 큰 항목을 확인

fig, axs = plt.subplots(figsize=(16, 20), ncols=2, nrows=5)    # 10x10 크기의 Figure에 2열 5행으로 총 10개의 서브플롯 생성
x_features = features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'weathersit', 'temp', 'hum', 'windspeed']   # 그래프에 표시할 독립 변수 설정

# 모델 평가 결과 시각화
for i, feature in enumerate(x_features):
    row = i // 2  # 서브 플롯의 행 위치 계산
    col = i%2       # 서브 플롯의 열 위치 계산
    # 독립 변수와 대여량의 관계를 서브플롯에 산점도 + 회귀선 그래프로 그리기
    sns.regplot(x=feature, y='cnt', data=data, ax=axs[row][col])
plt.subplots_adjust(hspace=0.6, wspace=0.3)
plt.show()

print('----------------------------------')
print('자전거 대여량을 예측하고 싶은 날씨 및 시간 정보를 입력해주세요.')
season = int(input('계절 (1:봄, 2:여름, 3:가을, 4:겨울): '))
yr = int(input('연도 (0: 2011, 1: 2012): '))
mnth = int(input('월 (1~12): '))
hr = int(input('시간 (0~23): '))
holiday = int(input('공휴일 여부 (0: 아니오, 1: 예): '))
weekday = int(input('요일 (0: 일요일 ~ 6: 토요일): '))
weathersit = int(input('날씨 상황 (1~4): '))
temp = float(input('기온 (0~1, 정규화된 값): '))
hum = float(input('습도 (0~1): '))
windspeed = float(input('풍속 (0~1): '))

cnt_predict = lr.predict([[season, yr, mnth, hr, holiday, weekday,
                          weathersit, temp, hum, windspeed]])
print('예상 자전거 대여량은 %d입니다.' %cnt_predict[0])
print('by BD 오후 컴퓨터소트프웨어학과 21101849 이지은') 