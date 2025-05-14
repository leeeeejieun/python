import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print('항목에 따른 자동차 연비 예측하기')
# 1. 데이터 준비 및 탐색
df = pd.read_csv('./week06/자동차연비분석/auto-mpg.csv', header=0, engine='python')
df = df.drop(['car_name', 'origin', 'horsepower'], axis=1, inplace=False)  # 분석에 필요없는 컬럼(모델명, 제조국, 출력) 제거


# 2. 선형 회귀 분석 모델 구축
# X(종속 변수), Y(독립 변수) 분할하기
Y = df['mpg']   # 연비 데이터만 저장
X = df.drop(['mpg'], axis=1, inplace=False)  # 연비를 제외한 나머지 데이터 저장


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)    # 훈련 데이터(70%)와 평가 데이터(30%) 분할하기
lr = LinearRegression()  # 선형 회귀 분석 모델 생성
lr.fit(X_train.values, Y_train)        # 모델 훈련
Y_predict = lr.predict(X_test.values)  # 평가 데이터(독립 변수)를 모델에 넣어서 예상 연비를 계산

mse = mean_squared_error(Y_test, Y_predict)   # 평가 지표 MSE를 구함(값이 작을 수록 예측 정확도가 높음)
rmse = np.sqrt(mse)  # MSE에 제곱근을 씌워서 RMSE를 구함 (MSE와 같은 방식으로 계산되지만, 단위가 원래 데이터와 동일해짐)
print('MSE: {0:.3f}, RMSE : {1: .3f}'.format(mse, rmse))
print('R^2(Variance score) : {0:.3f}'.format(r2_score(Y_test, Y_predict)))  # R² Score를 구함(0과 1 사이의 값을 가지며, 1에 가까울 수록 독립 변수가 종속 변수를 얼마나 잘 설명해주는지 보여줌)
print('Y 절편 값: ', np.round(lr.intercept_, 2))
print('회귀 계수 값: ', np.round(lr.coef_, 2))

# 3. 평가 지표를 통해 선형 회귀 분석 모델을 평가하고 회귀 계수를 확인하여 자동차 연비에 끼치는 피치의 영향을 분석하기
coef = pd.Series(data=np.round(lr.coef_,2), index=X.columns)  # 회귀 모델에서 구한 회귀 계수 값과 피처 이름을 묶어서 Series 자료형으로 만들기
print(coef.sort_values(ascending=False))  # 회귀 계수 값을 기준으로 내림차순으로 정렬하여 회귀 계수 값이 큰 항목을 확인

# 4. 산점도 + 선형 회귀 그래프로 독립 변수가 회귀 분석에 미치는 영향을 시각화
fig, axs = plt.subplots(figsize=(16, 16), ncols=3, nrows=2)    # 16x16 크기의 Figure에 3열 2행으로 총 6개의 서브플롯 생성
x_features = ['model_year', 'acceleration', 'displacement', 'weight', 'cylinders']   # 그래프에 표시할 독립 변수 설정
plot_color = ['r', 'b', 'y', 'g', 'r']   # 각 독립 변수의 컬러 지정

for i, feature in enumerate(x_features):
    row = int(i/3)  # 서브 플롯의 행 위치 계산
    col = i%3       # 서브 플롯의 열 위치 계산

    # 독립 변수와 연비의 관계를 서브플롯에 산점도 + 회귀선 그래프로 그리기
    sns.regplot(x = feature, y = 'mpg', data = df, ax = axs[row][col], color = plot_color[i])
plt.show()  # 그래프 표시

# 5. 임의의 데이터를 입력하여 연비 예측 
print('5. 임의의 데이터를 입력하여 연비 예측 ')
print('----------------------------------')
print('연비를 예측하고 싶은 차의 정보를 입력해주세요.')

cylinders = int(input('cylinders : '))        # 실린더 수 입력
displacement = int(input('displacement : '))  #  배기량 입력
weight = int(input('weight : '))              #  자동차 무게 입력
acceleration = int(input('acceleration : '))  # 가속능력 입력
model_year = int(input('model_year : '))      # 출시년도 입력

mpg_predict = lr.predict([[cylinders, displacement, weight, acceleration, model_year]])
print('이 자동차의 예상 연비(MPG)는 %.2f입니다.' %mpg_predict[0])  # 결과가 배열 형태로 반환되기 때문에 첫 번째 값을 추출
print('by BD 오후 컴퓨터소트프웨어학과 21101849 이지은') 