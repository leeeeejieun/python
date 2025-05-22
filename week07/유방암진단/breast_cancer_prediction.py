import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer   # 사이킷런에서 제공하는 유방암 진단 데이터셋 불러오기
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

print('! [로지스틱 회귀 분석] 특정 데이터로 유방암 진단하기')
b_cancer  = load_breast_cancer()   # 데이터셋을 로드하여 객체 생성
print(b_cancer.DESCR)    # 데아터셋에 대한 설명 확인

# ------- 데이터 탐색 -------
b_cancer_df = pd.DataFrame(b_cancer.data, columns=b_cancer.feature_names)  # 독립 변수가 되는 피처를 데이터프레임으로 변환
b_cancer_df['diagnosis'] = b_cancer.target     # 유방암 악성/양성 여부를 나타내는 종속 변수 추가
print(b_cancer_df.head())  # 데이터 확인

print('유방암 진단 데이터셋 크ㅈ기: ', b_cancer_df.shape)  # 데이터셋 행/열 개수 확인
print(b_cancer_df.info())   # 데이터 기본 정보 확인

# ------- 로지스틱 회귀 분석 모델링 -------
scaler = StandardScaler()  # StandardScaler 객체를 생성
b_cancer_scaled = scaler.fit_transform(b_cancer.data)  # 독립 변수를 평균 0, 분산 1인 표준 정규 분포 형태로 표준화
print(b_cancer.data[0])   # 표준화 적용 전 데이터 확인
print(b_cancer_scaled[0])   # 표준화 적용 후 데이터 확인

# X, Y 설정하기
Y = b_cancer_df['diagnosis']   # 유방암 악성/양성 여부를 나타내는 종속 변수
X = b_cancer_scaled   # 표준화가 적용 된 독립 변수

# 훈련 데이터와 평가 데이터 분할하기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=10)  # 샘플 569개를 7(학습) : 3(평가)로 분할

lr_b_cancer = LogisticRegression()  # 로지스틱 회귀 분석 모델 생성
lr_b_cancer.fit(X_train, Y_train)   # 모델 훈련
Y_predict = lr_b_cancer.predict(X_test)  # 평가 데이터를 사용해 예측값 계산

confusion_matrix(Y_test, Y_predict) # 오차 행렬
accuracy = accuracy_score(Y_test, Y_predict)   # 정확도 
precision = precision_score(Y_test, Y_predict) # 정밀도 
recall = recall_score(Y_test, Y_predict)  # 재현율 
f1 = f1_score(Y_test, Y_predict)  # f1 score 
roc_auc = roc_auc_score(Y_test, Y_predict)   # ROC-AUC score

print('정확도 {0: .3f}, 정밀도: {1: .3f}, 재현율: {2: .3f}, F1: : {3: .3f}'.format(accuracy, precision, recall, f1))
print('ROC_AUC: : {0: .3f}'.format(roc_auc))
print('by BD 오후 컴퓨터소트프웨어학과 21101849 이지은') 
