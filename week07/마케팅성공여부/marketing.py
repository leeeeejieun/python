import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# 마케팅 데이터 파일 불러오기
print('----- 마케팅 성공 여부 판단 -----') 
df = pd.read_csv('./week07/마케팅성공여부/marketing_data.csv',
                 sep=';', header=0, engine='python')

# 작업 확인용 출력
print(df.head())
print(df.info())
print(df['y'].value_counts())
# ------ 범주형 데이터를 수치형 데이터로 변환하기 ------
binary_cols = ['default', 'housing', 'loan', 'y']  # yes/no 값을 가지는 컬럼 리스트
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})  # 데이터를 0과 1로 변환

# month 컬럼 숫자로 변환
months = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
             'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
df['month'] = df['month'].map(months)

# job 컬럼은 범주가 많으므로 Frequency Encoding 적용
job_freq = df['job'].value_counts(normalize=True)   # 각 컬럼 데이터의 출현 빈도 계산
df['job_freq'] = df['job'].map(job_freq)  # 새로운 컬럼에 빈도 저장
df = df.drop(columns=['job'])  # 기존 컬럼 삭제

# 범주가 적은 컬럼에는 one-hot encoding 적용
oh_cols = ['marital', 'education', 'contact', 'poutcome']  # 결혼 여부, 교육 수준, 연락 방식, 이전 캠페인 결과
df = pd.get_dummies(df, columns=oh_cols, drop_first=True, dtype=int)

# 작업 확인용 출력
print(df.head())
print(df.info())

# ------ 로지스틱 회귀 분석 모델링 ------
Y = df['y']    # 마케팅 성공 여부를 나타내는 종속 변수
X = df.drop(['y'], axis=1)  # 독립 변수
print('표준화 적용 후 데이터 확인')
print(X.values[0]) 

scaler = StandardScaler()    # StandardScaler 객체를 생성
X = scaler.fit_transform(X)  # 평균 0, 분산 1인 표준 정규 분포 형태로 표준화
print('표준화 적용 후 데이터 확인')
print(X[0])   

# 훈련 데이터와 평가 데이터 분할하기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=10)  

lr = LogisticRegression()  # 로지스틱 회귀 분석 모델 생성
lr.fit(X_train, Y_train)     # 모델 훈련
Y_predict = lr.predict(X_test)  # 평가 데이터를 사용해 예측값 계산

accuracy = accuracy_score(Y_test, Y_predict)   # 정확도 
precision = precision_score(Y_test, Y_predict) # 정밀도 
recall = recall_score(Y_test, Y_predict)  # 재현율 
f1 = f1_score(Y_test, Y_predict)  # f1 score 
roc_auc = roc_auc_score(Y_test, Y_predict)   # ROC-AUC score

print('정확도 {0: .3f}, 정밀도: {1: .3f}, 재현율: {2: .3f}, F1: : {3: .3f}'.format(accuracy, precision, recall, f1))
print('ROC_AUC: : {0: .3f}'.format(roc_auc))
print('by BD 오후 컴퓨터소트프웨어학과 21101849 이지은') 