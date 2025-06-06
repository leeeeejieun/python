import pandas as pd
import re    
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

print('------ 텍스트 마이닝을 이용하여 영화 리뷰 데이터로 감성 분석 모델링하기 ------')

train_df = pd.read_csv('./텍스트마이닝/영화리뷰_감성분석/ratings_train.txt',
                 encoding='utf8', sep='\t', engine='python')   # 영화 리뷰 훈련용 데이터 파일 불러오기
test_df = pd.read_csv('./텍스트마이닝/영화리뷰_감성분석/ratings_test.txt',
                 encoding='utf8', sep='\t', engine='python')   # 영화 리뷰 평가용 데이터 파일 불러오기
# 작업 확인용 출력 
print('** 결측치 제거 전 정보 **')
print(train_df.info())
print(train_df.isnull().sum())
print(test_df.info())
print(test_df.isnull().sum())

# 결측치 제거
train_df = train_df[train_df['document'].notnull()]    
test_df = test_df[test_df['document'].notnull()]  
print('** 결측치 제거 후 정보 **')
print(train_df.info())
print(train_df.isnull().sum())
print(test_df.info())
print(test_df.isnull().sum())

# 감성 분류 클래스 구성 확인
print(train_df['label'].value_counts())    
print(test_df['label'].value_counts())  

# 한글 외의 문자 제거하기
train_df['document'] = train_df['document'].apply(lambda x: re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', x))  
test_df['document'] = test_df['document'].apply(lambda x: re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', x))  

okt = Okt()   # okt 객체 생성

# 형태소 단위로 토큰화 작업을 수행하는 함수
def okt_tokenizer(text):
    tokens = okt.morphs(text)
    return tokens

# 형태소 단위로 토큰화한 한글 단어에 대해 TF-IDF 방식을 사용하여 벡터화 작업을 수행
tfidf = TfidfVectorizer(tokenizer = okt_tokenizer, ngram_range=(1, 2),
                        min_df=3, max_df=0.9, token_pattern=None)
tfidf.fit(train_df['document'])  # 전체 데이터로 단어 사전과 IDF 학습
train_tfidf = tfidf.transform(train_df['document'])  # 학습된 정보로 각 데이터를 TF-IDF 벡터로 변환

SA_lr = LogisticRegression(random_state=0, max_iter=500)   # 로지스틱 회귀 모델 객체 생성
SA_lr.fit(train_tfidf, train_df['label'])    # 모델 학습

params = {'C': [1, 3, 3.5, 4, 4.5, 5]}  # C 값(규제 강도) 후보 리스트 설정
SA_lr_grid_cv = GridSearchCV(SA_lr, param_grid=params, cv=3,
                             scoring='accuracy', verbose=1)  # 3겹 교차검증으로 최적 C를 찾는 GridSearchCV 객체 생성
SA_lr_grid_cv.fit(train_tfidf, train_df['label'])  # 모델 학습
print(SA_lr_grid_cv.best_params_, round(SA_lr_grid_cv.best_score_, 4))  # GridSearchCV에 의해 찾은 최적의 C 매개변수와 최고 점수 확인
SA_lr_best = SA_lr_grid_cv.best_estimator_  # 최적의 매개변수가 설정된 모델 저장

test_tfidf = tfidf.transform(test_df['document'])   # 평가 데이터의 피처 벡터화
test_predict = SA_lr_best.predict(test_tfidf)  #  평가 데이터를 이용해 감성 예측
print('감성 분석 정확도: ', round(accuracy_score(test_df['label'], test_predict), 3))

# 새로운 텍스트로 감성 예측 확인하기
st = input('감성 분석할 문장 입력>> ')
# 입력 테스트에 대한 전처리 수행
st = re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', st)
st = [st] 
print(st)

st_tfidf = tfidf.transform(st)  # 입력 텍스트의 피처 벡터화
st_predict = SA_lr_best.predict(st_tfidf)  # 최적 감성 분석 모델에 적용하여 감성 분석 평가

# 예측값 출력
if(st_predict[0] == 0):
    print(st, '->> 부정 감성')
else:
    print(st, '->> 긍정 감성')
print('by BD 오후 컴퓨터소트프웨어학과 21101849 이지은') 

