import pandas as pd
import glob
import re
from functools import reduce
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS, WordCloud

print('[영문 분석 + 워드클라우드] 영문 문서 제목의 키워드 분석하기')

# exportExcelData로 시작하는 모든 엑셀 파일을 리스트에 저장
all_file = glob.glob('./week04/english/exportExcelData_*.xls')

all_files_data = []     # 병합 데이터를 저장할 리스트
# 파일을 하나씩 읽어서 리스트에 저장
for file in all_file:
    df = pd.read_excel(file)
    all_files_data.append(df)

# 데이터프레임들을 행을 기준으로 병합
all_files_data_concat = pd.concat(all_files_data, axis=0, ignore_index=True)
# csv 파일에 저장
all_files_data_concat.to_csv('./week04/english/riss_bigData.csv', encoding='utf-8', index=False)
# 제목 추출
all_title = all_files_data_concat['제목']

# nlkt.corpus에서 제공하는 영어 불용어를 중복 없이 불러오기
stopWords = set(stopwords.words('english'))
# 표제어 추출 작업을 위한 WordNetLemmatizer 객체 생성
lemma = WordNetLemmatizer()

words = []


for title in all_title:
    EnWords = re.sub(r'[^a-zA-Z]+', ' ', str(title)) # 제목에서 알파벳으로 시작하지 않는 단어를 공백으로 치환
    EnWordsToken = word_tokenize(EnWords.lower())    # 소문자로 변환 후 단어 단위로 토큰화
    EnWordsTokenStop = [w for w in EnWordsToken if w not in stopWords]  # 불용어 제거
    EnWordsTokenStopLemma = [lemma.lemmatize(w) for w in EnWordsTokenStop]  # 표제어 추철
    words.append(EnWordsTokenStopLemma)

words2 = list(reduce(lambda x, y: x+y, words))  # 1차원 리스트로 변환

count = Counter(words2)  # 단어 빈도수 계산

word_count = dict()

# 단어 빈도가 높은 상위 50개 단어 중에서 길이가 1보다 큰 것만 딕셔너리에 저장
for tag, counts in count.most_common(50):
    if(len(str(tag)) > 1):
        word_count[tag] = counts  # 단어와 빈도수를 키&값 쌍으로 저장
        print('%s : %d' %(tag, counts))  

# 빈도수가 높은 단어부터 내림차순으로 정렬하여 저장
sorted_keys = sorted(word_count, key=word_count.get, reverse=True)
# 빈도수를 내림차순으로 정렬하여 저장
sorted_values = sorted(word_count.values(), reverse=True)
# x축은 단어의 인덱스, y축은 해당 단어의 빈도수인 막대 그래프 그리기
plt.bar(range(len(word_count)), sorted_values, align='center')
# x축 눈금은 상위 50개 단어를 순서대로 사용하고 85도 회전시킴
plt.xticks(range(len(word_count)), sorted_keys, rotation=85)
plt.show()  

all_files_data_concat['doc_count'] = 0
# 출판일를 기준으로 그룹을 만들고 그룹별 데이터 개수를 저장
summary_year = all_files_data_concat.groupby('출판일', as_index=False)['doc_count'].count()

plt.figure(figsize=(12, 5))  # 그래프 크기 설정(가로 12, 세로 5)
plt.xlabel('year')      # x축 레이블을 'year'로 설정
plt.ylabel('doc-count') # y축 레이블을 'doc-count'로 설정
plt.grid(True)    # 그리드 추가
plt.plot(range(len(summary_year)), summary_year['doc_count'])   # x축 값은 0부터 'summary_year'의 길이까지, y축 값은 'doc_count' 값으로 선 그래프 그리기
plt.xticks(range(len(summary_year)), [text for text in summary_year['출판일']])   # x축 눈금의 값을 '출판일'로 설정
plt.show() 

# STOPWORDS에서 중복을 제거한 불용어 저장
stopwords = set(STOPWORDS)
# WordCloud 객체 생성
wc = WordCloud(background_color='ivory', stopwords=stopwords, width=800, height=600)
cloud = wc.generate_from_frequencies(word_count) # word_count의 단어 빈도를 바탕으로 워드클라우드를 생성
plt.figure(figsize=(8, 8))  # 워드클라우드를 시각화할 그래프 크기 설정 (8x8)
plt.imshow(cloud)  # 워드클라우드를 이미지로 표시
plt.axis('off')    # 축을 표시하지 않음
plt.show()  # 워드클라우드 화면에 표시


# 검색어로 사용한 'big'괴 'data'항목 제거
del word_count['big']
del word_count['data']

plt.figure(figsize=(12,5))  # 그래프 크기 설정
plt.xlabel('word')   # x축 레이블을 'word'로 설정
plt.ylabel('count')  # y축 레이블을 'count'로 설정
plt.grid(True)  # 그래프에 그리드 추가

# 'word_count'에서 값을 기준으로 내림차순으로 정렬한 키 목록을 가져오기
sorted_keys = sorted(word_count, key=word_count.get, reverse=True)
# 'word_count'에서 값(빈도수)을 기준으로 내림차순으로 정렬한 값 목록을 가져오기
sorted_values = sorted(word_count.values(), reverse=True)

# x축은 단어의 인덱스, y축은 해당 단어의 빈도수인 막대 그래프 그리기
plt.bar(range(len(word_count)), sorted_values, align='center')
# x축 눈금은 상위 50개 단어를 순서대로 사용하고 85도 회전시킴
plt.xticks(range(len(word_count)), sorted_keys, rotation=85)
plt.show()