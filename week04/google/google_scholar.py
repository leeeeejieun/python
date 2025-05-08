from serpapi import GoogleSearch
import random
import time
import re
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

# 키워드 목록
keywords = ["large language model", "deep learning", "blockchain", "climate change", "Unmanned Aerial Vehicle"]

# SerpAPI 사용을 위한 API 키 설정
api_key = "90349559c8ea81f59eda9ae16aadf5703a09aadd62fe271fb86ca4bdb7392161"

# NLP 전처리 도구 설정
stopWords = set(stopwords.words('english'))  # 불용어 가져오기
lemma = WordNetLemmatizer()  # 표제어 추출을 위한 Lemmatizer 설정

titles = {}  # 각 키워드에 해당하는 논문 제목들을 저장할 딕셔너리

# 각 키워드에 대해 5페이지의 논문 제목을 수집
for keyword in keywords:
    keyword_titles = []       # 각 키워드에 해당하는 논문 제목을 저장할 리스트
    for page in range(1, 6):  # 각 키워드에 대해 5페이지 반복
        params = {
            "q": keyword,     # 검색할 키워드 
            "engine": "google_scholar",   # Google Scholar 검색 엔진 사용
            "start": (page - 1) * 10,     # 시작 페이지 지정
            "api_key": api_key
        }

        search = GoogleSearch(params)   # 검색 요청
        results = search.get_dict()     # 검색 결과를 저장

        # 결과에서 논문 제목 추출
        for result in results.get("organic_results", []):  # 결과에서 'organic_results' 항목만 추출
            title = result.get("title")   # 제목을 가져옴
            # 제목이 있는 경우만 리스트에 추가
            if title:
                keyword_titles.append(title)

        time.sleep(random.uniform(1, 5))  # 딜레이 시간 랜덤 설정
    titles[keyword] = keyword_titles      # 해당 키워드에 대해 수집된 논문 제목을 저장

# 키워드별 단어 빈도 분석 결과 저장
keyword_word_counts = {}

# 시각화: 키워드별 상위 10개 단어
for keyword in keywords: 
    keyword_titles = titles[keyword]  # 현재 키워드에 해당하는 논문 제목 리스트
    all_words = []   # 모든 단어를 저장할 리스트
    # 텍스트 전처리 및 토큰화
    for title in keyword_titles:
        cleaned = re.sub(r'[^a-zA-Z]+', ' ', title)    # 제목에서 알파벳으로 시작하지 않는 단어를 공백으로 치환
        tokens = word_tokenize(cleaned.lower())        # 소문자로 변환 후 단어 단위로 토큰화
        tokens = [w for w in tokens if w not in stopWords]  # 불용어 제거
        lemmatized = [lemma.lemmatize(w) for w in tokens]   # 표제어 추츌
        all_words.extend(lemmatized)
    # 단어 빈도 계산
    word_freq = Counter(all_words)
    keyword_word_counts[keyword] = word_freq

# 키워드별 상위 10개 단어 시각화
for keyword in keywords:
    top_words = keyword_word_counts[keyword].most_common(10)  # 상위 10개의 단어어만 추출
    
    words, counts = zip(*top_words)  # 단어와 빈도를 각각 분리
    plt.rc('font', family = 'Malgun Gothic') # 폰트 설정
    plt.figure(figsize=(10, 5))  # 그래프 크기 설정(가로 10, 세로 5)
    plt.bar(words, counts, color='skyblue')  # 막대 그래프 그리기
    plt.title(f"'{keyword}' 관련 상위 10개 단어")  # 그래프 제목 한글로 변경
    plt.xlabel("단어")          # x축 레이블 한글로 변경
    plt.ylabel("빈도수")        # y축 레이블 한글로 변경
    plt.xticks(rotation=45)    # x축 레이블 45도도 회전
    plt.tight_layout()   # 레이아웃 자동 조정
    plt.show()

print('by BD 오후 컴퓨터소트프웨어학과 21101849 이지은')