import json
import re
from konlpy.tag import Okt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from wordcloud import WordCloud
from collections import Counter

# 아시안컵 뉴스 크롤링 결과 json 파일 불러오기
data = json.loads(open('./week04/korea/아시안컵_naver_news.json', 'r', encoding='utf-8').read())

description = ''
for item in data:
    if 'description' in item.keys():
        description = description + re.sub(r'[^\w]', ' ', item['description'] )  # 문자나 숫자가 아닌 것은 공백으로 치환 후 하나의 문자열로 구성

npl = Okt()
description_N = npl.nouns(description)  # 명사 추출

count = Counter(description_N)  # 단어 빈도수 계산

word_count = dict()
# 상위 빈도수 100개 단어 중에서 길이가 1보다 긴 키워드만 선별
for tag, counts in count.most_common(100):
    if(len(str(tag)) > 1):
        word_count[tag] = counts
        print('%s : %d' %(tag, counts))
    
plt.rc('font', family = 'Malgun Gothic') # 폰트 설정
font_path = 'C:/Windows/Fonts/malgun.ttf' # Malgun Gothic 경로
plt.figure(figsize=(12, 5))  # 그래프 크기를 가로 12, 세로 5로 설정
plt.xlabel('키워드')    # x축 레이블 설정
plt.ylabel('빈도수')    # y축 레이블 설정
plt.grid(True)  # 격자 표시
# 단어 빈도수 기준으로 키를 내림차순 정렬
sorted_keys = sorted(word_count, key=word_count.get, reverse=True)
# 단어의 빈도수만 내림차순 정렬
sorted_values = sorted(word_count.values(), reverse=True)
# 막대그래프 그리기: x축은 단어 인덱스, y축은 빈도수
plt.bar(range(len(word_count)), sorted_values, align='center')
# x축 눈금을 단어로 설정하고, 글자가 겹치지 않도록 75도 회전
plt.xticks(range(len(word_count)), sorted_keys, rotation=75)
plt.show()

# WordCloud 객체 생성
wc = WordCloud(font_path, background_color='ivory', width=800, height=600)
cloud = wc.generate_from_frequencies(word_count) # word_count의 단어 빈도를 바탕으로 워드클라우드를 생성
plt.figure(figsize=(8, 8))  # 워드클라우드를 시각화할 그래프 크기 설정 (8x8)
plt.imshow(cloud)  # 워드클라우드를 이미지로 표시
plt.axis('off')    # 축을 표시하지 않음
plt.show()  # 워드클라우드 화면에 표시

cloud.to_file('./week04/korea/아시안컵_cloud.jpg') # 워드클라우드를 jpg 파일로 저장