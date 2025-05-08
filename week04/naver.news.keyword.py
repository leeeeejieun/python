from bs4 import BeautifulSoup
import re
import requests
from konlpy.tag import Okt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import time

# 섹션별 고유 번호
sections = {
    '정치': 100,
    '경제': 101,
    '사회': 102,
    '생활/문화': 103,
    '세계': 104,
    'IT/과학': 105
}

# 네이버 뉴스에서 특정 섹션과 키워드를 기준으로 기사 제목 수집 
def naver_news(section, keyword):
    section_number = sections[section]

    titles = []  # 수집된 기사 제목 리스트
    base_url = f'https://news.naver.com/section/{section_number}'  # 섹션 URL

    # 100개가 초과되면 크롤링 종료
    while len(titles) < 100:
        response = requests.get(base_url)     # HTTP 요청
        soup = BeautifulSoup(response.text, 'html.parser')  # HTML 태그 파싱

        news_items = soup.select('a.sa_text_title > strong.sa_text_strong')  # 기사 제목이 들어있는 태그 파싱
        for item in news_items:
            title = item.get_text(strip=True)  # 제목 텍스트 추출

            # 제목에 키워드가 포함될 경우만 저장
            if keyword in title:
                titles.append(title)
                if len(titles) >= 100:
                    break

        # 더보기 버튼이 없으면 종료
        more_button = soup.find('a', class_='section_more_inner')
        if not more_button or not more_button.get('href'):
            break

        # 다음 페이지 URL로 갱신
        base_url = 'https://news.naver.com' + more_button.get('href')
        time.sleep(1)  # 1초 대기

    return titles

# 기사 제목 리스트에서 명사 추출 및 빈도수 계산
def text_processing(titles):
    okt = Okt() 
    nouns = []  # 명사를 저장할 리스트

    for title in titles:
        clean_title = re.sub(r'[^\w]', ' ', title)   # 문자나 숫자가 아닌 것은 공백으로 치환 후 하나의 문자열로 구성
        nouns.extend(okt.nouns(clean_title))  # 명사만 추출

    
    count = Counter(nouns)   # 단어 빈도수 계산
    word_count = {word: freq for word, freq in count.items() if len(word) > 1}  # 한 글자 단어 제거
    return word_count

# 워드클라우드 시각화
def visualize_wordcloud(word_freq):
    # WordCloud 객체 생성
    wc = WordCloud(
        font_path='C:/Windows/Fonts/malgun.ttf',  # 한글 폰트 지정
        background_color='ivory',
        width=800,
        height=600
    )
    cloud = wc.generate_from_frequencies(word_freq)  # 빈도 기반 워드클라우드 생성
    plt.figure(figsize=(8, 8))  # 워드클라우드를 시각화할 그래프 크기 설정 (8x8)
    plt.imshow(cloud)    # 워드클라우드를 이미지로 표시
    plt.axis('off')   # 축을 표시하지 않음
    plt.show()  # 워드클라우드 화면에 표시

# 전체 실행 흐름
def main():
    section = input('분야 입력 (정치, 경제, 사회, 생활/문화, 세계, IT/과학): ').strip()
    keyword = input('포함할 키워드 입력: ').strip()
    print(f'{section} 분야에서 "{keyword}" 포함 기사 크롤링 시작')
  
    titles = naver_news(section, keyword)    # 키워드가 포함된 기사 제목 추출 함수 호출
    print(f'총 {len(titles)}개 기사 수집 완료')

    word_freq = text_processing(titles)    # 텍스트 전처리 및 빈도수 계산 함수 호출
    visualize_wordcloud(word_freq)  # 워드클라우드 시각화
    print('by BD 오후 컴퓨터소트프웨어학과 21101849 이지은')

if __name__ == '__main__':
    main()
