from bs4 import BeautifulSoup
import requests  
import pandas as pd

'''
requests / urlib.request 모듈 차이
1. requests
- 데이터를 보낼 때 딕셔너리 형태로 보낸다.
- 없는 페이지를 요청해도 에러를 띄우지 않는다.

2. urlib.request
- 데이터를 보낼 때 인코딩하여 바이너리 형태로 보낸다.
- 없는 페이지를 요청해도 에러를 띄운다
'''


# 크롤링을 이용하여 네이버 뉴스 기사 찾기
def naver_news(result, query):
   page = 1  # 1 페이지부터 시작
    
   while True:
    naver_news_url = 'https://search.naver.com/search.naver?ssc=tab.news.all&where=news&sm=tab_jum&query=%s&start=%d' % (query, page) # query : 검색어 / page : 페이지 번호
    
    html = requests.get(naver_news_url)  # 해당 URL에 HTTP 요청
    soupNews = BeautifulSoup(html.text, 'html.parser') # HTML 태그 파싱

    titles = soupNews.find_all('a', {'class': 'news_tit'})  # 뉴스 기사의 제목이 존재하는 모든 a 태그 파싱
    
    # 기사가 존재하지 않는 경우 크롤링 종료
    if not titles:
       print('마지막 페이지입니다.')
       break
    
   # 기사가 존재하는 경우 각 기사의 제목&링크 파싱
    for t in titles:
        news_title = t.attrs['title']  # 기사 제목
        news_href = t.attrs['href']  # 기사 링크
        result.append([news_title, news_href])  # 제목&링크 저장
    page += 10  # 페이지 번호 증가(한 페이지마다 10씩 증가함)

   return result

# main 함수
def main():
  result = []  # 검색 결과를 저장할 리스트
  search = input('검색어를 입력하세요: ')

  print(search + ' 검색어에 대한 네이버 뉴스 크롤링을 시작합니다 >>')
  naver_news(result, search)
   
  print(result)
  # 데이터 프레임 생성
  news_tbl = pd.DataFrame(result, columns=('기사 제목', 'url'))
  
  # 데이터 프레임을 csv 파일로 저장
  news_tbl.to_csv('./python/week02/news.csv', encoding ='utf-8', mode='w', index=True)  
  print('news.csv 파일 저장이 완료되었습니다. >>')

# 프로그램 실행
if __name__ == '__main__':
 main()