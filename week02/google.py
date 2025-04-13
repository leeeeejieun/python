from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import urllib.request

# 구글 이미지 검색
def image_search(search):
    google_url = 'https://www.google.com/imghp?hl=ko&ogbl'  # 구글 이미지 검색 기본 url

    wd = webdriver.Chrome()  # 크롬 WebDriver 객체 생성
    
    wd.get(google_url)  # 웹페이지 연결
    time.sleep(1)  # 웹페이지 연결할 동안 1초 대기

    # 검색어 입력
    search_tag = wd.find_element(By.CLASS_NAME, 'gLFyf')  # 검색창 찾기
    search_tag.send_keys(search + Keys.ENTER)   # 검색어 입력 후 엔터키 실행

    scroll(wd)  # 스크롤 내리는 함수 호출
    images = wd.find_elements(By.CSS_SELECTOR, 'div.H8Rx8c > g-img > img')  # 해당 페이지의 모든 이미지 가져오기 
    image_download(wd, images)  # 이미지 다운로드 받는 함수 호출
  
   
# 페이지 스크롤을 끝까지 내려서 이미지 불러오는 함수
def scroll(wd):
    last_height = wd.execute_script('return document.body.scrollHeight')  # 현재 페이지 높이 가져옴

    while True:
       wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")  # 페이지의 가장 아래로 스크롤 내리기
       time.sleep(1)  # 페이지 로딩 대기
       
       # 스크롤 내린 후 실행 전과 후의 높이 비교
       new_height = wd.execute_script('return document.body.scrollHeight') # 스크롤을 내린 후의 현재 페이지 높이 가져옴
       # 페이지 높이가 더 이상 변하지 않으면 종료
       if new_height == last_height:
            break
       last_height = new_height


# 이미지를 다운로드 하는 함수
def image_download(wd, images):
    count = 1
    for image in images:
        # 이미지가 100개가 넘으면 다운로드 종료
        if count > 100 :
            break  

        wd.execute_script('arguments[0].click();', image) # Image Lazy Loading 방지를 위해서 이미지 클릭 수행(원본 이미지 데이터를 가져오기 위함)
        time.sleep(1)  # 클릭 후 1초 대기
        
        url = image.get_attribute('src') # 이미지의 url 가져오기
        urllib.request.urlretrieve(url, f'./python/week02/images/{count}.jpg')  # 이미지를 지정된 경로에 다운로드
        count += 1  # 이미지 개수 증가
    print('이미지 저장이 완료되었습니다.')
      
# main 함수
def main():
    search = input('검색어를 입력하세요: ')
    print('%s에 대한 이미지 검색을 시작합니다. >>>>' %search)
    image_search(search)  # 이미지 검색 함수 호출

# 프로그램 실행
if __name__ == '__main__':
    main()

   
   