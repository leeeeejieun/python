from bs4 import BeautifulSoup
import pandas as pd

from selenium import webdriver # selenium : 오픈 소스 테스트 자동화 프레임워크
import time

# 커피빈 매장 정보 추출 
def coffeeBean_store(result):
    coffeeBean_url = 'https://www.coffeebeankorea.com/store/store.asp'
    
    wd = webdriver.Chrome()  # 크롬 WebDriver 객체 생성
  
    # 매장 수만큼 반복
    for i in range(1, ):
        wd.get(coffeeBean_url)  # 웹페이지 연결
        time.sleep(1)  # 웹페이지 연결할 동안 1초 대기
        try:
            wd.execute_script('storePop2(%d)' %i) # js 함수를 호출하여 매장 정보 페이지 열기
            time.sleep(1)  # 스크립트 실행할 동안 1초 대기
            html = wd.page_source  # js 함숙 수행된 페이지의 소스 코드 저장
            soupCB1 = BeautifulSoup(html, 'html.parser') # html 태그 파싱

            # 매장 이름(지점명) 추출하기
            store_name_h2 = soupCB1.select('div.store_txt > h2')
            print(store_name_h2[0])
            store_name = store_name_h2[0].string
            print(store_name)

            # 매장 정보 추출하기
            store_info = soupCB1.select('div.store_txt > table.store_table > tbody > tr > td')
            store_address_list = list(store_info[2])  # td 태그 내부의 자식 노드들을 리스트 형태로 분할
            store_address = store_address_list[0]

            # 매장 전화번호 추출
            store_phone = store_info[3].string
            
            # 매장 정보 데이터 처리
            result.append([store_name]+[store_address]+[store_phone])
        # 예외처리
        except:
            continue
    return

# main 함수
def main():
    result = []  # 매장 정보를 저장할 리스트
    print('CoffeeBean Store 크롤링을 시작합니다. >>>> ')
    coffeeBean_store(result)  

    # 데이터 프레임 생성
    cb_tbl =  pd.DataFrame(result, columns=('지점명', '주소', '전화번호'))
    # csv 파일 생성
    cb_tbl.to_csv('./python/week02/coffeeBean.csv', encoding='utf-8', mode='w', index=True)
    print('coffeeBean.csv 파일 저장이 완료되었습니다. >>>> ')

    print('by BD 오후 컴퓨터소트프웨어학과 21101849 이지은')

# 프로그램 실행
if __name__ == '__main__':
    main()