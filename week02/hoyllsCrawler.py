from bs4 import BeautifulSoup
import urllib.request
import pandas as pd

# 크롤링을 이용하여 할리스 매장 정보 찾기
def hollys_store(result):
    for page in range(1,52):
        Hollys_url = 'https://www.hollys.co.kr/store/korea/korStore2.do?pageNo=%d&sido=&gugun=&store=' %page # 페이지 넘버 1~51까지 순환
        print(Hollys_url)
        html = urllib.request.urlopen(Hollys_url)  # 해당 URL에 HTTP 요청
        
        soupHollys = BeautifulSoup(html, 'html.parser')  # html 태그 파싱
        tag_tbody = soupHollys.find('tbody')  # tbody 태그 파싱

        # tbody의 하위 태그인 모든 tr 태그 순환
        for store in tag_tbody.find_all('tr'):
            # 마지막 tr 태그인 경우 매장 정보가 없으므로 크롤링 중단
            if len(store) <=3 :
                break
            store_td = store.find_all('td')    # 모든 td 태그 파싱
            store_name = store_td[1].string    # 매장 이름 
            store_sido = store_td[0].string    # 매장 지역 
            store_address = store_td[3].string # 매장 주소
            store_phone = store_td[5].string   # 매장 전화번호

            # 매장 정보 데이터 생성
            result.append([store_name]+[store_sido]+[store_address]+[store_phone])
    return result 
    
# main 함수
def main():
    result = []  # 매장 정보를 저장할 배열
    print('할리스 매장 정보 크롤링을 시작합니다 >>')
    hollys_store(result)  # 매장 정보 찾는 함수 호출
    
    # 데이터 프레임 생성
    hollys_tbl = pd.DataFrame(result, columns=('지점명', '지역명', '주소', '전화번호'))

    # 데이터 프레임을 csv 파일로 저장
    hollys_tbl.to_csv('./python/week02/hollys.csv', encoding='cp949', mode='w', index=True)
    print('hollys.csv 파일 저장이 완료되었습니다. >>')

    
# 프로그램 실행
if __name__ == '__main__':
 main()