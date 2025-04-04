import os
from dotenv import load_dotenv
import requests
import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt


# 한글 폰트 설정
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# .env 파일 로드
load_dotenv()
 
# 환경 변수 가져오기
service_key = os.getenv('SERVICE_KEY')

# url 접속을 요청하고 응답을 반환
def getRequestUrl(url):
    try:
        response = requests.get(url)  # requests 라이브러리로 GET 요청
        if response.status_code == 200:  # 응답 코드가 200이면 성공
            print("[%s] Url Request Success" % datetime.datetime.now())  # 성공 메시지 출력
            return response.text  # 응답 내용 반환
    except Exception as e:
        print(e)  # 예외 발생 시 에러 출력
        print("[%s] Error for URL : %s" % (datetime.datetime.now(), url))  # 에러 메시지 출력
        return None  # 에러 발생 시 None 반환

# 자전거 교통사고 다발지역 정보 API 호출하는 함수
def getBicycleAccidentAreas(year, siDo, guGun, num, page):
    base_url = 'http://apis.data.go.kr/B552061/frequentzoneBicycle/getRestFrequentzoneBicycle'  # 기본 URL

    # 파라미터 설정
    parameters = '?serviceKey=' + service_key # 인증키 추가
    parameters += '&searchYearCd=' + year   # 요청할 연도
    parameters += '&siDo=' + siDo  # 시도 코드
    parameters += '&guGun=' + guGun  # 시군구 코드
    parameters += '&type=json&numOfRows=%s&pageNo=%s' %(num, page)  # num : 검색 건수 / page : 페이지 번호

    # 최종 URL 생성
    url = base_url + parameters
    print(url)
    responseDecode = getRequestUrl(url)  # getResultUrl 함수 호출 후 응답 저장

    if(responseDecode == None):
        print('API 호출 실패')
        return None  # 응답이 없으면 None 반환
    else:
        return json.loads(responseDecode)  # 응답 문자열을 딕셔너리 타입으로 변환
    
# 수집한 자전거 교통사고 다발지역 데이터 가공
def getBicycleAccidentData(area, jsonResult):
    jsonResult.append({
        '사고 지점명': area.get('spot_nm'),
        '사고건수': area.get('occrrnc_cnt'),
        '사상자수': area.get('caslt_cnt'),
        '사망자수': area.get('dth_dnv_cnt'),
        '중상자수': area.get('se_dnv_cnt'),
        '경상자수': area.get('sl_dnv_cnt'),
    })
    return

# main 함수
def main():
    result = []  # 검색 결과를 정리하여 저장할 리스트
    print("<< 자전거 교통사고 다발지역 데이터를 수집합니다. >>")

    # 사용자 입력 받기
    siDo = input('시도 코드를 입력하세요(서울특별시: 11 / 경기도: 41 ) : ')
    guGun = input('시군구 코드를 입력하세요(강남구: 680/ 부천시 원미구: 195) : ')
    year = input('수집할 연도를 입력하세요 : ')

    page = 1
    while True:
        jsonResponse = getBicycleAccidentAreas(year, siDo, guGun, num=100, page=page)

        # jsonResponse가 None이면 종료
        if jsonResponse is None:
            break
        
        # 각 페이지에서 데이터를 처리
        for area in jsonResponse['items']['item']:
            getBicycleAccidentData(area, result)
        
        # 데이터가 더 이상 없으면 종료
        if len(jsonResponse['items']['item']) < 100:  # 100개 미만이면 마지막 페이지
            break
        
        page += 1  # 페이지 번호 증가


    print("모든 데이터를 수집했습니다.")

    df = pd.DataFrame(result) # DataFrame으로 변환
    df.to_csv('./week01/bicycle_accident_data.csv', index=False, encoding='utf-8-sig')  # CSV 파일로 저장
    
    # 사고 지점명을 인덱스로 설정
    df.set_index('사고 지점명', inplace=True)

    # 데이터 전치
    df.T.plot()

    # 그래프 제목과 축 레이블
    plt.title('각 사고 지점별 사고 유형별 수치')
    plt.xlabel('항목명')  # 사고건수, 사상자수 등
    plt.ylabel('건수')

    # 그래프 표시
    plt.show()

# 프로그램 실행
if __name__ == "__main__":
    main()
