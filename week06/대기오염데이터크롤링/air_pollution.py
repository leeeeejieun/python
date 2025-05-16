import requests
import datetime
import json
import urllib.parse
import pandas as pd
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 공공데이터포털에서 발급 받은 인증키
ServiceKey = os.getenv('ServiceKey')

# url 접속을 요청하고 응답을 반환
def getRequestUrl(url):
    try:
        response = requests.get(url)  # 해당 URL에 HTTP GET 요청
        # 요청이 성공한 경우
        if response.status_code == 200:
            print('[%s] Url Request Success' %datetime.datetime.now())  # 성공 응답 출력
            print(response.text)
            return response.text # 응답을 반환
        # 요청이 실패한 경우
    except Exception as e:
        print(e)
        print('[%s] Url Error for URL : %s' %(datetime.datetime.now(), url))  # 실패 응답 출력
        return None  

# 대기 오염 데이터 API 호출 함수
def reqAirInfo(where, beginDay, endDay, numOfRows):
    service_url = 'http://apis.data.go.kr/B552584/ArpltnStatsSvc/getMsrstnAcctoRDyrg'
    # 파라미터 설정
    params = {
        'returnType': 'json',          # 데이터 표출 방식을 json으로 지정
        'serviceKey': ServiceKey,      # 서비스 키 
        'msrstnName': where,  #  측정소명
        'inqBginDt' : beginDay,   # 조회 시작일
        'inqEndDt' : endDay,      # 조회 종료일
        'numOfRows': numOfRows    # 한 페이지 결과 수
    }

    query_string = urllib.parse.urlencode(params)
    url = service_url + '?' + query_string
    print(url)
    responseDecode = getRequestUrl(url)   # 해당 url에 데이터를 요청하는 함수 호출
    # 응답이 없는 경우
    if(responseDecode == None):
        return None
    # 응답이 존재하는 경우
    else:
        return json.loads(responseDecode)   # 응답으로 받은 json 문자열을 딕셔너리 자료형으로 변환

# 응답 데이터 처리 함수
def getAirInfoItem(item, result):
    # 응답 데이터에서 필요한 데이터 추출
    msrstnName = item['msrstnName']  # 측정소
    msurDt = item['msurDt']      # 측정일
    so2Value = item['so2Value']  # 아황산가스 평균 농도
    coValue  = item['coValue']   # 일산화탄소 평균 농도
    o3Value  = item ['o3Value']  # 오존 평균 농도
    no2Value = item['no2Value']  # 이산화질소 평균 농도
    pm10Value = item['pm10Value']  # 미세먼지 평균 농도
    pm25Value = item['pm25Value']  # 초미세먼지 평균 농도

    result.append([msrstnName, msurDt, so2Value, coValue, o3Value,
                    no2Value, pm10Value, pm25Value])  #  리스트에 저장
# main 함수
def main():
    print('대기오염 데이터와 미세먼지의 연관성 분석하기')
    print('--------------------------------------')

    where = input('대기오염 데이터 수집 측정소를 입력하세요 : ')
    # API 수정일 기준으로 최근 약 3~6개월 내의 데이터만 제공 (마지막 수정일: 2025-04-02)
    beginDay = input('대기오염 데이터 수집 시작 일자를 입력하세요(YYYYMMDD) : ')
    endDay = input('대기오염 데이터 수집 종료 일자를 입력하세요(YYYYMMDD) : ')

    # 날짜 형식을 20220401 -> 2022-04-01로 변환 
    b_date = datetime.datetime.strptime(beginDay, '%Y%m%d')
    e_date = datetime.datetime.strptime(endDay, '%Y%m%d')
    days = e_date - b_date      # 날짜 차이 계산
    numOfRows = str(days.days)  # 크롤링 데이터 수 : 수집 기간의 일수

    jsonResponse = []   # 크롤링한 JSON 데이터를 저장할 리스트
    result = []   # JSON 항목을 추출하여 저장할 리스트 -> csv로 저장

    jsonResponse = reqAirInfo(where, beginDay, endDay, numOfRows)   # 대기오염 데이터 API 호출
    for item in jsonResponse['response']['body']['items']:
        getAirInfoItem(item, result)   # 응답 데이터 처리 함수 호출
    
    columnNames = ['location', 'day',' so2', 'co', 'o3', 'no2', 'pm10', 'pm25']  # 컬럼명 지정
    result_df = pd.DataFrame(result, columns=columnNames)  # 데이터프레임 생성
    result_df.to_csv('./week06/대기오염데이터크롤링/대기오염데이터_%s_%s_%s.csv' 
                     %(where, beginDay, endDay), index=False, encoding='cp949') # 최종 데이터를 csv 파일에 저장
    print('./week06/대기오염데이터크롤링/대기오염데이터_%s_%s_%s.csv 저장 완료.' %(where, beginDay, endDay))
    print('by BD 오후 컴퓨터소트프웨어학과 21101849 이지은')

# 프로그램 실행
if __name__ == '__main__':
    main()