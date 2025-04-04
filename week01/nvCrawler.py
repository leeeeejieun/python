import os
from dotenv import load_dotenv
import urllib.request
import datetime
import json

# .env 파일 로드
load_dotenv()

# 환경 변수 가져오기
client_id =  os.getenv('CLIENT_ID')          # 애플리케이션 등록 시 발급받은 클라이언트 아이디 값
client_secret = os.getenv('CLIENT_SECRET')   # 애플리케이션 등록 시 발급받은 클라이언트 시크릿 값

# url 접속을 요청하고 응답을 반환
def getResultUrl(url):
    # HTTP 요청 객체 생성
    req = urllib.request.Request(url)   
    req.add_header('X-Naver-Client-Id', client_id)
    req.add_header('X-Naver-Client-Secret', client_secret)

    try:
        response = urllib.request.urlopen(req)  # URL 요청을 보냄
        if response.getcode() == 200:  # 응답 코드가 200이면 성공
            print('[%s] Url Request Success' %datetime.datetime.now())   # 성공 메시지 출력
            return response.read().decode('utf-8')  # 응답 내용 반환 (utf-8로 디코딩)
    except Exception as e:
        print(e)  # 예외 발생 시 에러 출력
        print('[%s] Error for URL: %s' %(datetime.datetime.now(), url))  # 에러 메시지 출력
        return None  

# 네이버 검색 API 호출 함수
def getNaverSearch(node, srcText, start, display):  # start : 검색 시작 위치 / display : 한 번에 표시할 검색 결과 개수
    base = 'https://openapi.naver.com/v1/search'    # 네이버 검색 API 기본 URL
    node = '/%s.json' %node   # 검색 결과는 json 형식으로 반환
    parameters = '?query=%s&start=%s&display=%s' % (urllib.parse.quote(srcText), start, display)  # 파라미터 설정

    url = base + node + parameters      # 최종 URL 생성
    responseDecode = getResultUrl(url)  # getResultUrl 함수 호출 후 응답 저장

    if(responseDecode == None):
        print('API 호출 실패')
        return None  # 응답이 없으면 None 반환
    else:
        return json.loads(responseDecode) # 응답 문자열을 딕셔너리 타입으로 변환

# json 형식의 응답 데이터를 필요한 항목만 정리하여 딕셔너리 리스트인 jsonResult를 구성하고 반환
def getPostData(post, jsonResult, cnt):
    title = post['title']              # 게시글 제목
    description = post['description']  # 게시글 설명
    org_link = post.get('originallink', 'N/A')  # 원본 링크 (없을 경우'N/A'로 대체)
    link = post['link']  # 게시글 링크
    
    pDate = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 현재 날짜 및 시간

    # 데이터를 리스트에 추가
    jsonResult.append({'cnt':cnt, 'title':title, 'description': description, 'org_link':org_link, 'link': link, 'pDate' :pDate})
    return

# main 함수
def main():
    node = 'blog'    # 검색할 노드 설정 ('news' : 뉴스, 'blog' : 블로그 등)
    scrText = input('검색어를 입력하세요: ')  # 사용자 입력으로 받은 검색어 저장
    cnt = 0          # 검색 결과 카운트
    jsonResult = []  # 검색 결과를 정리하여 저장할 리스트 객체

    # 첫 번째 페이지(시작 위치 1, 한 번에 100개 표시)부터 검색 시작
    jsonResponse = getNaverSearch(node, scrText, 1, 100) 
    total = jsonResponse['total']   # 총 검색 결과 개수 저장

    # 검색 결과가 존재하고 표시할 항목이 있을 때까지 반복
    while ((jsonResponse != None) and (jsonResponse['display'] != 0)):
        for post in jsonResponse['items']:   # 검색 결과를 하나씩 가져옴
            cnt += 1  # 검색 결과 카운트 1씩 증가
            getPostData(post, jsonResult, cnt) # 게시글 데이터를 처리하여 리스트에 저장

        # 다음 페이지로 넘어가기 위한 시작 위치 계산
        start = jsonResponse['start'] + jsonResponse['display']
        if start == 1001: break   # 최대 1000개까지만 가져오므로 1001부터는 종료
        jsonResponse = getNaverSearch(node, scrText, start, 100)  # 다음 페이지 데이터 요청

    # 전체 검색 결과 개수 출력
    print('전체 검색: %d건 '%total)

    # JSON 형식으로 결과 파일 저장
    with open('./week01/%s_naver_%s.json' %(scrText, node), 'w', encoding = 'utf-8') as outfile:
        jsonFile = json.dumps(jsonResult, indent = 4, sort_keys = True, ensure_ascii = False)

        outfile.write(jsonFile)

    # 검색된 데이터 개수 출력
    print('기져온 데이터 %d 건 ' %(cnt))
    print('%s_naver%s.json SAVED' %(scrText, node))            
    print('by BD 오후 컴퓨터소트프웨어학과 21101849 이지은')

# 프로그램 실행
main()