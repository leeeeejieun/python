import pandas as pd
import folium

# Coffee Bean 매장 정보 불러오기
cb = pd.read_csv('./week05/커비빈매장_주소데이터분석/coffeeBean.csv', encoding='utf-8', index_col=0, header=0, engine='python')
# print(cb.head())  # 작업 확인용 출력

addr = []  # 매장 주소를 저장할 리스트
for address in cb['주소']:
    addr.append(str(address).split())  # 주소를 공백마다 분리해서 저장(예시: ['서울시', '강남구', '논현로', '566', '강남차병원1층'])


addr2 = []  # 주소 체계에 맞게 변환된 매장 주소를 저장할 리스트

# 주소 데이터를 행정구역 주소 체계에 맞게 변환
for i in range(len(addr)):
    if addr[i][0] == '서울': addr[i][0] = '서울특별시'
    elif addr[i][0] == '서울시': addr[i][0] = '서울특별시'
    elif addr[i][0] == '부산': addr[i][0] = '부산광역시'
    elif addr[i][0] == '인천': addr[i][0] = '인천광역시'
    elif addr[i][0] == '광주': addr[i][0] = '광주광역시'
    elif addr[i][0] == '대전시': addr[i][0] = '대전광역시'
    elif addr[i][0] == '울산시': addr[i][0] = '울산광역시'
    elif addr[i][0] == '세종시': addr[i][0] = '세종특별자치시'
    elif addr[i][0] == '경기': addr[i][0] = '경기도'
    elif addr[i][0] == '충북': addr[i][0] = '충청북도'
    elif addr[i][0] == '충남': addr[i][0] = '충청남도'
    elif addr[i][0] == '전북': addr[i][0] = '전라북도' 
    elif addr[i][0] == '전남': addr[i][0] = '전라남도'
    elif addr[i][0] == '경북': addr[i][0] = '경상북도'
    elif addr[i][0] == '경남': addr[i][0] = '경상남도'
    elif addr[i][0] == '제주': addr[i][0] = '제주특별자치도'
    elif addr[i][0] == '제주도': addr[i][0] = '제주특별자치도'
    elif addr[i][0] == '제주시': addr[i][0] = '제주특별자치도'
   
    addr2.append(' '.join(addr[i]))  # 공백마다 분리된 문자열들을 다시 합친 후 저장

addr2 = pd.DataFrame(addr2, columns=['address2']) # 데이터프레임으로 변환
cb2 = pd.concat([cb, addr2], axis=1) # cb와 addr2를 옆으로 결합하여 저장
cb2.to_csv('./week05/커비빈매장_주소데이터분석/coffeeBean2.csv', encoding='utf-8-sig', index=False) # csv 파일로 저장

cb_geo_data = pd.read_csv('./week05/커비빈매장_주소데이터분석/cb_geo.csv', encoding='utf-8', engine='python') # 지오데이터 가져오기
cb_geo_data.rename(columns={'field1': '지점명', 'field2': '주소', 'field3': '전화번호', 'field4': 'address2'}, inplace=True) # 컬럼명 변경

map_cb = map_osm = folium.Map(location=[37.56016, 126.9754], zoom_start=15) # 숭례문 좌표를 시작점으로 지도 객체 생성

# 매장 정보를 하나씩 읽어오기
for i, store in cb_geo_data.iterrows():  # iterrows()는 DataFrame의 각 행을 튜플(index, Series) 형태로 하나씩 반환
    '''
    위도 - 경도 순서대로 location 값을 설정
    매장에 대한 마커의 팝업 글자는 지점명으로 설정
    마커 모양은 빨간색 별 모양으로 설정하여 마커를 만든 뒤 지도 객체에 추가 
    '''
    folium.Marker(location = [store['_Y'], store['_X']], popup=store['지점명'],
                            icon=folium.Icon(color='red', icon='star')).add_to(map_cb)
    
map_osm.save('./week05/커비빈매장_주소데이터분석/map_cb.html')  # 완성된 지오맵을 html파일로 저장
