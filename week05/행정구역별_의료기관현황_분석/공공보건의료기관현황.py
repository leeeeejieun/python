import pandas as pd
pd.set_option('mode.chained_assignment', None)
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import rcParams, style
style.use('ggplot')

plt.rc('font', family = 'Malgun Gothic') # 한글 폰트 설정

# 공공보건의료기관 현황 데이터 가져오기
data = pd.read_csv('./week05/행정구역별_의료기관현황_분석/공공보건의료기관현황.csv', 
                    index_col=0, encoding='cp949', engine='python')

# 주소에서 시도, 군구 정보 분리
addr = pd.DataFrame(data['주소'].apply(lambda v: v.split()[:2]).
                    tolist(), columns=('시도', '군구'))

# 확인 결과 행정구역 주소 체계에 맞게 잘 저장되어 있음
print(addr['시도'].unique()) # 시도 컬럼 값에서 고유값 확인 
print(addr['군구'].unique())   # 군구 컬럼 값에서 고유값 확인

addr['시도군구'] = addr.apply(lambda r: r['시도'] + ' ' + r['군구'], axis=1)  # 시도, 군구 컬럼값을 결합하여 새로운 컬럼으로 추가
addr['count'] = 0   # 행정구역별 공공보건의료기관 수를 저장하기 위한 컬럼 추가

addr_group = pd.DataFrame(addr.groupby(['시도', '군구', '시도군구'], as_index=False).count()) # 시도, 군구, 시도군구 컬럼을 기준으로 그룹을 만든 후 각 그룹별 원소 개수 저장
addr_group = addr_group.set_index('시도군구')  # 시도군구 컬럼을 데이터프레임 병합에 사용할 인덱스로 설정

population = pd.read_excel('./week05/행정구역별_의료기관현황_분석/행정구역_시군구_별__성별_인구수.xlsx') # 행정구역별 인구수 데이터 가져오기
population.rename(columns={'행정구역(시군구)별(1)': '시도', '행정구역(시군구)별(2)': '군구'}, inplace=True)  # 컬럼명 변경

# 군구 컬럼의 문자열 앞뒤에 포함된 공백 제거
for i in range(len(population)):
    population.loc[i, '군구'] = population.loc[i, '군구'].strip()

population['시도군구'] = population.apply(lambda r: r['시도'] + ' ' + r['군구'], axis=1)  # 시도, 군구 컬럼값을 결합하여 새로운 컬럼으로 추가
population = population[population.군구 != '합계']  # 컬럼 값이 '합계'인 행은 제거
population = population[population.군구 != '소계']  # 컬럼 값이 '소계'인 행은 제거
population = population.set_index('시도군구')  # 시도군구 컬럼을 데이터프레임 병합에 사용할 인덱스로 설정

addr_population_merge = pd.merge(addr_group, population, how='inner', left_index=True, right_index=True)  # 인덱스를 기준으로 두 개의 데이터프레임을 병합
local_mc_population = addr_population_merge[['시도_x', '군구_x', 'count', '총인구수 (명)']]   # 필요한 컬럼만 추출하여 저장
local_mc_population.rename(columns={'시도_x': '시도', '군구_x': '군구', '총인구수 (명)': '인구수'}, inplace=True)  # 컬럼명 변경

mc_count = local_mc_population['count']
local_mc_population['mc_ratio'] = mc_count.div(local_mc_population['인구수'], axis=0) * 100000  # 인구수 대비 공공보건의료기관 비율(단위수: 10만명) 계산 후 새로운 컬럼에 추가

mc_ratio = local_mc_population[['count']]   # 행정구역별 공공보건의료기관 수 추출
mc_ratio = mc_ratio.sort_values('count', ascending=False)   # 내림차순 정렬
plt.rcParams['figure.figsize'] =  (25, 5)  # 그래프 크기 설정(가로 25, 세로 5)
mc_ratio.plot(kind='bar', rot=90)    # 막대 그래프 그리기, x축 레이블 90도 회전
plt.xticks(range(0, len(mc_ratio), 2))  # x축 레이블 간격을 2칸씩 띄움
plt.subplots_adjust(bottom=0.3)  # 아래쪽 여백을 넓혀서 x축 레이블이 잘리지 않게 함
plt.title('행정구역별 공공의료기관 수')  # 그래프 제목 설정
plt.show()  # 그래프 출력

mc_ratio = local_mc_population[['mc_ratio']]  # 행정구역별 인구수 대비 공공의료기관 비율 추출
mc_ratio = mc_ratio.sort_values('mc_ratio', ascending=False)   # 내림차순 정렬
plt.rcParams['figure.figsize'] =  (25, 5)  # 그래프 크기 설정(가로 25, 세로 5)
mc_ratio.plot(kind='bar', rot=90)    # 막대 그래프 그리기, x축 레이블 90도 회전
plt.xticks(range(0, len(mc_ratio), 2))  # x축 레이블 간격을 2칸씩 띄움
plt.subplots_adjust(bottom=0.3)  # 아래쪽 여백을 넓혀서 x축 레이블이 잘리지 않게 함
plt.title('행정구역별 인구수 대비 공공보건의료기관 비율')  # 그래프 제목 설정
plt.show()  # 그래프 출력

data_draw_korea = pd.read_csv('./week05/행정구역별_의료기관현황_분석/data_draw_korea.csv',
                              index_col=0, encoding='utf-8', engine='python')

# 2023년 6월에 변경된 ‘강원특별자치도’와 ‘전북특별자치도’ 행정구역명 수정하기
addr_aliases = {'강원도': '강원특별자치도', '전라북도': '전북특별자치도'}  
data_draw_korea['광역시도'] = data_draw_korea['광역시도'].apply(lambda v : addr_aliases.get(v, v))
data_draw_korea['시도군구'] = data_draw_korea.apply(lambda r: r['광역시도'] + ' ' + r['행정구역'], axis=1) 
data_draw_korea = data_draw_korea.set_index('시도군구')

data_draw_korea_MC_Population_all = pd.merge(data_draw_korea,local_mc_population,  how='outer',  left_index=True, right_index=True) # 외부 병합
data_draw_korea_MC_Population_all = data_draw_korea_MC_Population_all.fillna(0) # 결측값 처리

# 한국지도의 블록맵 경계선 좌표를 리스트로 생성
BORDER_LINES = [
    [(3, 2), (5, 2), (5, 3), (9, 3), (9, 1)], # 인천
    [(2, 5), (3, 5), (3, 4), (8, 4), (8, 7), (7, 7), (7, 9), (4, 9), (4, 7), (1, 7)], # 서울
    [(1, 6), (1, 9), (3, 9), (3, 10), (8, 10), (8, 9),
     (9, 9), (9, 8), (10, 8), (10, 5), (9, 5), (9, 3)], # 경기도
    [(9, 12), (9, 10), (8, 10)], # 강원도
    [(10, 5), (11, 5), (11, 4), (12, 4), (12, 5), (13, 5),
     (13, 4), (14, 4), (14, 2)], # 충청남도
    [(11, 5), (12, 5), (12, 6), (15, 6), (15, 7), (13, 7),
     (13, 8), (11, 8), (11, 9), (10, 9), (10, 8)], # 충청북도
    [(14, 4), (15, 4), (15, 6)], # 대전시
    [(14, 7), (14, 9), (13, 9), (13, 11), (13, 13)], # 경상북도
    [(14, 8), (16, 8), (16, 10), (15, 10),
     (15, 11), (14, 11), (14, 12), (13, 12)], # 대구시
    [(15, 11), (16, 11), (16, 13)], # 울산시
    [(17, 1), (17, 3), (18, 3), (18, 6), (15, 6)], # 전라북도
    [(19, 2), (19, 4), (21, 4), (21, 3), (22, 3), (22, 2), (19, 2)], # 광주시
    [(18, 5), (20, 5), (20, 6)], # 전라남도
    [(16, 9), (18, 9), (18, 8), (19, 8), (19, 9), (20, 9), (20, 10)], # 부산시
]

# 블록맵으로 시각화하기
def draw_blockMap(blockedMap, targetData, title, color ):
    whitelabelmin = (max(blockedMap[targetData]) - min(blockedMap[targetData])) * 0.25 + min(blockedMap[targetData])

    datalabel = targetData

    vmin = min(blockedMap[targetData])
    vmax = max(blockedMap[targetData])

    mapdata = blockedMap.pivot(index='y', columns='x', values=targetData)
    masked_mapdata = np.ma.masked_where(np.isnan(mapdata), mapdata)
    
    plt.figure(figsize=(8, 13))
    plt.title(title)
    plt.pcolor(masked_mapdata, vmin=vmin, vmax=vmax, cmap=color, edgecolor='#aaaaaa', linewidth=0.5)
    
    # 지역 이름 표시
    for idx, row in blockedMap.iterrows():
        annocolor = 'white' if row[targetData] > whitelabelmin else 'black'
        
        # 광역시는 구 이름이 겹치는 경우가 많아서 시단위 이름도 같이 표시 
        if isinstance(row['광역시도'], str) and row['광역시도'].endswith('시') and not row['광역시도'].startswith('세종'):
            dispname = '{}\n{}'.format(row['광역시도'][:2], row['행정구역'][:-1] if isinstance(row['행정구역'], str) else row['행정구역'])
            if len(row['행정구역']) <= 2:
                dispname += row['행정구역'][-1] if isinstance(row['행정구역'], str) else ''
        else:
            dispname = row['행정구역'][:-1] if isinstance(row['행정구역'], str) else row['행정구역']

        # dispname이 문자열인지 확인하고 splitlines() 적용
        if isinstance(dispname, str) and len(dispname.splitlines()[-1]) >= 3:
            fontsize, linespacing = 9.5, 1.5
        else:
            fontsize, linespacing = 11, 1.2

        plt.annotate(dispname, (row['x']+0.5, row['y']+0.5), weight='bold',
                    fontsize=fontsize, ha='center', va='center', color=annocolor,
                    linespacing=linespacing)

    # 시도 경계 그리기
    for path in BORDER_LINES:
        ys, xs = zip(*path)
        plt.plot(xs, ys, c='black', lw=4)

    plt.gca().invert_yaxis()
    plt.axis('off')

    cb = plt.colorbar(shrink=.1, aspect=10)
    cb.set_label(datalabel)

    plt.tight_layout()
    plt.savefig('./week05/행정구역별_의료기관현황_분석/' + 'blockMap_' + targetData + '.png')
    plt.show()

draw_blockMap(data_draw_korea_MC_Population_all, 'count', '행정구역별 공공보건의료기관 수', 'Blues')
draw_blockMap(data_draw_korea_MC_Population_all, 'mc_ratio', '행정구역별 인구수 대비 공공보건의료기관 비율', 'Reds' )