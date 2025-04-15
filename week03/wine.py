'''
목표: 와인의 속성을 분석한 뒤 품질 등급 예측
[목표를 위한 작업]
- 데이터에 대한 기술 통계
- 레드 와인과 화이트 와인 크룹의 품질에 대한 t-검정 수행
- 와인 속성을 독립 변수로, 품질 등급을 종속 변수로 선형 회귀 분석 수행
'''
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# 와인 데이터 로드 및 모델링 함수
def process_and_model_wine_data():
    # 레드 와인과 화이트 와인 데이터를 읽어옵니다.
    red_df = pd.read_csv('./python/week03/wine_data/winequality-red.csv', sep=';', header=0, engine='python')
    white_df = pd.read_csv('./python/week03/wine_data/winequality-white.csv', sep=';', header=0, engine='python')

    # 정리된 데이터를 각각 저장합니다.
    red_df.to_csv('./python/week03/wine_data/winequality-red2.csv', index=False)
    white_df.to_csv('./python/week03/wine_data/winequality-white2.csv', index=False)

    # 레드 와인 데이터 확인 및 정리
    print(red_df.head())   # 상위 5개의 행 출력
    red_df.insert(0, column='type', value='red') # 첫 번째 열에 'type' 컬럼 추가 후 값을 'red'로 설정
    print(red_df.head())
    print(red_df.shape)    # 행과 열 개수 확인

    # 화이트 와인 데이터 확인 및 정리
    print(white_df.head())   # 상위 5개의 행 출력
    white_df.insert(0, column='type', value='white') # 첫 번째 열에 'type' 컬럼 추가 후 값을 'white'로 설정
    print(white_df.head())
    print(white_df.shape)    # 행과 열 개수 확인

    # 레드 & 화이트 와인 데이터 병합하여 csv 파일로 저장
    wine = pd.concat([red_df, white_df])
    print(wine.shape)
    wine.to_csv('./python/week03/wine_data/wine.csv', index=False)

    # 와인 데이터 탐색
    wine = pd.read_csv('./python/week03/wine_data/wine.csv')
    wine.columns = wine.columns.str.replace(' ', '_')  # 컬럼명에 공백이 있으면 '_'로 변환(fixed acidity -> fixed_acidity)

    # 와인 데이터의 기술 통계 및 품질 정보 출력
    print(wine.describe()) # 컬럼별 통계 정보 확인
    print(sorted(wine.quality.unique().tolist()))  # quality 속성값 중에서 유일한 값 출력
    print(wine.quality.value_counts())  # quality 속성값에 대한 빈도수 출력

    # 와인 데이터 모델링
    print(wine.groupby('type')['quality'].describe())  # 레드 & 화이트 와인의 품질 비교
    print(wine.groupby('type')['quality'].mean())  # 품질의 평균값 출력
    print(wine.groupby('type')['quality'].std())  # 품질의 표준편차 출력
    print(wine.groupby('type')['quality'].agg(['mean', 'std']))  # 평균과 표준편차를 동시에 출력

    # 레드 와인 품질 값과 화이트 와인 품질 값 추출
    red_wine_quality = wine.loc[wine['type'] == 'red', 'quality']  # 레드 와인 품질 값 저장
    white_wine_quality = wine.loc[wine['type'] == 'white', 'quality']  # 화이트 와인 품질 값 저장

    # t-검정을을 수행하여 품질에 차이가 있는지 확인
    print(stats.ttest_ind(red_wine_quality, white_wine_quality, equal_var=False))  # t-검정 결과 출력

    # 선형 회귀 분석
    Rformula = 'quality ~ fixed_acidity + volatile_acidity + citric_acid + \
            residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + \
            density + pH + sulphates + alcohol'  # 회귀 분석에 사용할 수식 정의
    regression_result = ols(Rformula, data=wine).fit()  # 선형 회귀 분석 수행
    print(regression_result.summary())  # 회귀 분석 결과 출력

    # 선형 회귀 분석 모델로 새로운 샘플의 품질 등급 예측하기
    print('------ 선형 회귀 분석 모델로 새로운 샘플의 품질 등급 예측하기 ------')
    sample1 = wine[wine.columns.difference(['query', 'type'])]  # 품질과 타입을 제외한 나머지 컬럼 저장
    sample1 = sample1[0:5][:]  # 5개의 샘플 선택
    sample1_predict = regression_result.predict(sample1)  # 품질 예측
    print(sample1_predict)  # 예측 결과 확인
    print(wine[0:5]['quality'])  # 실제 품질 값을 출력하여 비교

    # 새로운 샘플 생성
    data = {"fixed_acidity": [8.5, 8.1], "volatile_acidity": [0.8, 0.5],
            "citric_acid": [0.3, 0.4], "residual_sugar": [6.1, 5.8], "chlorides": [0.055, 0.04],
            "free_sulfur_dioxide": [30.0, 31.0], "total_sulfur_dioxide": [98.0, 99],
            "density": [0.996, 0.91], "pH": [3.25, 3.01], "sulphates": [0.4, 0.35], "alcohol": [9.0, 0.88]}
    sample2 = pd.DataFrame(data, columns=sample1.columns)  # 새로운 데이터 프레임 생성
    print(sample2)
    sample2_predict = regression_result.predict(sample2)  # 품질 예측
    print(sample2_predict)  # 예측 결과 확인

    # 와인 유형에 따른 품질 등급 히스토그램 그리기
    sns.set_style('dark')  # seaborn의 스타일을 'dark'로 설정
    sns.histplot(red_wine_quality, stat='density', kde=True, color="red", label='red wine')  # 레드 와인의 품질 히스토그램
    sns.histplot(white_wine_quality, stat='density', kde=True, label='white wine')  # 화이트 와인의 품질 히스토그램
    plt.title("Quality of Wine Type")
    plt.legend()
    plt.show()

    # 부분 회귀 플롯으로 시각화 하기
    others = list(set(wine.columns).difference(set(['quality', 'fixed_acidity'])))  # 'quality'와 'fixed_acidity'를 제외한 나머지 독립 변수들의 리스트 생성
    p, resids = sm.graphics.plot_partregress('quality', 'fixed_acidity', others, data=wine, ret_coords=True)  # 부분회귀 플롯 수행
    plt.show()

    # 각 독립 변수의 부분 회귀 플롯 구하기
    fig = plt.figure(figsize=(8, 13))
    sm.graphics.plot_partregress_grid(regression_result, fig=fig)  # 각 독립 변수의 부분 회귀 플롯 구하기
    plt.show()

# main 함수
def main():
    process_and_model_wine_data()
    print('by BD 오후 컴퓨터소트프웨어학과 21101849 이지은')


# 프로그램 실행
if __name__ == '__main__':
    main()
    