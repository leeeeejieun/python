'''
[목표 설정정]
- 타이타닉호의 생존자와 관련된 변수의 상관관계 찾아보기
- 생존과 가장 상관도가 높은 변수는 무엇인지 분석
- 상관 분석을 위해 피어슨 상관 계수를 사용
- 변수 간의 상관관계는 시각화하여 분석
'''
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def process_and_model_titanic_data():
    # 1. 데이터 준비
    titanic = sns.load_dataset("titanic")  # 타이타닉 데이터 로드
    titanic.to_csv('./python/week03/titanic.csv', index = False)  # csv 파일에 저장

    print(titanic.isnull().sum())  # 결측치가 있는 열과 그 개수를 출력
    titanic['age'] = titanic['age'].fillna(titanic['age'].median()) # 'age' 열의 결측치를 해당 열의 중앙값으로 채움
    print(titanic['embarked'].value_counts())  # 'embarked' 열의 각 값이 몇 개인지 출력
    titanic['embarked'] = titanic['embarked'].fillna('S')  # 'embarked' 열의 결측치를 가장 많이 나타나는 값 'S'로 채움
    print(titanic['embark_town'].value_counts())  # 'embark_town' 열의 각 값이 몇 개인지 출력
    titanic['embark_town'] = titanic['embark_town'].fillna('Southampton')  # 'embark_town' 열의 결측치를 'Southampton'으로 채움
    print(titanic['deck'].value_counts())  # 'deck' 열의 각 값이 몇 개인지 출력
    titanic['deck'] = titanic['deck'].fillna('C') # 'deck' 열의 결측치를 임의로 'C'로 채움
    print(titanic.isnull().sum())# 결측치가 모두 잘 처리됐는지 확인


    # 2. 데이터 탐색
    titanic.info()   # 기본 정보 확인

    # 차트를 그려 데이터를 시각적으로 탐색하기
    f, ax = plt.subplots(1, 2, figsize = (10, 5))  # 한 줄에 두 개의 차트를 그리도록 하고 크기를 설정
    # 첫 번째 pie 차트는 남자 승객의 생존율을 나타내도록 설정
    titanic['survived'][titanic['sex'] == 'male'].value_counts().plot.pie(explode = [0,0.1], autopct = '%1.1f%%', ax = ax[0], shadow = True)
    # 두 번째 pie 차트는 여자 승객의 생존율을 나타내도록 설정
    titanic['survived'][titanic['sex'] == 'female'].value_counts().plot.pie(explode = [0,0.1], autopct = '%1.1f%%', ax = ax[1], shadow = True)
    # 차트 제목 설정
    ax[0].set_title('Survived (Male)')
    ax[1].set_title('Survived (Female)')
    plt.show()

    # 3. 등급별 생존자 수를 차트로 나타내기
    # pclass 유형 1,2,3을 x축으로 하고 survived =0과 survived =1의 개수를 계산하여 y축으로 하는 countplot을 설정
    sns.countplot(x='pclass', hue = 'survived', data = titanic)
    plt.title('Pclass vs Survived')
    plt.show()

    # 3. 데이터 모델링
    
    titanic2 = titanic.select_dtypes(include=[int, float,bool])  # 자료형(dtype)이 int, float, boolean 인 것만 추출해서 tianic2에 저장
    print(titanic2.shape)
    titanic_corr = titanic2.corr(method = 'pearson')  # 피어슨 상관 계수를 적용하여 상관 계수를 구함
    print(titanic_corr)
    titanic_corr.to_csv('./python/week03/titanic_corr.csv', index = False)
   
    # 특정 변수 사이의 상관 계수 구하기
    print('특정 변수 사이의 상관 계수 구하기')
    print(titanic['survived'].corr(titanic['adult_male']))  # survived와 adult_male 변수 사이의 상관 계수 구하기
    print( titanic['survived'].corr(titanic['fare']))   # survived와 fare 변수 사이의 상관 계수 구하기

    # 4. 결과 시각화 

    # 산점도로 상관 분석 시각화하기
    sns.pairplot(titanic, hue = 'survived')  # pairplot() 함수를 사용하여 타이타닉 데이터의 차트 그리기. hue는 종속 변수를 지정
    plt.show()  

    # 생존자의 객실 등급과 성별 관계를 catplot()으로 그리기
    sns.catplot(x = 'pclass', y = 'survived', hue = 'sex', data = titanic, kind = 'point')
    plt.show()

    # 변수 사이의 상관 계수를 히트맵으로 시각화 하기
    titanic['age2'] = titanic['age'].apply(category_age)    # category_age 함수를 적용하여 새로운 age2 열을 만들어 추가
    titanic['sex'] = titanic['sex'].map({'male': 1, 'female': 0})  # 성별을 male/female에서 1/0으로 치환
    titanic['family'] = titanic['sibsp'] + titanic['parch'] + 1    # 가족의 수를 구하여 family 열을 추가
    titanic.to_csv('./python/week03/titanic2.csv', index=False)    # 수정된 데이터프레임을 titanic2.csv로 저장

    heatmap_data = titanic[['survived', 'sex', 'age2', 'family', 'pclass', 'fare']]  # 히트맵에 사용할 데이터를 추출
    colormap = plt.cm.RdBu   # 히트맵에 사용할 색상맵을 지정
    # corr() 함수로 구한 상관 계수로 히트맵을 생성
    sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True,
                cmap=colormap, linecolor='white', annot=True, annot_kws={'size': 10})
    plt.show()


# 10살 단위로 등급을 나누어 0~7의 값으로 바꿔주는 함수
def category_age(x):
    if x < 10:
        return 0
    elif x < 20:
        return 1
    elif x < 30:
        return 2
    elif x < 40:
        return 3
    elif x < 50:
        return 4
    elif x < 60:
        return 5
    elif x < 70:
        return 6
    else:
        return 7

def main():
    process_and_model_titanic_data()
    print('by BD 오후 컴퓨터소트프웨어학과 21101849 이지은')
    
# 프로그램 실행
if __name__ == '__main__':
    main()
    