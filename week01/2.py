import pandas as pd
import matplotlib.pyplot as plt

print("04주 연습문제 8번")

# csv 파일 불러오기 (index_col을 통해 index를 제외하고 불러옴)
df = pd.read_csv('./week01/연습문제1.csv', index_col = 0)

# 열 이름 변경
df.columns = ['first', 'second', 'third', 'fourth']
# 데이터 전치 (행과 열을 바꿔서 x축에는 분기, y축에는 분기 매출 표시)
df.T.plot()

# 그래프 제목, x축/y축 라벨 설정
plt.title('2015-2020 Quarterly sales')
plt.xlabel('Quarters')
plt.ylabel('sales')

# 화면에 그래프 표시
plt.show()




