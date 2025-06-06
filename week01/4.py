print("04주 연습문제 10번")
print("한빛 가구점에서는 책상이 하루에 1개씩 판매된다. 현재 N개의 책상을 가지고 있고, M일에 한 번씩 도매점으로부터 책상이 1개씩 입고된다." \
"책상의 재고가 0이 될 때 까지 며칠이 걸리는지를 계산하여 반환하는 solution 함수를 작성하시오." \
"단, 착생의 재고가 0이 되는 날이 도매점으로부터 입고되는 날이면 재고가 다시 1개 늘어나므로 결국 그날은 재고가 1인 날이 된다.")

# 입력이 조건에 부합하는지 확인
def check_input():
    N = 0
    M = 0

    # if문이 거짓이면 계속 반복
    while True:
        N = int(input('N = '))  # 재고 입력
        M = int(input('M = '))  # 입고 주기일 입력

        # N이 100 이하의 자연수이고, M이 2보다 크거나 같고, 100보다 작거나 같은 자연수인 경우
        if(0 <= N and 100 >=  N and 2 <= M and 100 >= M):
            return N, M
        print('다시 입력하세요')

# 재고가 0이 되는데 얼마나 걸리는 지 계산
def solution(N, M):
  day = 0
  # 재고가 0이 아닐 때까지 반복복
  while (N > 0):
    day += 1  # 하루 증가
    N -= 1    # 하루 마다 책상 1개씩 판매

    # M일에 한 번 재고가 1개씩 입고
    if(day % M == 0):
       N += 1
   
  return day  # 재고가 0이 되는 날 수 반환

# 함수 호출
N, M = check_input()
print("현재 %d개의 재고가 있고, %d일 마다 책상이 1개씩 입고되면 " \
"재고가 0이 되는데 %d일이 걸립니다." %(N, M, solution(N, M)))


