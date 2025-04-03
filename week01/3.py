print("04주 연습문제 9번")
print("한빛 식당에는 메인 메뉴와 토핑을 1개씩 선택할 수 있다. 주문할 수 있는 모든 메뉴를 출력하는 set_menu 함수를 완성하시오." \
"단 냉면에 치즈 토핑은 선택할 수 없다")

def set_menu():
    main = ['라면', '공깃밥', '돈까스', '냉면']
    topping = ['계란', '치즈']

    menu = []

    for m in main:
        for t in topping:
            # 냉면과 치즈 조합이 아닌 경우에만 menu에 저장함
            if not(m == '냉면' and t == '치즈'):
                # 데이터를 튜플 형태로 저장
                menu.append((m, t))
    return menu

print(set_menu())
