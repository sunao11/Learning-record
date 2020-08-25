n_list = range(2,10+1)

for i in range(2,int(10 ** 0.5) + 1):
#2,3,...と順に割り切れるかを調べていく
  n_list=[x for x in n_list if (x== i or x % i !=0)]

for j in n_list:
  print(j)

#関数の定義
def calc_prime_num(N):
    n_list = range(2,N,1)

    for i in range(2,int(N**0.5)+1):
        #2,3,...と順に割り切れるかを調べていく
        n_list=[x for x in n_list if (x== i or x % i !=0)]

        for j in n_list:
            print(j)
    #計算実行
    calc_prime_num(10)