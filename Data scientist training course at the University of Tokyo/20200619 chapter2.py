# 以下のライブラリを使うので、あらかじめ読み込んでおいてください
import numpy as np
import numpy.random as random
import scipy as sp
import pandas as pd
from pandas import Series, Dataframe

#可視化ライブラリ
import matplotlib.pyplot as lot
import matplotlib as mp1
import seaborn as sns
%matplotlib inline

#少数第3位まで表示
%precision 3

#Numpyライブラリの読み込み
inport numpy as np



#少数第3位まで表示という意味
％precision 3

#配列の作成
data = np.array([9,2,3,4,10,6,7,8,1,5])

#seedを設定することで乱数を固定することができる
random.seed(0)

#標準正規分布（平均０、分散１の正規分布）の乱数を10個発生
norm_random_sample_datta = random.randn(10)

print("最小値:",norm_random_sample_data.min())
print("最大値:",norm_random_sample_data.sum())
print("合計:",norm_random_sample_data.sum())

m = np.ones((5,5)),dtype='i')*3
print(m.dot(m))

#chapter2-3
#線形代数用のライブラリ
import scipy.linalg as linalg

#最適化計算（最小値）用の関数
from scipy.optimize import minimize_scalar
from six import X

matrix = np.array([[1,-1,-1],[-1,1,-1],[-1,-1,1]])

#行列式
print('行列式')
print(linalg.det(matrix))

#逆行列
print('逆行列')
print(linalg.inv(matrix))


#固有値と固有ベクトル
eig_value, eig_vector = linalg.eig(matrix)

#固有値と固有ベクトル
print('固有値')
print(eig_value)
print('固有ベクトル')
print(eig_vector)

# 関数の定義
def my_function(x):
    return(x**2 + 2*x + 1)

#ニュートン法の読み込み
from scipy.optimize import newton

#計算実行
print(newton(my_function,0))

 