{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 以下のライブラリを使うので、あらかじめ読み込んでおいてください\n",
    "import numpy as np\n",
    "import numpy.random as random\n",
    "import scipy as sp\n",
    "import pandas as pd\n",
    "from pandas import Series, Dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#可視化ライブラリ\n",
    "import matplotlib.pyplot as lot\n",
    "import matplotlib as mp1\n",
    "import seaborn as sns\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#少数第3位まで表示\n",
    "%precision 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Numpyライブラリの読み込み\n",
    "inport numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#少数第3位まで表示という意味\n",
    "％precision 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#配列の作成\n",
    "data = np.array([9,2,3,4,10,6,7,8,1,5])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#seedを設定することで乱数を固定することができる\n",
    "random.seed(0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#標準正規分布（平均０、分散１の正規分布）の乱数を10個発生\n",
    "norm_random_sample_datta = random.randn(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"最小値:\",norm_random_sample_data.min())\n",
    "print(\"最大値:\",norm_random_sample_data.sum())\n",
    "print(\"合計:\",norm_random_sample_data.sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "m = np.ones((5,5)),dtype='i')*3\n",
    "print(m.dot(m))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#chapter2-3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#線形代数用のライブラリ\n",
    "import scipy.linalg as linalg"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#最適化計算（最小値）用の関数\n",
    "from scipy.optimize import minimize_scalar\n",
    "from six import X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "matrix = np.array([[1,-1,-1],[-1,1,-1],[-1,-1,1]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#行列式\n",
    "print('行列式')\n",
    "print(linalg.det(matrix))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#逆行列\n",
    "print('逆行列')\n",
    "print(linalg.inv(matrix))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#固有値と固有ベクトル\n",
    "eig_value, eig_vector = linalg.eig(matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#固有値と固有ベクトル\n",
    "print('固有値')\n",
    "print(eig_value)\n",
    "print('固有ベクトル')\n",
    "print(eig_vector)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 関数の定義\n",
    "def my_function(x):\n",
    "    return(x**2 + 2*x + 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#ニュートン法の読み込み\n",
    "from scipy.optimize import newton"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#計算実行\n",
    "print(newton(my_function,0))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": 3
  },
  "orig_nbformat": 2
 },
 "nbformat": 4,
 "nbformat_minor": 2
}