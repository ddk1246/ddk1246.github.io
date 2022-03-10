# 主成分分析


### 协方差矩阵对角化

设一点集
$$
D= \left[\begin{matrix}
    x_1,y_1 \cr
    x_2,y_2\cr
    ...\cr
    x_n,y_n
    \end{matrix}\right]^T
$$
协方差公式为：

$$
\begin{aligned}
	cov(x,y)=\frac{ \sum_i^n{(x_i-\bar x)(y_i-\bar y)}}{n-1}
\end{aligned}
$$
去中心化后$\bar x,\bar y$为0，（1）式可化为

$$
\begin{gathered}
	cov(x,y)=\frac{ \sum_i^n{x_iy_i}}{n-1}
\end{gathered}
$$
协方差矩阵为

$$
C = \left(\begin{matrix}
		cov(x,x)&cov(x,y)\cr
		cov(y,x)&cov(y,y)
	\end{matrix}\right)
$$


$$C = \frac{1}{n-1}\cdot DD^T$$

$D$为原始数据分布，$D_s$为 PCA 后其对应的分布，则存在旋转（R）、拉伸（S）矩阵，使得$D=RSD_s$,其中

$$
R=\left(\begin{matrix} 
		cos(\theta)&-sin(\theta) \cr
		sin(\theta)&cos(\theta)
	\end{matrix}\right),
S=\left(\begin{matrix}
		a&\cr &b
	\end{matrix}\right)
$$

所以

$$
\begin{aligned} 
	 C^`&=\frac{1}{n-1} \cdot DD^{T}= \frac{1}{n-1}\cdot RSD_s \cdot (RSD_s)^T \cr
		&=RS\cdot (\frac{1}{n-1}D_sD_s^T)\cdot S^TR^T \cr
		&=RSCS^TR^T=RSS^TR^T \cr
		&=RLR^T \cr
\end{aligned}
$$

其中

$$
R=
\left( \begin{matrix} 
	cos(\theta) & -sin(\theta) \cr
	sin(\theta) & cos(\theta)
\end{matrix} \right),
L=
\left(\begin{matrix} 
	a^2 & \cr 
	& b^2
\end{matrix}\right)
$$

可见L矩阵为对角阵，R矩阵为正交矩阵$R\cdot R^T=E$,求R与L相当于对 **协方差矩阵对角化**。

```python
import numpy as np
data = np.random.rand(10,2) #二维点集 n*d
cov = np.cov(data.T)        #d*d
w, v = np.linalg.eig(cov)  #此处对特征向量做了单位化，即列向量模长为1
#cov = v@np.diag(w)@v.T
indiex = np.argsort(-w)
v = v[:,indiex]
w = w[indiex]
```

#### 正交化证明1

$$
\begin{aligned}
  	MM^T &= v * w * v^{-1} \cr
	MM^T &= v^{-T} * w * v^T \cr
	v^{-T} * w * v^T &= v * w * v^{-1} \cr
	w * v^T * v &= v^T * v * w \cr
	w * N &= N * w \cr
	a * N[:,0] &= N[0,:] * a \cr
\end{aligned}
$$
所以N只有对角元素上有值，即特征向量两两正交。**但v不一定是酉矩阵**，因为没做单位化。而numpy的算法做了单位化处理，具有特异性。

#### 正交化证明2

n维实对称矩阵S，用$\lambda ,\alpha$表示其两个不等的特征值，用$x,y$分别表示其对应的特征向量。$S=S^T,Sx=\lambda x,Sy=\alpha y(\alpha \neq \lambda)$.

对$Sx=\lambda x$两边转置右端成$y$

$$
\begin{aligned}
	x^TS^T &= \lambda x^T\cr
	 x^TS  &= \lambda x^T\cr
	x^TSy  &= \lambda x^Ty
\end{aligned}
$$

对$Sy=\alpha y$两段左乘$x^T$

$$x^TSy=x^T\alpha y=\alpha x^Ty$$

所以

$$
\begin{aligned}
\alpha x^Ty &=\lambda x^Ty \cr
		  0 &=(\alpha-\lambda)x^Ty
\end{aligned}
$$

已知$\alpha \neq \lambda$,所以$x^Ty=0$,即不同特征值的特征向量两两正交.

<br/>

### PCA投影

维度，k为特征值占比85%的索引数 	$D^T: n * d, C: d * d, v: d * d, u: d * k$（v是特征向量矩阵，u包含了特征向量对应特征值占比为前85%的列索引）

投影，$u$的每一列是单位化后的**方向向量**$|u_{*j}|=1$，点积相当于cos投影：

对D任意一行与u任意一列：
$$
\vec D^T_{i*} \cdot \vec u_{j}=|D^T_{i}|\cdot |u_{j}| \cdot cos(\theta)=|D^T_{i}|\cdot cos(\theta)(投影值)
$$

$D^T \cdot u:n*k$

$s^2=\frac{\sum x_i^2}{n-1}=\frac{u^TDD^Tu}{n-1}=u^TCu=\lambda$

证明如公式3，PCA后的方差是矩阵特征值。方差越大，数据含有的信息越多，数据信号越强

```python
k = 1
Q = v[:,:k]
data_norm = data - data.mean(0) # normalization
Y = np.matmul(data_norm,Q) #n*d @ d*k = n*k
data_ = np.matmul(Y,Q.T)+data.mean(0) # n*k @ k*d = n * d
# 
plt.scatter(data_[:,0],data_[:,1])
```

<br/>

### OBB包围盒

todo 

