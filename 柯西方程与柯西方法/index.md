# 柯西方程与柯西方法


**形式二**：为证明$f(a+b)=f(a)*f(b)$存在唯一实数解$f(x)=a^x$

### 证明

设n为正整数，则

$$
\begin{aligned}
	&f(n+1)=f(n)*f(1) \cr
	f(n)&=f(n-1)*f(1)\cr
	&=f(n-2)*f(1)^2 \cr
	&=\cdots \cr
	&=f(1)^n
\end{aligned}
$$

所以 **正整数** 成立。

由题干性质 $ f(0+0)=f(0)*f(0) $ 可知，$f(0)= 1 \ or \ 0$ ，显然为0时，整个函数为0常函数，结论平凡。以下默认$f(0)=1$ .

所以

$$
\begin{gathered}
	f(0)=f(n+(-n))=f(n)*f(-n) \cr
	f(-n)=f(n)^{-1}=f(1)^{-n}
\end{gathered}
$$

即对 **整数** 成立。

设n,m为整数

$$
\begin{aligned}
	f(n)&=f(\underbrace { \frac{1}{n}+\frac{1}{n}+, \cdots, +  \frac{1}	{n} }_{n^2}) \cr
	&=f(\frac{1}{n})^{n^2}
\end{aligned}
$$

所以

$$
\begin{aligned}
	f(\frac{1}{n})=f(n)^{\frac{1}{n^2}}=[f(1)^n]^{\frac{1}	{n^2}}=f(1)^{\frac{1}{n}}
\end{aligned}
$$

则

$$
\begin{aligned}
	f(\frac{m}{n})=f(\underbrace { \frac{1}{n} + \frac{1}{n} + ,\cdots , +  \frac{1}{n} }_{m})=f(\frac{1}{n})^m=f(1)^{\frac{m}{n}}
\end{aligned}
$$

任意有理数可表示为$\frac{m}{n}$ 的形式，可知结果对 **有理数** 成立。

如满足下列条件其一

1. f 连续；

2. f 在一个区间上单调；

3. f 在一个区间上有上界或下界。

易证以上结论在 **实数域** 上成立。

### 结论

综上，$f(a+b)=f(a)*f(b)$存在唯一实数解$f(x)=a^x$

参考：[https://zhuanlan.zhihu.com/p/80543711](https://)

$$
\begin{aligned} 
	--1--
\end{aligned}
$$

