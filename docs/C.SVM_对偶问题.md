# 附录 C、SVM 对偶问题

为了理解对偶性，你首先得理解拉格朗日乘子法。它基本思想是将一个有约束优化问题转化为一个无约束优化问题，其方法是将约束条件移动到目标函数中去。让我们看一个简单的例子，例如要找到合适的 ![x](../images/tex-9dd4e461268c8034f5c8564e155c67a6.gif) 和 ![y](../images/tex-415290769594460e2e485922904f345d.gif) 使得函数 ![f(x, y) = x^2 + 2y](../images/tex-ec2b11c28f337d90d1a55c83bd738475.gif) 最小化，且其约束条件是一个等式约束：![3x + 2y + 1 = 0](../images/tex-08bc7e7224cfe0e39e04b69d4ed96298.gif)。使用拉格朗日乘子法，我们首先定义一个函数，称为**拉格朗日函数**：![g(x, y, \alpha) = f(x, y) - \alpha(3x + 2y + 1)](../images/tex-e78acaa101dc94594e813eab3f01f428.gif)。每个约束条件（在这个例子中只有一个）与新的变量（称为拉格朗日乘数）相乘，作为原目标函数的减数。

Joseph-Louis Lagrange 大牛证明了如果 ![(\bar{x}, \bar{y})](../images/tex-fe85b05b6cd2641c29612bc75a270208.gif) 是原约束优化问题的解，那么一定存在一个 ![\bar{\alpha}](../images/tex-d9c29791dd3b792c7702ed2b7cf5ac40.gif)，使得 ![(\bar{x}, \bar{y}, \bar{\alpha})](../images/tex-ac2d6cc9cbad11acc20ba9f6dd0ef830.gif) 是拉格朗日函数的驻点（驻点指的是，在该点处，该函数所有的偏导数均为 0）。换句话说，我们可以计算拉格朗日函数 ![g(x, y, \alpha) ](../images/tex-c663f5d534674fc3f1b13074c6ae467b.gif) 关于 ![x, y](../images/tex-2317793a8de61ab32c0f17adff9ea8d4.gif) 以及 ![\alpha](../images/tex-7b7f9dbfea05c83784f8b85149852f08.gif) 的偏导数；然后我们可以找到那些偏导数均为 0 的驻点；最后原约束优化问题的解（如果存在）一定在这些驻点里面。

在上述例子里，偏导数为

![\begin{align}\frac{\partial}{\partial x}g(x, y, \alpha) = 2x - 3\alpha \\ \frac{\partial}{\partial y}g(x, y, \alpha) = 2 - 2\alpha \\ \frac{\partial}{\partial \alpha}g(x, y, \alpha) = -3x - 2y - 1 \end{align}](../images/tex-96d88ee52e2a53c2350376ac3b1f3c30.gif)  

当这些偏导数均为 0 时，即 ![2x − 3\alpha = 2 − 2\alpha = − 3x − 2y − 1 = 0](../images/tex-f74dbceea979dd6f7f807de601aaa240.gif)，即可得 ![x = \frac{3}{2}, y=-\frac{11}{4}, \alpha=1](../images/tex-9439cb7cb2e1c01f22745401287a0638.gif)。这是唯一一个驻点，那它一定是原约束优化问题的解。然而，上述方法仅应用于等式约束，幸运的是，在某些正则性条件下，这种方法也可被一般化应用于不等式约束条件（例如不等式约束，![3x + 2y + 1 \geq 0](../images/tex-3b6ed467ca1ae09aeafe40f4b40251c7.gif)）。如下公式 C-1 ，给了 SVM 硬间隔问题时的一般化拉格朗日函数。在该公式中，![\alpha^{(i)}](../images/tex-99f0c7b568236eb0a52bf15cbbfa342e.gif) 是 KKT 乘子，它必须大于或等于 0。

> 译者注
> 
> ![\alpha^{(i)}](../images/tex-99f0c7b568236eb0a52bf15cbbfa342e.gif) 是 ![\geq0](../images/tex-e56717342e6431bdaa1f37c90f7ba7b3.gif) 抑或 ![\leq0](../images/tex-825a000824ab58528de14389acafd231.gif)，取决于拉格朗日函数的写法，以及原目标函数函数最大化抑或最小化。

![公式C-1](../images/Appendix/E_C-1.png)

就像拉格朗日乘子法，我们可以计算上述式子的偏导数、定位驻点。如果该原问题存在一个解，那它一定在驻点 ![(\bar{w}, \bar{b}, \bar{\alpha})](../images/tex-e0569dbaecb2e5955e7ff0bad0749154.gif) 之中，且遵循 KKT 条件：

- 遵循原问题的约束：![t^{(i)}((\bar{w})^T x^{(i)} +\bar{b}) \geq 1](../images/tex-5c71c1287e7a6b8fef19874c553b0cd4.gif), 对于 ![i = 1, 2, ..., m](../images/tex-31f8dedd0a66fb646ef261c638243923.gif)
- 遵循现问题里的约束，即 ![\bar{\alpha}^{(i)} \geq 0](../images/tex-61b00d67f0968d7be5bf4b7a3260b1f4.gif)
-  ![\bar{\alpha}^{(i)} = 0](../images/tex-9eaf40b81df456c80b338612aa1e6fb7.gif) 或者第`i`个约束条件是积极约束，意味着该等式成立：![t^{(i)}((\bar{w})^T x^{(i)} +\bar{b}) = 1](../images/tex-758714f96d598b4cbb8a7642bc3fb017.gif)。这个条件叫做 互补松弛条件。它暗示了 ![\bar{\alpha}^{(i)} = 0](../images/tex-9eaf40b81df456c80b338612aa1e6fb7.gif) 和第`i`个样本位于 SVM 间隔的边界上（该样本是支持向量）。

注意 KKT 条件是确定驻点是否为原问题解的必要条件。在某些条件下，KKT 条件也是充分条件。幸运的是，SVM 优化问题碰巧满足这些条件，所以任何满足 KKT 条件的驻点保证是原问题的解。

我们可以计算上述一般化拉格朗日函数关于`w`和`b`的偏导数，如公式 C-2。

![公式C-2](../images/Appendix/E_C-2.png)

令上述偏导数为 0，可得到公式 C-3。

![公式C-3](../images/Appendix/E_C-3.png)

如果我们把上述式子代入到一般化拉格朗日函数（公式 C-1）中，某些项会消失，从而得到公式 C-4，并称之为原问题的对偶形式。

![公式C-4](../images/Appendix/E_C-4.png)

现在该对偶形式的目标是找到合适的向量 ![\bar{\alpha}](../images/tex-d9c29791dd3b792c7702ed2b7cf5ac40.gif)，使得该函数 ![L(w, b, \alpha)](../images/tex-8d9370145286bec564a001265dd85ff9.gif) 最小化，且 ![\bar{\alpha}^{(i)} \geq 0](../images/tex-61b00d67f0968d7be5bf4b7a3260b1f4.gif)。现在这个有约束优化问题正是我们苦苦追寻的对偶问题。

一旦你找到了最优的 ![\bar{\alpha}](../images/tex-d9c29791dd3b792c7702ed2b7cf5ac40.gif)，你可以利用公式 C-3 第一行计算 ![\bar{w}](../images/tex-28175dc40d9c53d6d2c186a7817cf866.gif)。为了计算 ![\bar{b}](../images/tex-222e2caf9c7b49d3432466e360eceba6.gif)，你可以使用支持向量的已知条件 ![t^{(i)}((\bar{w})^T x^{(i)} +\bar{b}) = 1](../images/tex-758714f96d598b4cbb8a7642bc3fb017.gif)，当第 k 个样本是支持向量时（即它对应的 ![\alpha_k > 0](../images/tex-fe658058e9257029aa88bc89b34348de.gif)），此时使用它计算 ![\bar{b} =1-t^{(k)}((\bar{w})^T . x^{(k)}) ](../images/tex-75080a1229e54394d1c6d95b9e542eaa.gif)。对了，我们更喜欢利用所有支持向量计算一个平均值，以获得更稳定和更准确的结果，如公式 C-5。

![公式C-5](../images/Appendix/E_C-5.png)
