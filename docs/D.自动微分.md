# 附录 D、自动微分

这个附录解释了 TensorFlow 的自动微分功能是如何工作的，以及它与其他解决方案的对比。

假定你定义了函数 ![f(x, y) = x^2y + y + 2](../images/tex-162751afe7e0aa904426973dbac3654e.gif)，需要得到它的偏导数 ![\frac{\partial f}{\partial x}](../images/tex-f6e0346d1d3410b0fbe32b41b85999aa.gif) 和 ![\frac{\partial f}{\partial y}](../images/tex-408378e8bc55170258126d10000c53d9.gif)，以用于梯度下降或者其他优化算法。你的可选方案有手动微分法，符号微分法，数值微分法，前向自动微分，和反向自动微分。TensorFlow 实现的反向自动微分法。我们来看看每种方案。

## 手动微分法

第一个方法是拿起一直笔和一张纸，使用你的代数知识去手动的求偏导数。对于已定义的函数，求它的偏导并不太困难。你需要使用如下 5 条规则：

- 常数的导数为 0。
- ![\lambda x](../images/tex-d08b62e799e1ff8f24464dc26a2daebe.gif) 的导数为 ![\lambda](../images/tex-c6a6eb61fd9c6c913da73b3642ca147d.gif)，![\lambda](../images/tex-c6a6eb61fd9c6c913da73b3642ca147d.gif) 为常数。
- ![x^{\lambda}](../images/tex-74bde878aa116856d62aba260e55c67a.gif) 的导数是 ![\lambda x^{\lambda - 1}](../images/tex-6c620d50445244971a9718316db37470.gif)
- 函数的和的导数，等于函数的导数的和
- ![\lambda](../images/tex-c6a6eb61fd9c6c913da73b3642ca147d.gif) 乘以函数，再求导，等于 ![\lambda](../images/tex-c6a6eb61fd9c6c913da73b3642ca147d.gif) 乘以函数的导数

从上述这些规则，可得到公式 D-1。

![公式D-1](../images/Appendix/E_D-1.png)

这个种方法应用于更复杂函数时将变得非常罗嗦，并且有可能出错。好消息是，像刚才我们做的求数学式子的偏导数可以被自动化，通过一个称为符号微分的过程。

## 符号微分

图 D-1 展示了符号微分是如何运行在相当简单的函数上的，![g(x,y) = 5 + xy](../images/tex-595a140c599de3ceab7b72d4aaab8a41.gif)。该函数的计算图如图的左边所示。通过符号微分，我们可得到图的右部分，它代表了 ![\frac{\partial g}{\partial x} = 0 + (0 \times x + y \times 1) = y](../images/tex-9e9fa7bbdcb31a3b04a549685db18042.gif)，相似地也可得到关于`y`的导数。

![D-1](../images/Appendix/D-1.png)

概算法先获得叶子节点的偏导数。常数 5 返回常数 0，因为常数的导数总是 0。变量`x`返回常数 1，变量`y`返回常数 0，因为 ![\frac{\partial y}{\partial x} = 0](../images/tex-ea6d21230d9c335a071d341ceb54d780.gif)（如果我们找关于`y`的偏导数，那它将反过来）。

现在我们移动到计算图的相乘节点处，代数告诉我们，`u`和`v`相乘后的导数为 ![\frac{\partial (u \times v)}{\partial x} = \frac{\partial v}{\partial x} \times u + \frac{\partial u}{\partial x} \times v ](../images/tex-1cf5205e2548cc4e0ce9e5343ab1a377.gif)。因此我们可以构造有图中大的部分，代表`0 × x + y × 1`。

最后我们往上走到计算图的相加节点处，正如 5 条规则里提到的，和的导数等于导数的和。所以我们只需要创建一个相加节点，连接我们已经计算出来的部分。我们可以得到正确的偏导数，即：![\frac{\partial g}{\partial x} = 0 + (0 \times x + y \times 1) ](../images/tex-7e03e8e758791a8db7937cbbcc78f2b9.gif)。

然而，这个过程可简化。对该图应用一些微不足道的剪枝步骤，可以去掉所有不必要的操作，然后我们可以得到一个小得多的只有一个节点的偏导计算图：![\frac{\partial g}{\partial x} = y](../images/tex-1fda7e8979ad0fdf4a2022ee529661d0.gif)。

在这个例子里，简化操作是相当简单的，但对更复杂的函数来说，符号微分会产生一个巨大的计算图，该图可能很难去简化，以导致次优的性能。更重要的是，符号微分不能处理由任意代码定义的函数，例如，如下已在第 9 章讨论过的函数：

```python
def my_func(a, b):
    z = 0
    for i in range(100):
        z = a * np.cos(z + i) + z * np.sin(b - i)
    return z
```

## 数值微分

从数值上说，最简单的方案是去计算导数的近似值。回忆`h(x)`在 ![x_0](../images/tex-3e0d691f3a530e6c7e079636f20c111b.gif) 的导数 ![h^{'}(x_0)](../images/tex-6499b5277397390a9878a93fa4205525.gif)，是该函数在该点处的斜率，或者更准确如公式 D-2 所示。

![E_D-2](../images/Appendix/E_D-2.png)

因此如果我们想要计算 ![f(x,y)](../images/tex-3baf1600ae50930a155f58ae172b51bd.gif) 关于`x`，在 ![x=3, y=4](../images/tex-99e7bebb7eb398dc777eea8fa1bfe3ba.gif) 处的导数，我们可以简单计算 ![f(3+\epsilon, 4) - f(3, 4)](../images/tex-5dcd5b36cf658a9fbb13000a4cac6989.gif) 的值，将这个结果除以 ![\epsilon](../images/tex-92e4da341fe8f4cd46192f21b6ff3aa7.gif)，且 ![\epsilon](../images/tex-92e4da341fe8f4cd46192f21b6ff3aa7.gif) 去很小的值。这个过程正是如下的代码所要干的。

```python
def f(x, y):
    return x**2*y + y + 2
  
def derivative(f, x, y, x_eps, y_eps):
    return (f(x + x_eps, y + y_eps) - f(x, y)) / (x_eps + y_eps)
  
df_dx = derivative(f, 3, 4, 0.00001, 0)
df_dy = derivative(f, 3, 4, 0, 0.00001)
```

不幸的是，偏导的结果并不准确（并且可能在求解复杂函数时更糟糕）。上述正确答案分别是 24 和 10 ，但我们得到的是：

```python
>>> print(df_dx)
24.000039999805264
>>> print(df_dy)
10.000000000331966
```

注意到为了计算两个偏导数， 我们不得不调用`f()`至少三次（在上述代码里我们调用了四次，但可以优化）。如果存在 1000 个参数，我们将会调用`f()`至少 1001 次。当处理大的神经网络时，这样的操作很没有效率。

然而，数值微分实现起来如此简单，以至于它是检查其他方法正确性的优秀工具。例如，如果它的结果与您手动计算的导数不同，那么你的导数可能包含错误。

## 前向自动微分

前向自动微分既不是数值微分，也不是符号微分，但在某些方面，它是他们的爱情结晶。它依赖对偶数。对偶数是奇怪但迷人的，是 ![a + b\epsilon](../images/tex-595b3d916d7b666f7cec8f222f665759.gif) 形式的数，这里`a`和`b`是实数，![\epsilon](../images/tex-92e4da341fe8f4cd46192f21b6ff3aa7.gif) 是无穷小的数，满足 ![\epsilon ^ 2 = 0](../images/tex-0fe16f5f8178c40813008f32155da044.gif)，但 ![\epsilon \ne 0](../images/tex-11096ba55e57b0ba1b35efb241f87569.gif)。你可以认为对偶数 ![42 + 24\epsilon](../images/tex-63b17a82b832b929bd916f01c8a4dadd.gif) 类似于有着无穷个 0 的 42.0000⋯000024（但当然这是简化后的，仅仅给你对偶数什么的想法）。一个对偶数在内存中表示为一个浮点数对，例如，![42 + 24\epsilon](../images/tex-63b17a82b832b929bd916f01c8a4dadd.gif) 表示为`(42.0, 24.0)`。

对偶数可相加、相乘、等等操作，正如公式 D-3 所示。

![E_D-3](../images/Appendix/E_D-3.png)

最重要的，可证明`h(a + bϵ) = h(a) + b × h'(a)ϵ`，所以计算一次`h(a + ϵ)`就得到了两个值`h(a)`和`h'(a)`。图 D-2 展示了前向自动微分如何计算 ![f(x,y)=x^2y + y + 2](../images/tex-bf7d4f41a093293adbb04e43c7d12839.gif) 关于`x`，在 ![x=3, y=4](../images/tex-99e7bebb7eb398dc777eea8fa1bfe3ba.gif) 处的导数。我们所要做的一切只是计算 ![f(3+\epsilon, 4)](../images/tex-da5577f9751e71377558278256ff1115.gif)；它将输出一个对偶数，其第一部分等于 ![f(3, 4)](../images/tex-744a84046c00c267c037276ee9483cff.gif)，第二部分等于 ![f^{'}(3, 4) = \frac{\partial f}{\partial x} (3,4)](../images/tex-399b8bab86aa930cdbf5c93b2e3fa818.gif)。

![D-2](../images/Appendix/D-2.png)

为了计算 ![\frac{\partial f}{\partial y} (3,4)](../images/tex-3b5f49ee9fe10430f81eeef7000f1b30.gif) 我们不得不再遍历一遍计算图，但这次前馈的值为 ![x=3, y = 4 + \epsilon](../images/tex-a6ef39467ae1ecfdf09a7e93357c3154.gif)。

所以前向自动微分比数值微分准确得多，但它遭受同样的缺陷：如果有 1000 个参数，那为了计算所有的偏导数，得历经计算图 1000 次。这正是反向自动微分耀眼的地方：计算所有的偏导数，它只需要遍历计算图 2 次。

## 反向自动微分

反向自动微分是 TensorFlow 采取的方案。它首先前馈遍历计算图（即，从输入到输出），计算出每个节点的值。然后进行第二次遍历，这次是反向遍历（即，从输出到输入），计算出所有的偏导数。图 D-3 展示了第二次遍历的过程。在第一次遍历过程中，所有节点值已被计算，输入是 ![x=3, y=4](../images/tex-99e7bebb7eb398dc777eea8fa1bfe3ba.gif)。你可以在每个节点底部右方看到这些值（例如，![x \times x = 9](../images/tex-ddfd45b07cca3862ad001dc6551d826a.gif)）。节点已被标号，从 ![n_1](../images/tex-6c773b2b7798e5713845e475d0c4b4c7.gif) 到 ![n_7](../images/tex-97d045dcd64af5ae4cc4add328629288.gif)。输出节点是 ![n_7: f(3, 4) = n_7 = 42](../images/tex-17241d7ea090e8a7be55cacfcd5b2768.gif)。

![D-3](../images/Appendix/D-3.png)

这个计算关于每个连续节点的偏导数的思想逐渐地从上到下遍历图，直到到达变量节点。为实现这个，反向自动微分强烈依赖于链式法则，如公式 D-4 所示。

![E_D-4](../images/Appendix/E_D-4.png)

由于 ![n_7](../images/tex-97d045dcd64af5ae4cc4add328629288.gif) 是输出节点，即 ![f= n_7](../images/tex-9233369b2eac1c4808ae768a0534fa78.gif)，所以 ![\frac{\partial f}{\partial n_7} = 1](../images/tex-c052878d41402368d536c53f4937b012.gif)。

接着到了图的 ![n_5](../images/tex-53eba210fc14ef60860265ec70fb718d.gif) 节点：当 ![n_5](../images/tex-53eba210fc14ef60860265ec70fb718d.gif) 变化时，![f](../images/tex-8fa14cdd754f91cc6554c9e71929cce7.gif) 会变化多少？答案是 ![\frac{\partial f}{\partial n_5} = \frac{\partial f}{\partial n_7} \times \frac{\partial n_7}{\partial n_5}](../images/tex-c4664533339cdf3ddbe912caf82c5bdc.gif)。我们已经知道 ![\frac{\partial f}{\partial n_7} = 1](../images/tex-c052878d41402368d536c53f4937b012.gif)，因此我们只需要知道 ![\frac{\partial n_7}{\partial n_5}](../images/tex-3d189a2e226493acc6538bcd3e9cb366.gif) 就行。因为 ![n_7](../images/tex-97d045dcd64af5ae4cc4add328629288.gif) 是 ![n_5 + n_6](../images/tex-bf018abe4e43c0b3132cba23cb971907.gif) 的和，因此可得到 ![\frac{\partial n_7}{\partial n_5} = 1](../images/tex-68f34602f87a1f0669551323e59a17ea.gif)，因此 ![\frac{\partial f}{\partial n_5}=1 \times 1 = 1](../images/tex-d0a7f1641b3fe72530efcea74fd7a4d2.gif)。

现在前进到 ![n_4](../images/tex-43c5783d36b015e36edeecd60da73206.gif)：当 ![n_4](../images/tex-43c5783d36b015e36edeecd60da73206.gif) 变化时，![f](../images/tex-8fa14cdd754f91cc6554c9e71929cce7.gif) 会变化多少？答案是 ![\frac{\partial f}{\partial n_4} = \frac{\partial f}{\partial n_5} \times \frac{\partial n_5}{\partial n_4}](../images/tex-414889b175f816852566907db5edd6a5.gif)。由于 ![n_5 = n_4 \times n_2](../images/tex-c982adc41e9ee58af9aed4995717fa82.gif)，我们可得到 ![\frac{\partial n_5}{\partial n_4} = n_2](../images/tex-421556b6c8203ded772656e90a1a570c.gif)，所以 ![\frac{\partial f}{\partial n_4}= 1 \times n_2 = 4](../images/tex-5da5d4cf0bebe9ea96d3fbb2c2fd93ca.gif)。

这个遍历过程一直持续，此时我们达到图的底部。这时我们已经得到了所有偏导数在点 ![x=3, y=4](../images/tex-99e7bebb7eb398dc777eea8fa1bfe3ba.gif) 处的值。在这个例子里，我们得到 ![\frac{\partial f}{\partial x} = 24, \frac{\partial f}{\partial y} = 10](../images/tex-e39fd6874bfece3703cdd1eb53e170b0.gif)。听起来很美妙！

反向自动微分是非常强大且准确的技术，尤其是当有很多输入参数和极少输出时，因为它只要求一次前馈传递加上一次反向传递，就可计算所有输出关于所有输入的偏导数。最重要的是，它可以处理任意代码定义的函数。它也可以处理那些不完全可微的函数，只要  你要求他计算的偏导数在该点处是可微的。

如果你在 TensorFlow 中实现了新算子，你想使它与现有的自动微分相兼容，那你需要提供函数，该函数用于构建一个子图，来计算关于新算子输入的偏导数。例如，假设你实现了一个计算其输入的平方的函数，平方算子 ![f(x)= x ^2](../images/tex-d26940d88870bfe622e50be50381fdb9.gif)，在这个例子中你需要提供相应的导函数 ![f^{'}(x)= 2x ](../images/tex-8f515dd3c20d16c5ed6223da611b9a2f.gif)。注意这个导函数不计算一个数值结果，而是用于构建子图，该子图后续将计算偏导结果。这是非常有用的，因为这意味着你可以计算梯度的梯度（为了计算二阶导数，或者甚至更高阶的导数）。
