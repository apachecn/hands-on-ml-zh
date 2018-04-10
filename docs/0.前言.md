# Chapter 0.前言

## 1、机器学习海啸

2006年，Geoffrey Hinton等人发表了一篇论文，展示了如何训练能够识别具有最新精度（> 98%）的手写数字的深度神经网络。他们称这种技术为“Deep Learning”。当时，深度神经网络的训练被广泛认为是不可能的，并且大多数研究人员自 20 世纪 90 年代以来就放弃了这个想法。这篇论文重新激起了科学界的兴趣，不久之后，许多新发表的论文表明，深度学习不仅是可能的，而且能够取得其他的 Machine Learning 技术都难以匹配的令人兴奋的成就（借助巨大的计算能力和大量的数据）。这种热情很快扩展到机器学习的许多的其他领域。

Deep Learning 快速发展的 10 年间和机器学习已经征服了这个行业：它现在成为了当今高科技产品中的许多黑科技的核心，比如，为您的网络搜索结果排名，为智能手机的语音识别提供支持，为您推荐您喜欢的视频，在 Go 游戏中击败世界冠军。在你知道之前，它都可能会驾驶您的汽车。

## 2、您项目中的机器学习

现在你是不是对机器学习感到兴奋，并且很乐意加入到这个阵营中？
也许你希望给自己制造的机器人赋予一个自己的大脑？让它可以面部识别？还是学会到处走走？

也许你的公司有大量的数据（用户日志，财务数据，生产数据，机器传感器数据，热线统计数据，人力资源报告等），如果你知道在哪方面观察，你可能会发现一些隐藏着的瑰宝。例如：
* 细分客户，为每个团队找到最佳的营销策略
* 根据类似客户购买的产品为每个客户推荐产品
* 检测哪些交易可能是欺诈行为
* 预测下一年的收入
* 更多应用

无论什么原因，你决定开始学习机器学习，并在你的项目中实施，这是一个好主意！

## 3、目标和方法

本书假定你对机器学习几乎一无所知。它的目标是给你实际实现能够从数据中学习的程序所需的概念，直觉和工具。

我们将介绍大量的技术，从最简单的和最常用的（如线性回归）到一些定期赢得比赛的深度学习技术。

我们将使用现成的 Python 框架，而不是实现我们自己的每个算法的玩具版本：

* Scikit-learn 非常易于使用，并且实现了许多有效的机器学习算法，因此它为学习机器学习提供了一个很好的切入点。

* TensorFlow 是使用数据流图进行分布式数值计算的更复杂的库。它通过在潜在的数千个多 GPU 服务器上分布式计算，可以高效地训练和运行非常大的神经网络。TensorFlow 是被 Google 创造的，支持其大型机器学习应用程序。于 2015年11月开源。

本书倾向于实际操作的方法，通过具体的实例和一点理论来增加对机器学习的直观理解。虽然你可以在不拿笔记本电脑的情况下阅读此书，但是我们强烈建议你通过 https://github.com/ageron/handson-ml 在线实现 Jupyter notebooks 上的代码示例。

## 4、准备条件

本书假定您有一些 Python 编程经验，并且比较熟悉 Python 的主要科学库，特别是 NumPy，Pandas 和 Matplotlib 。

另外，如果你关心的是底层实现/原理，你应该对大学水平的数学（微积分，线性代数，概率和统计学）有一些了解。

如果你还不了解 Python，http://learnpython.org/ 是你学习使用 Python 的好地方。 python.org 官方教程也是相当不错的。

如果你从未使用过 Jupyter ，第 2 章将指导你完成安装和基本操作：它是你工具箱中的一个很好的工具。

如果你不熟悉 Python 的科学库，提供的一些 Jupyter notebook 包括了一些教程。还有一个线性代数的快速数学教程。

## 5、路线图

这本书分为两个部分。

第一部分，机器学习的基础知识，涵盖以下主题：

* 什么是机器学习？它被试图用来解决什么问题？机器学习系统的主要类别和基本概念是什么？
* 典型的机器学习项目中的主要步骤。
* 通过拟合数据来学习模型。
* 优化成本函数（cost function）。
* 处理，清洗和准备数据。
* 选择和设计特征。
* 使用交叉验证选择一个模型并调整超参数。
* 机器学习的主要挑战，特别是欠拟合和过拟合（偏差和方差权衡）。
* 对训练数据进行降维以对抗 the curse of dimensionality（维度诅咒）
* 最常见的学习算法：线性和多项式回归， Logistic 回归，k-最近邻，支持向量机，决策树，随机森林和集成方法。

第二部分，神经网络和深度学习，包括以下主题：

* 什么是神经网络？它们有啥优势？
* 使用 TensorFlow 构建和训练神经网络。
* 最重要的神经网络架构：前馈神经网络，卷积网络，递归网络，长期短期记忆网络（LSTM）和自动编码器。
* 训练深度神经网络的技巧。
* 对于大数据集缩放神经网络。
* 强化学习。

第一部分主要基于 scikit-learn ，而第二部分则使用 TensorFlow 。

注意：不要太急于深入学习到核心知识：深度学习无疑是机器学习中最令人兴奋的领域之一，但是你应该首先掌握基础知识。而且，大多数问题可以用较简单的技术很好地解决（而不需要深度学习），比如随机森林和集成方法（我们会在第一部分进行讨论）。如果你拥有足够的数据，计算能力和耐心，深度学习是最适合复杂的问题的，如图像识别，语音识别或自然语言处理。

## 6、其他资源

有许多资源可用于了解机器学习。Andrew Ng 在 Coursera 上的 [ML 课程](https://www.coursera.org/learn/machine-learning/)和 Geoffrey Hinton 关于[神经网络和深度学习](https://www.coursera.org/learn/neural-networks)的课程都是非常棒的，尽管这些课程需要大量的时间投入（大概是几个月）。 

还有许多关于机器学习的比较有趣的网站，当然还包括 scikit-learn 出色的 [用户指南](http://sklearn.apachecn.org/cn/0.19.0/user_guide.html)。你可能会喜欢上 [Dataquest](https://www.dataquest.io/) ，它提供了一个非常好的交互式教程，还有 ML 博客，比如那些在 [Quora](http://goo.gl/GwtU3A) 上列出来的博客。最后，[Deep Learning 网站](http://deeplearning.net/) 有一个很好的资源列表来学习更多。

当然，还有很多关于机器学习的其他介绍性书籍，特别是：

* Joel Grus, Data Science from Scratch (O’Reilly). 这本书介绍了机器学习的基础知识，并在纯 Python 中实现了一些主要算法（从名字上看就可以知道，从头开始）。

* Stephen Marsland, Machine Learning: An Algorithmic Perspective (Chapman andHall). 这本书对机器学习有一个很好的介绍，涵盖了广泛的主题，Python 中的代码示例（也是从零开始，但是使用 NumPy）。

* Sebastian Raschka, Python Machine Learning (Packt Publishing). 本书也对机器学习有一个很好的介绍，但是利用了 Python 的开源库（Pylearn 2 和 Theano）。

* Yaser S. Abu-Mostafa, Malik Magdon-Ismail, and Hsuan-Tien Lin, Learning fromData (AMLBook). 对 ML 有一个相对理论化的介绍，这本书提供了比较深刻的见解，特别是 bias/variance tradeoff （偏差/方差 权衡）（见第 4 章）。

* Stuart Russell and Peter Norvig, Artificial Intelligence: A Modern Approach, 3rd
Edition (Pearson).  这是一本很好的（并且很大）的书，涵盖了包括机器学习在内的大量主题。这有助于更加深刻地理解 ML 。

最后，一个很好的学习方法就是加入 ML 竞赛网站，例如 kaggle.com ，这样可以让你在现实世界的问题上锻炼自己的技能，并从一些最好的 ML 专业人士那里获得帮助和见解。

## 7、本书中的一些约定

本书使用以下印刷约定：

* 斜体 —— 指示新术语，网址，电子邮件地址，文件名和文件扩展名。

* 等宽 —— 用于程序清单，以及段落内用于引用程序元素，如变量或函数名称，数据库，数据类型，环境变量，语句和关键字。

* 等宽粗体 —— 显示应由用户逐字输入的命令或其他文本。

* 等宽斜体 —— 显示应由用户提供的值或由上下文确定的值替换的文本。

* 小松鼠图标 —— 此元素表示一个小提示或建议。

* 小乌鸦图标 —— 此元素表示一个普通的说明。

* 小蝎子图标 —— 此元素表示一个警告和注意。

## 8、使用代码示例

补充材料（代码示例，练习题等）可以从 https://github.com/ageron/handson-ml 下载。

这本书是为了帮助你完成工作。一般来说，如果本书提供了示例代码，则可以在程序和文档中使用它。除非你复制了大部分代码，否则你无需联系我们获得许可。例如，编写使用本书中几个代码块的程序不需要许可。销售或者分发 O’Reilly 书籍的 CD-ROM 例子需要获得许可。

通过引用本书和使用示例代码来回答问题并不需要获得许可。将大量来自本书的示例代码整合到产品文档中并不需要获得许可。

我们感谢，但是并不要求，贡献。贡献通常包括标题，作者，出版商和 ISBN 。例如：“Hands-On Machine Learning withScikit-Learn and TensorFlow by Aurélien Géron (O’Reilly). Copyright 2017 AurélienGéron, 978-1-491-96229-9.”

如果您觉得您对代码示例的使用超出了合理使用范围或上述权限，请随时联系我们：permissions@oreilly.com 。

## 9、O’Reilly Safari

Safari （以前被称为 Safari Books Online）是一个针对企业，政府，教育工作者和个人的基于会员的培训和参考平台。

会员可以访问 250 多家发布商的数千本图书，培训视频，学习路径，互动教程和策划播放列表，其中包括 O’Reilly Media，哈佛商业评论，Prentice Hall 专业人员，Addison-Wesley 专业人员，Microsoft Press， Sams， Que， Peachpit Press， Adobe， Focal Press， Cisco Press 等。想要了解更多信息，请访问 http://oreilly.com/safari 。

## 10、如何联系我们

请向出版商发表有关本书的评论和问题：

O’Reilly Media, Inc.

1005 Gravenstein Highway North

Sebastopol, CA 95472

800-998-9938 （在美国或者加拿大）

707-829-0515 （国际或地区）

707-829-0104 （传真）

我们有一个这本书的网页，在这里我们列出了勘误表，例子和任何额外的信息。你可以访问这个网页 http://bit.ly/hands-on-machine-learning-with-scikit-learn-and-tensorflow

要评论或者询问有关本书的技术问题，请发送电子邮件到 bookquestions@oreilly.com 。

有关我们的书籍，课程，会议和新闻的更多信息，请访问我们的网站 http://www.oreilly.com 。

在 facebook 上找到我们： http://facebook.com/oreilly

在 Twitter 上关注我们：http://twitter.com/oreillymedia

在 Youtube 上观看我们的视频： http://www.youtube.com/oreillymedia


## 11、致谢

我要感谢我的 Google 同事，特别是 Youtube 视频分类小组，教给我很多关于机器学习的知识。没有他们，我永远无法开始这个项目。特别感谢我的个人 ML 专家：Clément Courbet, Julien Dubois, Mathias Kende, Daniel Kitachewsky, James Pack, Alexander Pak, Anosh Raj, Vitor Sessak, Wiktor Tomczak, Ingrid von Glehn, Rich Washington, 以及 Youtube Paris 的所有人。

我非常感谢所有那些从繁忙的生活中抽出时间来仔细阅读我的书的人。感谢 Pete Warden 回答了我所有的 TensorFlow 的问题，回顾第二部分，提供了许多有趣的见解，当然也成为了 TensorFlow 核心团队的一员。你一定想要看看他的 [博客](https://petewarden.com/) ！非常感谢 Lukas Biewald 对第二部分的非常全面的审查：他毫不留情地尝试了所有的代码（并且发现了一些错误），做出了许多伟大的建议，而且他的热情是具有感染力的。你应该看看他的博客，和他的超酷的机器人！感谢 Justin Francis ，他也非常全面地审查了第二部分，特别是在第 16 章提到了错误并提供了很好的见解。你可以在 TensorFlow 上看到他的帖子！

也非常感谢 David Andrzejewski，他审查了第一部分，提供了非常有用的反馈意见，确定了不明确的部分并提出了改进建议。查看一下他的网页吧。感谢 Grégoire Mesnil，他审查了第二部分，并提供了非常有趣的关于神经网络的实用建议。感谢 Eddy Hung, Salim Sémaoune, Karim Matrah, Ingrid von Glehn,Iain Smears, 和 Vincent Guilbeau 对第一部分的审查和建议。我还要感谢我的岳父，前数学老师 Michel Tessier ，现在是 Anton Chekhov 的一名优秀翻译，帮助我在本书中提供了一些非常好的数学和符号，并且审查了线性代数 Jupyter notebook 。

当然，对我亲爱的弟弟说一个巨大的 “谢谢” ，他测试了每一行代码，几乎在每个部分都提供了反馈，并鼓励我从第一行到最后一行。爱你，我的兄弟。

非常感谢 O’Reilly 出色的员工，特别是 Nicole Tache ，他给出了深刻的反馈，并且总是开朗，鼓舞和乐于助人的。还要感谢 Marie Beaugureau, Ben Lorica, Mike Loukides, 和 Laurel Ruma 相信这个项目并帮助我确定其范围。感谢 Matt Hacker 和所有的 Atlasteam 回答了关于格式化，asciidoc 和 LaTeX 的所有技术团队问题，也感谢 Rachel Monaghan, Nick Adams, 和所有的制作团队进行了最终的审查和数百次的修正。

最后但也很重要的一点，我非常感谢我的爱妻 Emmanuelle 和三个非常棒的孩子，Alexandre, Rémi, 和 Gabrielle ，在这本书中写了很多，问了很多问题（谁说不能教 7 岁的孩子神经网络？），甚至帮我送饼干和咖啡。你还梦想得到什么呢？














