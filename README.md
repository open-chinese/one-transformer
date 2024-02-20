[English](#one-transformer-project) | [简体中文](#one-transformer项目)
# One-Transformer Project

## About this project
This is tutorial for training a PyTorch transformer from scratch

## Why I create this project
There are many tutorials for how to train a transformer, including pytorch official tutorials
while even the official tutorial only contains "half" of it -- it only trains the encoder part

and there are some other tutorials which uses some fake data just for demo, or they don't provide the complete code.

## How to run?
just run main.py to start training.

```
python main.py
```

for inferencing test, run
```
python inference.py
```
If you are using windows or mac, you can use PyCharm IDE, and right click to run main.py

## About this model
This is a small Transformer architecture model with 8 million parameters
When training a transformer model, typically we customize a wrapper model class for the pytorch library implemented Transformer.

it usually contains of:
- the embedding and positioning module
- the pytorch implemented transformer module
- the output linear module based on you tokenizer length


## The data
This is decided what kind of LLM we want to implement, here we don't use the torch rand data
or some simple data. we try to solve a real problem with our LLM, thought this problem can be done with other methods

the problem is:

give you a number, more precisely, a rational number. output the corresponding English words of this number.

| Number   |      English words      |
|----------|:-------------:|
| 5000 |  five thousand |
| -92101602.123 |  minus ninety two million , one hundred and one thousand , six hundred and two point one two three |

we regard the number as the input string, and the English words as the output string

This is actually a Translation task.



## Training records

### 1st training
I used 10k training data, batch size 16, lr init with 1e-4, and StepLR gamma 0.1 with step size 1, trained 4 epochs. get these results. althouugh just one correct answer, seems the results are real valid numbers.

```
1235678 --> five million , six hundred and twenty eight thousand , seven hundred and thirteen
200.3236 --> thirty two thousand , three hundred and sixty point zero
-2000 --> minus two thousand and twenty
10000001 --> ten million , one hundred thousand and
66666666 --> sixty six million , six hundred and sixty six thousand , six hundred and sixty six
-823982502.002 --> minus two hundred and twenty two million , eight hundred and twenty thousand , nine hundred and eighty point zero three five
987654321.12 --> two hundred and twenty one million , nine hundred and sixty five thousand , four hundred and eighty three point one seven
3295799.9873462 --> nine hundred and ninety three million , two hundred and ninety seven thousand , four hundred and sixty five point three eight two seven
```



# One-Transformer项目

## 关于这个项目
这是一个从头开始训练PyTorch transformer的教程。

## 为什么我创建这个项目
有很多关于如何训练transformer的教程，包括PyTorch官方教程，但即使是官方教程也只包含了一半内容——它只训练了编码器部分。

还有一些其他教程使用一些假数据仅用于演示，或者他们没有提供完整的代码。

## 如何运行？
只需运行main.py即可开始训练。

```
python main.py
```

对于推理测试，运行
```
python inference.py
```
如果您使用的是Windows或Mac，可以使用PyCharm IDE，并右键运行main.py。

## 关于这个模型
这是一个小型的Transformer架构模型，拥有800万个参数。在训练transformer模型时，通常我们会为PyTorch库实现的Transformer定制一个包装模型类。

它通常包含以下内容：

嵌入和定位模块
PyTorch实现的transformer模块
基于你的tokenizer长度的输出线性模块

## 数据
这决定了我们想要实现什么样的LLM，这里我们不使用torch rand数据或一些简单的数据。我们尝试用我们的LLM解决一个真实的问题，尽管这个问题可以用其他方法解决。

问题是：

给你一个数字，更准确地说，是一个有理数。输出这个数字对应的英文单词。

| Number   |      English words      |
|----------|:-------------:|
| 5000 |  five thousand |
| -92101602.123 |  minus ninety two million , one hundred and one thousand , six hundred and two point one two three |

我们将数字视为输入字符串，将英文单词视为输出字符串。

这实际上是一个翻译任务。

### 训练记录

第一次训练
我使用了1万个训练数据，批量大小为16，学习率初始值为1e-4，StepLR gamma为0.1，步长为1，训练了4个周期。得到了这些结果。尽管只有一个正确答案，但看起来结果是真实有效的数字。

```
1235678 --> five million , six hundred and twenty eight thousand , seven hundred and thirteen
200.3236 --> thirty two thousand , three hundred and sixty point zero
-2000 --> minus two thousand and twenty
10000001 --> ten million , one hundred thousand and
66666666 --> sixty six million , six hundred and sixty six thousand , six hundred and sixty six
-823982502.002 --> minus two hundred and twenty two million , eight hundred and twenty thousand , nine hundred and eighty point zero three five
987654321.12 --> two hundred and twenty one million , nine hundred and sixty five thousand , four hundred and eighty three point one seven
3295799.9873462 --> nine hundred and ninety three million , two hundred and ninety seven thousand , four hundred and sixty five point three eight two seven
```

