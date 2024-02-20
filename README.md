[English](#OneTransformer) | [简体中文](#中文)

# OneTransformer
This is tutorial for training a PyTorch transformer from scratch

# Why I create this project
There are many tutorials for how to train a transformer, including pytorch official tutorials
while even the official tutorial only contains "half" of it -- it only trains the encoder part

and there are some other tutorials which uses some fake data just for demo, or they don't provide the complete code.

# How to run?
just run main.py to start training.

```
python main.py
```

for inferencing test, run
```
python inference.py
```
If you are using windows or mac, you can use PyCharm IDE, and right click to run main.py

# About this model
This is a small Transformer architecture model with 8 million parameters
When training a transformer model, typically we customize a wrapper model class for the pytorch library implemented Transformer.

it usually contains of:
- the embedding and positioning module
- the pytorch implemented transformer module
- the output linear module based on you tokenizer length


# The data
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



# Training records

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

# 中文



