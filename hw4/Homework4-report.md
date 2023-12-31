# Homework 4: Machine Learning

> Date: 2022-04-01

Detailed results of Problem-1 and Problem-2 are in the attached `hw4.ipynb` file.

## 1. Multi-Layer Perceptron

**Data processing**

1. calculate "mean" and "std" of the training dataset

2. use the "mean" and "std" to normalize training dataset and test dataset; transformer pipeline is as follows

   ```python
   transformers.Compose([transforms.ToTensor(), transforms.Normalize(mean=data_mean, std=data_std)])
   ```

3. construct `DataLoader` with specific `batch_size`

**Parameter settings**

- optimizer: `torch.optim.Adam`
- learning rate (`lr`): 5e-4
- penalty factor of L2 regularization (`l2_lambda`): 1e-2; herein $L_2$ regularization is added via designating `weight_decay` parameter in the construction function of `torch.optim.Adam` class
- loss function: `torch.nn.CrossEntropy`
- batch size (`batch_size`): 64

**Results comparison**

| Settings                      | Error on Trainset | Error on Testset |
| ----------------------------- | ----------------- | ---------------- |
| 50 HLN + no regularization    | 0.4               | 0.4              |
| 250 HLN + no regularization   | 0.4               | 0.4              |
| 50 HLN + $L_2$ regularization | 0.8               | 0.8              |
| 50 HLN + $L_2$ regularization | 0.8               | 0.8              |

*Remarks: data in the table are the best results among all learning rounds*

## 2. Adaboost

**Dataset**

[adult_train.csv](data/adult_train.csv), [adult_test.csv](data/adult_test.csv)

The origin dataset is too large. Herein I just randomly selected 20000 samples from [adult_train.csv](data/adult_train.csv) and 10000 samples from [adult_test.csv](data/adult_test.csv).

**Comparison result** 

settings: `max_depth=3`, `n_estimators=10`

| Model                        | Accuracy |
| ---------------------------- | -------- |
| Self-implemented Adaboost    | 0.8429   |
| Single Decision Tree         | 0.8417   |
| Built-in Adaboost in sklearn | 0.8517   |

That result demonstrates that the boosting strategy does work, with about 0.15% accuracy improvement. And results of the self-implemented Adaboost and the built-in Adaboost in sklearn are close. Of course, the former is a little insufficient in comparison with the built-in Adaboost from sklean.

## 3. Recurrent Neural Networks for Language Modeling

The baseline show powerful ability of RNN in language generating. Although the results is almost correct in words spelling, it does not care about grammars. That limits its practical application. Herein I attempted to modify the model with reasonable insight, that is, **adding weights to characters of the vocabulary**. Specifically, the occurrence frequencies of English characters is almost fixed. For instance, character `e` occurs with the largest frequency around 13%. Therefore, I added a probability ancilla prediction module into the CNN model. This is like that external prior knowledge is introduced. Eventually, it does lead to loss improvement in contrast to the baseline, but no large improvement effect.
