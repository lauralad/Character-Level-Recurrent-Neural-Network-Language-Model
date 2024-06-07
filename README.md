# Character-Level Recurrent Neural Network Language Model
 Comp 545 - Natural Language Processing

## Description
This project involves the implementation of a character-based RNN language model that learns to predict the next character in a sequence, given the previous characters. The model was trained on various literary texts, enabling it to generate text that mimics the style and context of the source material.

### Objectives:
* Implement a character sequence data loader to process text data into trainable sequences.
* Develop a Character RNN using PyTorch to learn and predict character sequences.
* Explore the model's ability to generate coherent and contextually relevant text as it trains.
* Evaluate the impact of different parameters (like temperature and top_k filtering) on the diversity and quality of generated text.

### Technologies Used:
* Programming Language: Python
* Frameworks/Libraries: PyTorch, NumPy, Hugging Face Datasets, Matplotlib
* Tools: Kaggle

### Features:
* Data Preprocessing: Custom data loader to convert raw text into sequences of characters for the RNN.
* Model Architecture: Built and trained a character-level RNN from scratch.
* Text Generation: Implemented functions to generate text by sampling the model's predictions, incorporating techniques like temperature scaling and top-K filtering to control diversity.
* Parameter Tuning: Explored the effects of various hyperparameters on the model's text generation capabilities.

### Results:
![alt text](https://github.com/lauralad/Character-Level-Recurrent-Neural-Network-Language-Model/blob/main/a3_result.JPG?raw=true)

RNN Result Sample
```
Un I all the stain that that is side is I the slay as near the have the manded youm be in sear. It the but that 
the man ther the the but the blears and the dear the splen have brist the may be all hear
```


LSTM Result Sample
```
Ã:
"Yound the fack me for a more his a will had be a chere away I conger."
"We colled the prom which a last the wore will be at dear and the brow and his me her and a pars, 
some your his the a look
```


Considering all the samples and loss curves, I observed that higher values for all parameters introduces some 
nonsense in a text that is already hard to understand. However, it did better on lower values, 
close to medium. Comparing RNN and LSTM, I can conclude that LSTM produced more 
coherent text that somehow made sense in some parts of the sequence as opposed to RNN.

## How to Use:
* Clone the repository.
* Install required packages from ```requirements.txt```.
* Execute the script ```code.py``` locally or in notebook form on Kaggle to train the model and see text generation in action.