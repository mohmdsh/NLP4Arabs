## A Tutorial of Text Data Preparation for Machine Learning modeling. 



**A major question. What does** *modeling* **in the title indicate?**

The basic definition of *modeling* is training the machine to do a certain type of task. 
** A 


#### The main objectives of this tutorial:
> * Learn about types of data preparation before any modeling process: manual preparation, NLTK method.
> * Learn about two major libraries that provides text preparation and encoding prcoess. 
#### Main terms that will be  frequently utilize in this tutorial:
> * Tokenization.
> * Vectorization (feature extraction).

# Manaual Text Data Preparations
The first step before fitting text to a machine learning is **cleaning** your data. Any text contains punctuations, upper cases, white spaces,some characters such as emojis. So, our  task is to prepare our text by splitting it words and deal with puncutation. We will use a text called *Alice's Adventures in Wonderland*. You can find it in the following link https://www.gutenberg.org/files/28885/28885.txt. <sup id="a1">[1](#f1)</sup>
## Alice's Adentures in Wonderlang

This text -- available by  Gutenberg online library, contain an introduction by Gutenberg Project, a table of contents. You have to prepare your text for modeling by removing these unneeded redundunt stuff. In python, you can do this by using **slicing**. What is silicing? Basically, it is a python operation that can slice a text out of a text. As shown in the Figure 1, each string is identified by a numerical index. The index from the from 
Thus, we need to extract a new text out of Alice's text by excluding Gutenberg Project's introduction and the table of contents. 

```python
# import regular expression library
import re


def readAlice(file_name):
    Alice = open(file_name,'r')
    #A. read Alice
    Alice = Alice.read()
    # B. strip the header from readAlice 
    Alice = Alice[10830:] 
    Alice = Alice.lower()
    Alice = ' '.join(Alice.splitlines()[:3000])
    return Alice
alice_file_name = 
aliceText = readAlice(alice_file_name)
aliceText

```

# Preparing Alice's text with S


```python
x = open(file)
```
```markdown
# hello
```



<b id="f1">1</b> Carroll, L. Alice's Adventures in Wonderland. Project Gutenberg, May 19, 2009. https://www.gutenberg.org/files/28885/28885.txt.  (#a1)
