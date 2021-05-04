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

This text -- available by  Gutenberg online library, contains a header which is conmposed of Gutenberg project's  introduction, and a table of contents. You have manually to cleany the file by doing the following:
> * save the file in a variable.
> * Read the file. 
> * Strip the header [10830:], this slicing operation means start the text from the index 10830 (where the header ends) through the rest of the file.
> * Lower all upper cases.
> * Split the file into strings, then join.
> * Return the variable


```python
alice_file_name = 'alice.txt'

def readAlice(file_name):
    #open the file 
    Alice = open(file_name,'r')
    # read Alice
    Alice = Alice.read()
    # strip the header from readAlice 
    Alice = Alice[10830:] 
    Alice = Alice.lower()
    Alice = ' '.join(Alice.splitlines()[:3000])
    return Alice
aliceText = readAlice(alice_file_name)
aliceText

```
```python
>>>print(Alice[:100])
"             alice's adventures in wonderland                            lewis carroll                 the millennium fulcrum edition 3.0                                 chapter i                     "
```

# Preparing Alice's text with S


```python
x = open(file)
```
```markdown
# hello
```



<b id="f1">1</b> Carroll, L. Alice's Adventures in Wonderland. Project Gutenberg, May 19, 2009. https://www.gutenberg.org/files/28885/28885.txt.  (#a1)
