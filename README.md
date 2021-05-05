## A Tutorial of Text Data Preparation for Machine Learning modeling. 



**A major question. What does** *modeling* **in the title indicate?**

The basic definition of *modeling* is training the machine to do a certain type of task. 
** A 


#### The main objectives of this tutorial:
 * Learn about types of data preparation before any modeling process: manual preparation.
 * Learn about two major libraries that provides text preparation and encoding prcoess. 
#### Main terms that will be  frequently utilize in this tutorial:
 * Tokenization.
 * Vectorization (feature extraction).
 * Term Frequency-Inverse Document (TDF-IF).
 * Hashing.
 * Word embeddings.  

# Manaual Text Data Preparations
The first step before fitting text to a machine learning is **cleaning** your data. Any text contains punctuations, upper cases, white spaces,some characters such as emojis. So, our  task is to prepare our text by splitting it words and deal with puncutation. We will use a text called *Alice's Adventures in Wonderland*. You can find it in the following link https://www.gutenberg.org/files/28885/28885.txt. <sup id="a1">[1](#f1)</sup>
## Cleaning the header and table of contents in Alice's text

This text -- available by  Gutenberg online library, contains a header which is conmposed of Gutenberg project's  introduction, and a table of contents. You have manually to cleany the file by doing the following:
 * Save the file in your local machine in the same directory of the python file.
 * Read the file. 
 * Strip the header [10830:], this slicing operation means start the text from the index 10830 (where the header ends) through the rest of the file.
 * Lower all upper cases.
 * Split the file into strings, then join.
 * Return the variable


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
To view the file after slicing the header, you can print **aliceText[:100]** as the following:

```python
>>>print(aliceText[:100])
"             alice's adventures in wonderland                            lewis carroll                 the millennium fulcrum edition 3.0                                 chapter i                     "
```

## Cleaning the text manually
You will read the term *tokenization* in any NLP turorials or textbooks. For simplicity, it means breaking down a whole text into words or a string of characters. Each string is called *token*. In this section, we will turn alice's text into tokens or separate words.

Alice's text contains punctuations (such as commas and qoutes), and also abbreviated items (such as she's in ...). You can use regular expression to extract these items. You can do it by either using regular expression or use a ready-python tool. Here, we will use a tool called **string.punctuation**, which has a list of punctuation, to remove the target punctuations. 

```markedown
a list of punctuations !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
```


```python
import re
import string
# split aliceText into words by white space
words = aliceText.split()
# prepare regex for character filtering
character_filtering = re.compile('[%s]' % re.escape(string.punctuation)) # remove punctuation from each word
stripped_words = [character_filtering.sub('', w) for w in words]
```
To view the stripped words:

```python
>>>print(stripped_words[:100])
['alices', 'adventures', 'in', 'wonderland', 'lewis', 'carroll', 'the', 'millennium', 'fulcrum', 'edition', '30', 'chapter', 'i', 'down', 'the', 'rabbithole', 'alice', 'was', 'beginning', 'to', 'get', 'very', 'tired', 'of', 'sitting', 'by', 'her', 'sister', 'on', 'the', 'bank', 'and', 'of', 'having', 'nothing', 'to', 'do', 'once', 'or', 'twice', 'she', 'had', 'peeped', 'into', 'the', 'book', 'her', 'sister', 'was', 'reading', 'but', 'it', 'had', 'no', 'pictures', 'or', 'conversations', 'in', 'it', 'and', 'what', 'is', 'the', 'use', 'of', 'a', 'book', 'thought', 'alice', 'without', 'pictures', 'or', 'conversation', 'so', 'she', 'was', 'considering', 'in', 'her', 'own', 'mind', 'as', 'well', 'as', 'she', 'could', 'for', 'the', 'hot', 'day', 'made', 'her', 'feel', 'very', 'sleepy', 'and', 'stupid', 'whether', 'the', 'pleasure']
```
# Preparing Text Data with scikit-learn

Before training your data, you need to enable the machine to identify each string. Since machines can only understand numerical characters, we need to encode each word into numbers either intergers or floating values. This encoding process is called *vectorization* or *feature extraction*. One of the common libraries that offer encoding tools is **scikit-learn**. <sup id="a2">[2](#f2)</sup> 
In this section, we will use three tools in scikit-learn:
* **CountVectorizer** is a tool that converts a text to word count vectors.
* **TfidfVectorizer** is a tool that converts a text to word frequency.
* **HashingVectorizer** is a tool that converts a text to unique intergers. 
## Bag-of-words model
As mentioned above, we need to convert any text into numbers to feed it to a machine learning model. Usually, a text, which is our *input*, must be fed to a model to predict the *output*. One of the simples and old-fashioned models is the **Bag-of-Words** model  which is a method of converting words into numbers by disregarding the order of words in its context. \

### CountVectorizer tool
This tool provides a way to tokenize texts, and encode  these tokenzied items. To do so, we do the following steps:
* Import CountVectorizer from sklearn library.
* Load CountVectorizer in a variable to load the transofrm
* Fit the model, which is an operation to learn the vocabulary from a text.
* Then, trnasform the model, which is an operation to encode the tokenized items as vectors. 

During any process of encoding words into vectors, you have to be careful that vectors will contain a lot of zeros. These zeros are called *sparse*. Thus, we need an extra operation to turn these *sparse vectors* into real numbers by using a package **scipy.sparse**. This package provides a function called *todarry* that them into numbers. 
* Note that we will use the stripped words that we did in the previous section.

```python
from sklearn.feature_extraction.text import CountVectorizer

aliceText = [" Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do:  once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, `and what is the use of a book,' thought Alice `without pictures or conversation?'", " So she was considering in her own mind (as well as she could,for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her."]

vectorizer = CountVectorizer()

# tokenize and build vocab
vectorizer.fit(aliceText)


# show the the tokenized vocabulary 
print(vectorizer.vocabulary_)

# encode document
vector = vectorizer.transform(aliceText)

# summarize encoded vector 
shaped_vector = vector.shape 
type_vector = type(vector)
vector_toarray = vector.toarray()
```
To view the tokenized items, you can use a tool in CountVectorizer called *vocabulary_*

```python
>>>print(shaped_vector)
{'alice': 0, 'was': 66, 'beginning': 5, 'to': 60, 'get': 22, 'very': 65, 'tired': 59, 'of': 37, 'sitting': 52, 'by': 8, 'her': 26, 'sister': 51, 'on': 38, 'the': 57, 'bank': 3, 'and': 1, 'having': 25, 'nothing': 36, 'do': 18, 'once': 39, 'or': 40, 'twice': 62, 'she': 50, 'had': 24, 'peeped': 42, 'into': 29, 'book': 6, 'reading': 49, 'but': 7, 'it': 31, 'no': 35, 'pictures': 44, 'conversations': 13, 'in': 28, 'what': 68, 'is': 30, 'use': 64, 'thought': 58, 'without': 73, 'conversation': 12, 'so': 54, 'considering': 11, 'own': 41, 'mind': 34, 'as': 2, 'well': 67, 'could': 14, 'for': 21, 'hot': 27, 'day': 17, 'made': 32, 'feel': 20, 'sleepy': 53, 'stupid': 55, 'whether': 70, 'pleasure': 46, 'making': 33, 'daisy': 16, 'chain': 9, 'would': 75, 'be': 4, 'worth': 74, 'trouble': 61, 'getting': 23, 'up': 63, 'picking': 43, 'daisies': 15, 'when': 69, 'suddenly': 56, 'white': 71, 'rabbit': 47, 'with': 72, 'pink': 45, 'eyes': 19, 'ran': 48, 'close': 10}
```
Now let's print the shape of the voctorized item

```python
>>>print(shaped_vector)
(2, 76)
```

76 indicates that the encoded vectors have a length of 76 in two documents. 
 
Now let's print the type of arrays that have been encoded.

```python
>>>print(type_vector)
<class 'scipy.sparse.csr.csr_matrix'>
```

The above output means that the sparse zeros have been turned into vectors.

Finally, let's print the array of the encoded vectors.

```python 
>>>print(vector_toarray)
[[2 2 0 1 0 1 2 1 1 0 0 0 1 1 0 0 0 0 1 0 0 0 1 0 2 1 2 0 1 1 1 2 0 0 0 1
  1 3 1 1 3 0 1 0 2 0 0 0 0 1 1 2 1 0 0 0 0 3 1 1 2 0 1 0 1 1 2 0 1 0 0 0
  0 1 0 0]
 [0 2 2 0 1 0 0 0 1 1 1 1 0 0 1 1 1 1 0 1 1 1 0 1 0 0 3 1 1 0 0 0 1 1 1 0
  0 2 0 0 0 1 0 1 0 1 1 1 1 0 2 0 0 1 1 1 1 4 0 0 0 1 0 1 0 1 1 1 0 1 1 1
  1 0 1 1]]
```

After this step, the ecoded vectors are ready to be used in a predictive model.


### Tf-idf Vectorizer tool
We used CountVectorizer function to count the words, which is a very basic method. A serious problem that you will face is that some words such as function words will be highly frequent, which means that providing their counts will not be helpful. Thus, machine learning specialists suggest an alternative method which is *word frequency* instead of the word counts. They created a method called *Term Frequency - Inverse Document*. The term is hypentated which carries two meanings. First, *term frequency* which means the count of words within a document. The other part of term *Inverse Document Frequency* meaning downscale words that are freqeunt across documents. 

In other words, Tfidf vectorization method will provide the frequeny words in a document but not across documents. 

```python 
from sklearn.feature_extraction.text import TfidfVectorizer

aliceText = [" Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do:  once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, `and what is the use of a book,' thought Alice `without pictures or conversation?'", " So she was considering in her own mind (as well as she could,for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.", "There was nothing so VERY remarkable in that; nor did Alice think it so VERY much out of the way to hear the Rabbit say to itself, `Oh dear!  Oh dear!  I shall be late!'  (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually TOOK A WATCH OUT OF ITS WAISTCOAT-POCKET, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge."]

# create the transform
vectorizer = TfidfVectorizer()

# tokenize and build vocab 
vectorizer.fit(aliceText)
# show the tokenized vocabulary
vecotrized_vocab = vectorizer.vocabulary_
# show the inversed documents
idf_vectors = vectorizer.idf_
# transform
vector = vectorizer.transform([aliceText[0]])

# show the shape of vectors
vector_shape = vector.shape

# vectory into arrays
vector_toarray =vector.toarray()
```

* Let's print the counts of each word by using *.vocabulary_* function. 

```python
>>>print(vecotrized_vocab)
{'alice': 4, 'was': 123, 'beginning': 12, 'to': 114, 'get': 39, 'very': 121, 'tired': 113, 'of': 70, 'sitting': 98, 'by': 16, 'her': 46, 'sister': 97, 'on': 72, 'the': 106, 'bank': 9, 'and': 6, 'having': 43, 'nothing': 68, 'do': 29, 'once': 73, 'or': 74, 'twice': 117, 'she': 96, 'had': 41, 'peeped': 79, 'into': 51, 'book': 13, 'reading': 89, 'but': 15, 'it': 53, 'no': 66, 'pictures': 81, 'conversations': 21, 'in': 50, 'what': 127, 'is': 52, 'use': 120, 'thought': 111, 'without': 132, 'conversation': 20, 'so': 100, 'considering': 19, 'own': 78, 'mind': 62, 'as': 7, 'well': 126, 'could': 22, 'for': 37, 'hot': 48, 'day': 26, 'made': 60, 'feel': 33, 'sleepy': 99, 'stupid': 102, 'whether': 129, 'pleasure': 83, 'making': 61, 'daisy': 25, 'chain': 17, 'would': 135, 'be': 10, 'worth': 134, 'trouble': 116, 'getting': 40, 'up': 119, 'picking': 80, 'daisies': 24, 'when': 128, 'suddenly': 103, 'white': 130, 'rabbit': 87, 'with': 131, 'pink': 82, 'eyes': 32, 'ran': 88, 'close': 18, 'there': 108, 'remarkable': 90, 'that': 105, 'nor': 67, 'did': 28, 'think': 109, 'much': 63, 'out': 76, 'way': 125, 'hear': 44, 'say': 91, 'itself': 55, 'oh': 71, 'dear': 27, 'shall': 95, 'late': 58, 'over': 77, 'afterwards': 3, 'occurred': 69, 'ought': 75, 'have': 42, 'wondered': 133, 'at': 8, 'this': 110, 'time': 112, 'all': 5, 'seemed': 93, 'quite': 86, 'natural': 64, 'actually': 1, 'took': 115, 'watch': 124, 'its': 54, 'waistcoat': 122, 'pocket': 84, 'looked': 59, 'then': 107, 'hurried': 49, 'started': 101, 'feet': 34, 'flashed': 36, 'across': 0, 'never': 65, 'before': 11, 'seen': 94, 'either': 31, 'take': 104, 'burning': 14, 'curiosity': 23, 'field': 35, 'after': 2, 'fortunately': 38, 'just': 56, 'see': 92, 'pop': 85, 'down': 30, 'large': 57, 'hole': 47, 'under': 118, 'hedge': 45}
```

* Let's print the shape of vectors to show how many words were vectorized in *aliceText*.

```python
>>>print(vector_shape)
(1, 136)
```

The output means that the number of vocabulary learned from *aliceText* is 136.  

```python 
>>>print(idf_vectors)
[1.69314718 1.69314718 1.69314718 1.69314718 1.28768207 1.69314718
 1.         1.69314718 1.69314718 1.69314718 1.28768207 1.69314718
 1.69314718 1.69314718 1.69314718 1.28768207 1.28768207 1.69314718
 1.69314718 1.69314718 1.69314718 1.69314718 1.69314718 1.69314718
 1.69314718 1.69314718 1.69314718 1.69314718 1.69314718 1.69314718
 1.69314718 1.69314718 1.69314718 1.69314718 1.69314718 1.69314718
 1.69314718 1.28768207 1.69314718 1.69314718 1.69314718 1.28768207
 1.69314718 1.69314718 1.69314718 1.69314718 1.         1.69314718
 1.69314718 1.69314718 1.         1.69314718 1.69314718 1.28768207
 1.69314718 1.69314718 1.69314718 1.69314718 1.69314718 1.69314718
 1.69314718 1.69314718 1.28768207 1.69314718 1.69314718 1.69314718
 1.69314718 1.69314718 1.28768207 1.69314718 1.         1.69314718
 1.28768207 1.69314718 1.28768207 1.69314718 1.69314718 1.69314718
 1.69314718 1.69314718 1.69314718 1.69314718 1.69314718 1.69314718
 1.69314718 1.69314718 1.69314718 1.28768207 1.28768207 1.69314718
 1.69314718 1.69314718 1.69314718 1.69314718 1.69314718 1.69314718
 1.         1.69314718 1.69314718 1.69314718 1.28768207 1.69314718
 1.69314718 1.69314718 1.69314718 1.69314718 1.         1.69314718
 1.69314718 1.69314718 1.69314718 1.28768207 1.69314718 1.69314718
 1.28768207 1.69314718 1.69314718 1.69314718 1.69314718 1.69314718
 1.69314718 1.         1.69314718 1.         1.69314718 1.69314718
 1.69314718 1.69314718 1.28768207 1.69314718 1.69314718 1.28768207
 1.69314718 1.69314718 1.69314718 1.69314718]
```

* Let's show the array of the vectors

```python

>>>print(vector.toarray())
[[0.         0.         0.         0.         0.19659101 0.
  0.15267046 0.         0.         0.12924678 0.         0.
  0.12924678 0.25849355 0.         0.0982955  0.0982955  0.
  0.         0.         0.12924678 0.12924678 0.         0.
  0.         0.         0.         0.         0.         0.12924678
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.12924678 0.         0.19659101
  0.         0.12924678 0.         0.         0.15267046 0.
  0.         0.         0.07633523 0.12924678 0.12924678 0.19659101
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.12924678 0.         0.0982955  0.         0.22900568 0.
  0.0982955  0.12924678 0.29488651 0.         0.         0.
  0.         0.12924678 0.         0.25849355 0.         0.
  0.         0.         0.         0.         0.         0.12924678
  0.         0.         0.         0.         0.         0.
  0.07633523 0.25849355 0.12924678 0.         0.         0.
  0.         0.         0.         0.         0.22900568 0.
  0.         0.         0.         0.0982955  0.         0.12924678
  0.19659101 0.         0.         0.12924678 0.         0.
  0.12924678 0.07633523 0.         0.15267046 0.         0.
  0.         0.12924678 0.         0.         0.         0.
  0.12924678 0.         0.         0.        ]]
```

## Hashing Vectorizer
So far, we learned how to do counting and frequecies of words of a text. However, if we have a corpus that contain millions of words, we will need large vectors for encoding these words. As a consequence, our machine will need more memory to process them. The *hashing* method is a process that uses a one way hash of words to convert them into intergers.  However, the one-way hashing encoding has a major drawback which is once you convert words into intergers, you cannot encode them back to words. 

```python 
from sklearn.feature_extraction.text import HashingVectorizer

aliceText = [" Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do:  once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, `and what is the use of a book,' thought Alice `without pictures or conversation?'", " So she was considering in her own mind (as well as she could,for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.", "There was nothing so VERY remarkable in that; nor did Alice think it so VERY much out of the way to hear the Rabbit say to itself, `Oh dear!  Oh dear!  I shall be late!'  (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually TOOK A WATCH OUT OF ITS WAISTCOAT-POCKET, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge."]

# create the hashing transform
vectorizer = HashingVectorizer(n_features=20) # encode  a smaple of 20-element sparse array out of the documents.

# encode document
vector = vectorizer.transform(aliceText)

# show the vector shape
vector_shape = vector.shape

# show the array of vectors 
vector_to_array = vector.toarray()
```
* Print the sahpe
```python 
>>>print(vector_shape)
(3, 20)
```

* Print the vector to array

```python
>>>print(vector_to_array)
[[ 0.12909944  0.25819889  0.12909944  0.         -0.25819889 -0.25819889
   0.          0.          0.25819889  0.12909944  0.          0.25819889
   0.          0.          0.64549722 -0.12909944  0.38729833  0.12909944
   0.          0.12909944]
 [-0.1132277  -0.1132277   0.         -0.22645541 -0.33968311 -0.33968311
  -0.33968311  0.1132277  -0.1132277   0.33968311 -0.22645541 -0.1132277
   0.          0.1132277   0.1132277  -0.1132277   0.33968311  0.
  -0.45291081 -0.1132277 ]
 [ 0.06428243  0.          0.25712974  0.         -0.12856487 -0.25712974
  -0.12856487  0.1928473   0.32141217  0.32141217 -0.25712974  0.32141217
  -0.06428243  0.12856487  0.         -0.25712974  0.38569461 -0.12856487
   0.38569461  0.12856487]]
 ```

# Preparing Text Data With Keras 

Keras is a python library that provides some tools to prepare your data before modeling your data. In this section, we will learn how to use some tools similar to the ones in *sklearn*. These tools arre:
* Tokenizer API
* Hashing_trick
* One_hot encoder.

Keras provides a good tool for tokenizing your corpus. This tool is *text_to_word_sequence*. Again, we will use *aliceText* from previous sections. 

```python
from keras.preprocessing.text import text_to_word_sequence

aliceText = [" Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do:  once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, `and what is the use of a book,' thought Alice `without pictures or conversation?'", " So she was considering in her own mind (as well as she could,for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.", "There was nothing so VERY remarkable in that; nor did Alice think it so VERY much out of the way to hear the Rabbit say to itself, `Oh dear!  Oh dear!  I shall be late!'  (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually TOOK A WATCH OUT OF ITS WAISTCOAT-POCKET, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge."]

# tokenize the document
tokenized_alice = text_to_word_sequence(aliceText)
```

* Show the tokenized item
```python 
>>>print(tokenized_alice)
['alice',
 'was',
 'beginning',
 'to',
 'get',
 'very',
 'tired',
 'of',
 'sitting',
 'by',
 'her',
 'sister',
 'on',
 'the',
 'bank',
 'and',
 'of',
 'having',
 'nothing',
 'to',
 'do',
 'once',
 'or',
 'twice',
 'she',
 'had',
 'peeped',
 'into',
 'the',
 'book',
 'her',
 'sister',
 'was',
 'reading',
 'but',
 'it',
 'had',
 'no',
 'pictures',
 'or',
 'conversations',
 'in',
 'it',
 'and',
 'what',
 'is',
 'the',
 'use',
 'of',
 'a',
 'book',
 "'",
 'thought',
 'alice',
 'without',
 'pictures',
 'or',
 'conversation',
 "'"]
```













<b id="f1">1</b> Carroll, L. Alice's Adventures in Wonderland. Project Gutenberg, May 19, 2009. https://www.gutenberg.org/files/28885/28885.txt. 

<b id="f2">2</b> Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.

<b id="f3">3</b>  https://jalammar.github.io/illustrated-word2vec/
