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
The first step before fitting text to a machine learning is **cleaning** your data. Any text contains punctuations, upper cases, white spaces,some characters such as emojis. So, our  task is to prepare our text by splitting it words and deal with puncutation. We will use a text called *Alice's Adventures in Wonderland*. You can find it in the following link https://www.gutenberg.org/files/28885/28885.txt. Carroll, L. Alice's Adventures in Wonderland. Project Gutenberg, May 19, 2009. https://www.gutenberg.org/files/28885/28885.txt. [^1] 



```python
x = open(file)
```
```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/mohmdsh/lysaan/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
