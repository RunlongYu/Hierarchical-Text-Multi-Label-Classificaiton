# Hierarchical Text Multi Label Classificaiton

This repository is my research project, and it is also a study of TensorFlow, Deep Learning.

The main objective of the project is to solve the hierarchical multi-label text classification (**HMC**) problem. Different from the multi-label text classification, HMC classifies each instance (object) into several different paths of the class hierarchy.

## Requirements

- Python 3.6
- Tensorflow 1.8 +
- Numpy
- Gensim

## Introduction

Many real-world applications involve hierarchical multi-label classification and organize data in a hierarchical structure, classes are specialized into subclasses or grouped into superclasses, which is a good way to show the characteristics of data and provide a multidimensional perspective to tackle the problem. 

Like most type of electronic document (e.g. web-pages, digital libraries, patents and e-mails), they are usually associated with one or more categories and all these categories are stored hierarchically in a **tree** or **Direct Acyclic Graph (DAG)**.

![](https://farm8.staticflickr.com/7806/31717892987_e2e851eaaf_o.png)

The Figure show an example of predefined labels in hierarchical multi-label classification of documents in a patent texts. 

- Documents are shown as colored rectangles, labels as rounded rectangles. 
- Circles in the rounded rectangles indicate that the corresponding document has been assigned the label. 
- Arrows indicate hierarchical structure between labels.

## Data

See data format in `data` folder which including the data sample files.

### Text Segment

You can use `jieba` package if you are going to deal with the chinese text data.

### Data Format

This repository can be used in other datasets(text classification) by two ways:
1. Modify your datasets into the same format of the sample.
2. Modify the data preprocess code in `data_helpers.py`.

Anyway, it should depends on what your data and task are.

### Pre-trained Word Vectors

You can pre-training your word vectors(based on your corpus) in many ways:
- Use `gensim` package to pre-train data.
- Use `glove` tools to pre-train data.
- Even can use a **fasttext** network to pre-train data.

## Network Structure

### HMC-LMLP

![](https://farm8.staticflickr.com/7851/39694328973_a89d7aef51_o.png)

References:

- [Reduction strategies for hierarchical
multi-label classification in protein function
prediction](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-016-1232-1)

---

### HMCN

#### HMCN-F

![](https://farm5.staticflickr.com/4917/32784591828_558aaba6a5_o.png)

#### HMCN-R

![](https://farm8.staticflickr.com/7916/45744120765_3ba324e59f_o.png)

References:

- [Hierarchical Multi-Label Classification Networks](http://proceedings.mlr.press/v80/wehrmann18a/wehrmann18a.pdf)

---

## About Me

黄威，Randolph

SCU SE Bachelor; USTC CS Master

Email: chinawolfman@hotmail.com

My Blog: [randolph.pro](http://randolph.pro)

LinkedIn: [randolph's linkedin](https://www.linkedin.com/in/randolph-%E9%BB%84%E5%A8%81/)
