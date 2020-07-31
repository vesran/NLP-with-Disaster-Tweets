# NLP-with-Disaster-Tweets

Real or not ? Predict which tweets are about real disasters and which ones are not.

## Getting started

### Introduction

This repository contains my solution for the <a href='https://www.kaggle.com/c/nlp-getting-started'>Kaggle's NLP disaster tweets classification </a> competition.
You may find several solutions I've came up with as well as an exploratory data analysis notebook.

Embeddings used :
* <a href='https://nlp.stanford.edu/projects/glove/'>GloVe</a>
* <a href='https://arxiv.org/pdf/1810.04805.pdf'>BERT</a>

### Problem's description

Twitter has become an important communication channel in times of emergency.
The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).

But, it’s not always clear whether a person’s words are actually announcing a disaster.
Take this example:

<img src='https://storage.googleapis.com/kaggle-media/competitions/tweet_screenshot.png' height=500>

The author explicitly uses the word “ABLAZE” but means it metaphorically. This is clear to a human right away, especially with the visual aid. But it’s less clear to a machine.

The goal is to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t. 
The dataset is composed of 10,000 tweets that were hand classified.

