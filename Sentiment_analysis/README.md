# Sentiment Analysis

Thanks to modern technologies, we are now able to collect and analyze such data 
most efficiently. In this chapter, we will delve into a subfield of natural language processing (NLP) 
called sentiment analysis and learn how to use machine learning algorithms to classify documents based 
on their polarity: the attitude of the writer. In particular, we are going to work with a dataset of 
50,000 movie reviews from the Internet Movie Database (IMDb) and build a predictor that can distinguish between positive and negative reviews

The topics that we will cover in the following sections include the following:
- Cleaning and preparing text data
- Building feature vectors from text documents
- Training a machine learning model to classify positive and negative movie reviews
- Working with large text datasets using out-of-core learning
- Inferring topics from document collections for categorization

The movie review dataset consists of 50,000 polar movie reviews that are labeled as either positive 
or negative; here, positive means that a movie was rated with more than six stars on IMDb, and negative 
means that a movie was rated with fewer than five stars on IMDb.


### Transforming words into feature vectors

CountVectorizer - takes an array of text data, which can be documents or sentences, and constructs the bag-of-words model for us

Each index position in the feature vectors shown here corresponds to the integer values that are stored as dictionary 
items in the CountVectorizer vocabulary.
For example, the first feature at index position 0 resembles the count of the word 'and', which only occurs in the 
last document, and the word 'is', at index position 1 (the second feature in the document vectors), occurs in all three sentences.
These values in the feature vectors are also called the raw term frequencies: tf(t, d)—the number of times a 
term, t, occurs in a document, d. It should be noted that, in the bag-of- words model, the word or term order in a sentence 
or document does not matter. The order in which the term frequencies appear in the feature vector is derived from the vocabulary
 indices, which are usually assigned alphabetically.
To summarize the concept of the n-gram representation, the 1-gram and 2-gram representations of our first document 
"the sun is shining" would be constructed as follows:
- 1-gram: "the", "sun", "is", "shining"
- 2-gram: "the sun", "sun is", "is shining"

### Assessing word relevancy via 'term frequency-inverse document frequency' (td-idf)

can be used to downweight these frequently occurring words in the feature vectors

The word 'is' had the largest term frequency in the third document, being the most frequently occurring word.

However, after transforming the same feature vector into tf-idfs, the word 'is' is now associated with a 
relatively small tf-idf (0.45) in the third document, since it is also present in the first and second 
document and thus is unlikely to contain any useful discriminatory information


# Working with bigger data – online algorithms and out-of-core learning

We will now define a function, get_minibatch, that will take a document stream from the stream_docs function 
and return a particular number of documents specified by the size parameter

Unfortunately, we can't use CountVectorizer for out-of-core learning since it requires holding the complete vocabulary 
in memory. Also, TfidfVectorizer needs to keep all the feature vectors of the training dataset in memory to calculate 
the inverse document frequencies. However, another useful vectorizer for text processing implemented in scikit-learn 
is HashingVectorizer. HashingVectorizer is data-independent and makes use of the hashing trick via the 32-bit MurmurHash3 
function by Austin Appleby

We initialized the progress bar object with 45 iterations and, in the following for loop, we iterated over 
45 mini-batches of documents where each mini-batch consists of 1,000 documents. Having completed the incremental 
learning process, we will use the last 5,000 documents to evaluate the performance of our model

## Topic modeling with Latent Dirichlet Allocation (LDA)

Topic modeling describes the broad task of assigning topics to unlabeled text documents. For example, 
a typical application would be the categorization of documents in a large text corpus of newspaper articles.
 In applications of topic modeling, we then aim to assign category labels to those articles, for example, sports, 
 finance, world news, politics, local news, and so forth. Thus, in the context of the broad categories of machine 
 learning we can consider topic modeling as a clustering task, a subcategory of unsupervised learning.

LDA is a generative probabilistic model that tries to find groups of words that appear frequently together across 
different documents. These frequently appearing words represent our topics, assuming that each document is a mixture of different words.

The input to an LDA is the bag-of-words model that we discussed earlier in this chapter. Given a bag-of-words matrix 
as input, LDA decomposes it into two new matrices:
- A document-to-topic matrix
- A word-to-topic matrix

LDA decomposes the bag-of-words matrix in such a way that if we multiply those two matrices together, we will be able 
to reproduce the input, the bag-of-words matrix, with the lowest possible error. In practice, we are interested in 
those topics that LDA found in the bag-of-words matrix. The only downside may be that we must define the number of 
topics beforehand—the number of topics is a hyperparameter of LDA that has to be specified manually.


To analyze the results, let's print the five most important words for each of the
10 topics. Note that the word importance values are ranked in increasing order. 
Thus, to print the top five words, we need to sort the topic array in reverse order

Topic 1:

worst minutes awful script stupid

Topic 2:

family mother father children girl

Topic 3:

american war dvd music tv

Topic 4:

human audience cinema art sense

Topic 5:

police guy car dead murder

Topic 6:

horror house sex girl woman

Topic 7:

role performance comedy actor performances

Topic 8:

series episode war episodes tv

Topic 9:

book version original read novel

Topic 10:

action fight guy guys cool


Based on reading the 5 most important words for each topic, we may guess that the LDA identified the following topics:

Generally bad movies (not really a topic category)

- Movies about families
- War movies
- Art movies
- Crime movies
- Horror movies
- Comedies
- Movies somehow related to TV shows
- Movies based on books
- Action movies

To confirm that the categories make sense based on the reviews, let's plot 5 movies from the horror movie category 
(category 6 at index position 5):

Horror movie #1:
House of Dracula works from the same basic premise as House of Frankenstein from the year before; 
namely that Universal's three most famous monsters; Dracula, Frankenstein's Monster and The Wolf Man 
are appearing in the movie together. Naturally, the film is rather messy therefore, but the fact that ...

Horror movie #2:
Okay, what the hell kind of TRASH have I been watching now? "The Witches' Mountain" has got to be one 
of the most incoherent and insane Spanish exploitation flicks ever and yet, at the same time, it's also 
strangely compelling. There's absolutely nothing that makes sense here and I even doubt there  ...

Horror movie #3:
<br /><br />Horror movie time, Japanese style. Uzumaki/Spiral was a total freakfest from start to 
finish. A fun freakfest at that, but at times it was a tad too reliant on kitsch rather than the horror. 
The story is difficult to summarize succinctly: a carefree, normal teenage girl starts coming fac ...


