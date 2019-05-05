from collections import defaultdict
from gensim import corpora
import gensim

## Pre-processing

documents = ["Human machine interface for lab abc computer applications", 
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]
stop_words = "for a of the and to in".split()

texts = []
for document in documents:
    words = []
    for word in document.lower().split():
        if word not in stop_words: words.append(word)
    texts.append(words)

freq = defaultdict(int)
for text in texts:
    for token in text:
        freq[token] += 1
texts = [[token for token in text if freq[token] > 1] for text in texts]

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]


## Learning
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=5, id2word=dictionary)

lda.save('tmp/lda_1st_try.model')
## ########<- <- <- <- HERE !!!!!!!!!!!!! 
lda_recall = gensim.models.ldamodel.LdaModel.load('tmp/lda_1st_try.model')


## Prediction
test_documents = ["Computer themselves and software yet to be developed will revolutionize the way we learn"]
test_texts = [[word for word in document.lower().split()] for document in test_documents]
test_corpus = [dictionary.doc2bow(text) for text in test_texts]

for topic_prob in lda_recall[test_corpus]:
    print(topic_prob)

