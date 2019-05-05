from collections import defaultdict
from gensim import corpora
import gensim

def model_gen(documents):
    ## Prepare docs as traing data
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
    
    ## Transform docs into list of words
    ## Remove stop words
    texts = []
    for document in documents:
        words = []
        for word in document.lower().split():
            if word not in stop_words: words.append(word)
        texts.append(words)
    
    ## Remove low-freq words
    freq = defaultdict(int)
    for text in texts:
        for token in text:
            freq[token] += 1
    texts = [[token for token in text if freq[token] > 1] for text in texts]
    
    ## Create dictionary
    dictionary = corpora.Dictionary(texts)
    
    ## Create corpus
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    ## Growing lda model
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=5, id2word=dictionary)
    lda.save('.recomend/lda.model')
    

if __name__ == '__main__':
    pass
