from collections import defaultdict
from gensim import corpora
import gensim

def model_gen(documents):
    ## Prepare stop words
    # stop_words = "for a of the and to in".split()
    with open("recomend/stop_words.txt") as f:
        stop_words = f.read()
        stop_words = stop_words.split()
    ## for develop
    print("Stop words: " + str(stop_words))


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
    lda.save('recomend/lda.model')
    print("################## Recomendation Engine: Topic model files (lda.model ect...) generated.")
    

if __name__ == '__main__':
    pass
