import gensim

def topic_gen(documents):
    ## Load model
    lda_model = gensim.models.ldamodel.LdaModel.load('recomend/lda.model')
    dictionary = lda_model.id2word
    
    ## Transform docs into corpus
    texts = [[word for word in document.lower().split()] for document in documents]
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    ## for develop
    for topic_prob in lda_model[corpus]:
        print(topic_prob)

    ## Topic prediction and sort
    topic_prob = list(lda_model[corpus])[0]
    topic_prob.sort(key=lambda x: x[1], reverse=True)
    topic_line_up = [l[0] for l in topic_prob]

    ## Return
    print(topic_line_up)
    

if __name__ == '__main__':
    pass
