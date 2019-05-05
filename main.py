from recomend.topic_gen import topic_gen
from recomend.model_gen import model_gen

docs_train = ["Human machine interface for lab abc computer applications", 
              "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]
model_gen(docs_train)

document = ["Computer themselves and software yet to be developed will revolutionize the way we learn"]
topic_gen(document)

