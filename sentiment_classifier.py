import nltk
from nltk.corpus import wordnet, movie_reviews
import random
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.tokenize import word_tokenize
from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        
        choice_votes = votes.count(mode(votes))
        conf = choice_votes/len(votes)
        return conf


short_pos = open("positive.txt","r").read()
short_neg = open("negative.txt","r").read()

documents = []
all_words = []


allowed_word_types = ["J"]

tokenizer = nltk.RegexpTokenizer(r"\w+")

for p in short_pos.split('\n'):
    documents.append((p,"pos"))
    words = tokenizer.tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in short_neg.split('\n'):
    documents.append((p,"neg")) 
    words = tokenizer.tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
    

save_documents = open("documents.pickle","wb")
pickle.dump(documents, save_documents)

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

save_word_features = open("word_features.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(document):
    words = tokenizer.tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
        
    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]
save_featuresets = open("featuresets.pickle","wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()


random.shuffle(featuresets)
print(len(featuresets))


training_set = featuresets[:10000]
testing_set = featuresets[10000:]


classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo Accuracy : ", (nltk.classify.accuracy(classifier, testing_set))*100 , "%")
classifier.show_most_informative_features(15)

save_classifier = open("originalnaivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

#Multinomial Naive Bayes Algorithm
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("Multinomial Naive Bayes Algo Accuracy : ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100 , "%")

save_classifier = open("MNB_classifier.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()


#Bernoulli Naive Bayes Algorithm
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("Bernoulli Naive Bayes Algo Accuracy : ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100 , "%")

save_classifier = open("BNB_classifier.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()


#LogisticRegression Algorithm
LogisticRegression_classifier = SklearnClassifier(LogisticRegression(max_iter = 1500))
LogisticRegression_classifier.train(training_set)
print("LogisticRegression Algo Accuracy : ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100 , "%")

save_classifier = open("LogisticRegression_classifier.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()


#SGDClassifier Algorithm
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier Algo Accuracy : ", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100 , "%")

save_classifier = open("SGDC_classifier.pickle","wb")
pickle.dump(SGDClassifier_classifier, save_classifier)
save_classifier.close()

#SVC Algorithm
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC Algo Accuracy : ", (nltk.classify.accuracy(SVC_classifier, testing_set))*100 , "%")

save_classifier = open("SVC_classifier.pickle","wb")
pickle.dump(SVC_classifier, save_classifier)
save_classifier.close()

#LinearSVC Algorithm
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC Algo Accuracy : ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100 , "%")

save_classifier = open("LinearSVC_classifier.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

#NuSVC Algorithm
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC Naive Bayes Algo Accuracy : ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100 , "%")

save_classifier = open("NuSVC_classifier.pickle","wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()a

voted_classifier = VoteClassifier(MNB_classifier, BernoulliNB_classifier, LogisticRegression_classifier, SGDClassifier_classifier,
                                SVC_classifier, LinearSVC_classifier, NuSVC_classifier)
print("Voted Classifier Accuracy : ", (nltk.classify.accuracy(voted_classifier, testing_set))*100 , "%")

print("Classification :", voted_classifier.classify(testing_set[0][0]), "Confidence : ", voted_classifier.confidence(testing_set[0][0])*100, "%")


