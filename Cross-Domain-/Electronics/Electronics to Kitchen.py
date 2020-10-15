import nltk
import os
import re
import sklearn
import pandas as pd
import torch 
import numpy as np
!pip install transformers
!pip install deberta
from deBERTa import deberta
#side trace algorithms that will help in the computation ://
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.feature_selection import chi2,SelectKBest
from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier
nltk.download('punkt')
nltk.download('stopwords')

class DisentangledSelfAttention(torch.nn.Module):
def __init__(self, config):
        super().__init__()
 with open('../Bookstrain.txt','r',encoding='utf-8') as f:
    book_train = f.readlines()
    stopword = stopwords.words('english')
    self.bert = deberta.DeBERTa(pre_trained='base') 
    
    def preprocess(sentence):
    sentence = re.sub('[^\w\s]',' ',str(sentence))
    sentence = re.sub('[^a-zA-Z]',' ',str(sentence))
    new_sent = " "
    tok = word_tokenize(sentence)
    for i in range(len(tok)):
        if tok[i].lower() not in stopword:
            new_sent+=tok[i].lower()+" "
    return new_sent
    corpus_train = []
    label_train = np.zeros(1600)
    label_train[0:800] = 1
    for i in range(len(book_train)):
    sent = book_train[i]
    sent = sent[0:len(sent)-1]
    corpus_train.append(preprocess(sent))
    len(corpus_train)
    len(label_train)
    
  with open('../../Electronics/Dvdtest.txt','r',encoding='utf-8') as f:
    dvd_test = f.readlines()
label_test = np.zeros(400)
label_test[0:200] = 1
for i in range(400):
    sent = dvd_test[i]
    sent = sent[0:len(sent)-1]
    corpus_test.append(preprocess(sent))
    
    train_length = len(corpus_train)
test_length = len(corpus_test)
self.bert.apply_state()
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df = 2,max_df=0.8,use_idf=True,sublinear_tf=True,stop_words='english')
train_corpus_tf_idf = vectorizer.fit_transform(corpus_train)
test_corpus_tf_idf = vectorizer.transform(corpus_test)
vectorizer_chi2 = SelectKBest(chi2,k=5000)
chi_train_corpus_tf_idf = vectorizer_chi2.fit_transform(train_corpus_tf_idf,label_train)
chi_test_corpus_tf_idf = vectorizer_chi2.transform(test_corpus_tf_idf)

def forward(self, input_ids):

lr_classifier = LogisticRegression()
lr_classifier.fit(chi_train_corpus_tf_idf,label_train)
lr_pred = lr_classifier.predict(chi_test_corpus_tf_idf)
lr_acc = float((sum(lr_pred==label_test))/len(label_test))
lr_f1 = f1_score(lr_pred,label_test)
lr_cm = confusion_matrix(lr_pred,label_test)
print("The accuracy is :",lr_acc)
print("The f1 score is :",lr_f1)
print("confusion matrix is:")
print(lr_cm)

encoding = self.bert(input_ids)[-1]

from DeBERTa import deberta
tokenizer = deberta.GPT2Tokenizer()

#Bagging Classifier 
bg_classifier_lr = BaggingClassifier(base_estimator=lr_classifier,n_estimators=7,random_state=0,max_samples=0.5)
bg_classifier_lr.fit(chi_train_corpus_tf_idf,label_train)
bg_predict_lr = bg_classifier_lr.predict(chi_test_corpus_tf_idf)
bg_accuracy_lr = bg_classifier_lr.score(chi_test_corpus_tf_idf,label_test)
bg_f1 = f1_score(bg_predict_lr,label_test)
bg_cm_lr = confusion_matrix(label_test,bg_predict_lr)
print("The accuracy is : ",bg_accuracy_lr)
print("The f1 score is :",bg_f1)
print("The confusion matrix is:")
print(bg_cm_lr)
# We apply the same schema of special tokens as BERT, e.g. [CLS], [SEP], [MASK]
max_seq_len = 512
tokens = tokenizer.tokenize('Examples input text of DeBERTa')
# Truncate long sequence
tokens = tokens[:max_seq_len -2]
# Add special tokens to the `tokens`
tokens = ['[CLS]'] + tokens + ['[SEP]']
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_mask = [1]*len(input_ids)
# padding
paddings = max_seq_len-len(input_ids)
input_ids = input_ids + [0]*paddings
input_mask = input_mask + [0]*paddings
features = {
'input_ids': torch.tensor(input_ids, dtype=torch.int),
'input_mask': torch.tensor(input_mask, dtype=torch.int)
}
bo_classifier_lr = AdaBoostClassifier(base_estimator=lr_classifier,n_estimators=50,random_state=0,learning_rate=0.8)
bo_classifier_lr.fit(chi_train_corpus_tf_idf,label_train)
bo_predict_lr = bo_classifier_lr.predict(chi_test_corpus_tf_idf)
bo_accuracy_lr = bo_classifier_lr.score(chi_test_corpus_tf_idf,label_test)
bo_f1 = f1_score(bo_predict_lr,label_test)
bo_cm_lr = confusion_matrix(label_test,bo_predict_lr)
print("The accuracy is : ",bo_accuracy_lr)
print("The f1 score is :",bo_f1)
print("The confusion matrix is:")
print(bo_cm_lr)

bg_classifier_lr = BaggingClassifier(base_estimator=lr_classifier,n_estimators=7,random_state=0,max_features=0.85)
bg_classifier_lr.fit(chi_train_corpus_tf_idf,label_train)
bg_predict_lr = bg_classifier_lr.predict(chi_test_corpus_tf_idf)
bg_accuracy_lr = bg_classifier_lr.score(chi_test_corpus_tf_idf,label_test)
bg_f1 = f1_score(bg_predict_lr,label_test)
bg_cm_lr = confusion_matrix(label_test,bg_predict_lr)
print("The accuracy is : ",bg_accuracy_lr)
print("The f1 score is :",bg_f1)
print("The confusion matrix is:")
print(bg_cm_lr)


rbf_classifier = svm.SVC(gamma=0.9)
rbf_classifier.fit(chi_train_corpus_tf_idf,label_train)
rbf_pred = rbf_classifier.predict(chi_test_corpus_tf_idf)
rbf_accuracy = rbf_classifier.score(chi_test_corpus_tf_idf,label_test)
rbf_cm = confusion_matrix(rbf_pred,label_test)
rbf_f1 = f1_score(rbf_pred,label_test)
print("The SVM (rbf) Accuracy is: ",rbf_accuracy)
print("The F1 score is: ",rbf_f1)
print("The confusion matrix is:")
print(rbf_cm)

bg_classifier_rbf = BaggingClassifier(base_estimator=rbf_classifier,n_estimators=8,random_state=0,max_samples=0.8)
bg_classifier_rbf.fit(chi_train_corpus_tf_idf,label_train)
bg_predict_rbf = bg_classifier_rbf.predict(chi_test_corpus_tf_idf)
bg_accuracy_rbf = bg_classifier_rbf.score(chi_test_corpus_tf_idf,label_test)
bg_cm_rbf = confusion_matrix(label_test,bg_predict_rbf)
bg_f1_rbf = f1_score(label_test,bg_predict_rbf)
print("The accuracy is : ",bg_accuracy_rbf)
print("The F1 score is: ",bg_f1_rbf)
print("The confusion matrix is:")
print(bg_cm_rbf)

dt_classifier = DecisionTreeClassifier(random_state=0)
dt_classifier.fit(chi_train_corpus_tf_idf,label_train)
dt_pred = dt_classifier.predict(chi_test_corpus_tf_idf)
dt_accuracy = dt_classifier.score(chi_test_corpus_tf_idf,label_test)
dt_cm = confusion_matrix(dt_pred,label_test)
dt_f1 = f1_score(label_test,dt_pred)
print("The DT Accuracy is: ",dt_accuracy)
print("The F1 score is :",dt_f1)
print("The confusion matrix is:")
print(dt_cm)
bg_classifier_dt = BaggingClassifier(base_estimator=dt_classifier,n_estimators=8,random_state=0,max_samples=0.8)
bg_classifier_dt.fit(chi_train_corpus_tf_idf,label_train)
bg_predict_dt = bg_classifier_dt.predict(chi_test_corpus_tf_idf)
bg_accuracy_dt = bg_classifier_dt.score(chi_test_corpus_tf_idf,label_test)
bg_cm_dt = confusion_matrix(label_test,bg_predict_dt)
bg_f1_dt = f1_score(label_test,bg_predict_dt)
print("The accuracy is: ",bg_accuracy_dt)
print("The F1 score is: ",bg_f1_dt)
print("The confusion matrix is:")
print(bg_cm_dt)

linear_classifier = svm.SVC(kernel='linear')
linear_classifier.fit(chi_train_corpus_tf_idf,label_train)
linear_pred = linear_classifier.predict(chi_test_corpus_tf_idf)
linear_accuracy = linear_classifier.score(chi_test_corpus_tf_idf,label_test)
linear_cm = confusion_matrix(linear_pred,label_test)
linear_f1 = f1_score(label_test,linear_pred)
print("The SVM(linear) Accuracy is: ",linear_accuracy)
print("The F1 score is :",linear_f1)
print("The confusion matrix is:")
print(linear_cm)
bg_classifier_linear = BaggingClassifier(base_estimator=linear_classifier,n_estimators=8,random_state=0,max_samples=0.5)
bg_classifier_linear.fit(chi_train_corpus_tf_idf,label_train)
bg_predict_linear = bg_classifier_linear.predict(chi_test_corpus_tf_idf)
bg_accuracy_linear = bg_classifier_linear.score(chi_test_corpus_tf_idf,label_test)
bg_cm_linear = confusion_matrix(label_test,bg_predict_linear)
bg_f1_linear = f1_score(bg_predict_linear,label_test)
print("The accuracy is : ",bg_accuracy_linear)
print("The F1 score is :",bg_f1_linear)
print("The confusion matrix is:")
print(bg_cm_linear)

nb_classifier = MultinomialNB()
nb_classifier.fit(chi_train_corpus_tf_idf,label_train)
nb_acc = nb_classifier.score(chi_test_corpus_tf_idf,label_test)
nb_pred = nb_classifier.predict(chi_test_corpus_tf_idf)
nb_cm = confusion_matrix(nb_pred,label_test)
nb_f1 = f1_score(nb_pred,label_test)
print("Accuracy is :",nb_acc)
print("F1 score is :",nb_f1)
print("The confusion matrix is:")
print(nb_cm)
