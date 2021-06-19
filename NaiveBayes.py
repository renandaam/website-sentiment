import pandas as pd
from sklearn import model_selection, naive_bayes, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

Corpus = pd.read_csv("TranslateDataVaksin2021x.csv")
tweetsValue = Corpus['Tweets'].values.astype('U')
sentimentValue = Corpus['Sentiment']

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(tweetsValue, sentimentValue, test_size=0.2, random_state=1) # 70:30 70 = Training Data 30 = Testing Data

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(strip_accents=ascii, lowercase=True, stop_words="english")
labelEncoder = LabelEncoder()
X_train_cv = cv.fit_transform(Train_X)
X_test_cv = cv.transform(Test_X)

word_freq_df = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names())
top_words_df = pd.DataFrame(word_freq_df.sum()).sort_values(0, ascending=False)

naive_bayes = naive_bayes.MultinomialNB()
naive_bayes.fit(X_train_cv, Train_Y)
predictions = naive_bayes.predict(X_test_cv)

from sklearn.metrics import accuracy_score, precision_score, recall_score
print("Accuracy score: ", accuracy_score(Test_Y, predictions))
print("Precision score: ", precision_score(Test_Y, predictions, average="binary", pos_label="neg"))
print("Recall score: ", recall_score(Test_Y, predictions, average="binary", pos_label="neg"))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
cm = confusion_matrix(Test_Y, predictions)
cr = classification_report(Test_Y, predictions)
print(cm)
print(cr)
# # Import Gaussian Naive Bayes model
# from sklearn.naive_bayes import GaussianNB# Mengaktifkan/memanggil/membuat fungsi klasifikasi Naive bayes
# modelnb = GaussianNB()# Memasukkan data training pada fungsi klasifikasi naive bayes
# nbtrain = modelnb.fit(Train_X, Train_Y)
#
# # Menentukan hasil prediksi dari x_test
# Pred_Y = nbtrain.predict(Test_X)
#
# print(confusion_matrix(Test_Y, Pred_Y))
# print(classification_report(Test_Y,Pred_Y))
#
