#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat May 23 12:14:40 2020
@author: FY
"""

# ------------------ Import Library and Modules ------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

#-----------------------------------------------------------------


# ------------------ CSV Dosyalarını Aldık -----------------------

metadata = pd.read_csv('ISIC_2019_Training_Metadata.csv')
groundtruth = pd.read_csv('ISIC_2019_Training_GroundTruth.csv')

# ----------------------------------------------------------------


# ------ CEVAP-1  ------ 
# Analiz için ödevde belirtilen gerekli özellikler haricinde gerekli olmayan özelliklerin silinmesi :
metadata.drop('lesion_id',axis = 1, inplace = True)
groundtruth.drop('image', axis = 1, inplace = True)


# ------ CEVAP-3  ------ 
# Meta-data dosyası ile etiketlerin olduğu (ground-truth) dosyayı birleştiren, 
# Bu birleştirme sonucunda her bir örneğin karşısında ait olduğu sınıfı yazan kod :
frame = pd.concat([metadata,groundtruth], axis = 1)
sınıf_liste = pd.DataFrame(columns = ['Class'], index = range(25331))
dizi = np.array(frame)

a = 0
for i in range(len(dizi)):
    for j in range(len(dizi[i])):
        if dizi[i][j] == 1:
            dizi[i][j] = frame.columns[j]
            sınıf_liste.Class[a] = frame.columns[j]
    a+=1
print(dizi)

sonuc = pd.concat([frame,sınıf_liste],axis = 1)
sonuc = sonuc.dropna() # ------ CEVAP-2 NaN değerleri datadan kaldırır
sonuc.drop(['NV', 'MEL', 'BCC','AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK'], axis = 1, inplace = True)
sonuc2 = pd.concat([frame,sınıf_liste], axis = 1)
sonuc2 = sonuc2.dropna()
print(sonuc2)


# ------ CEVAP-12  ------
# Verideki kategorik değerleri nümerik (sayısal) hale getiren kod :
gender = sonuc.iloc[:,3:4].values 
le = LabelEncoder()
gender[:, 0] = le.fit_transform(gender[:,0])
ohe = ColumnTransformer([('gender', OneHotEncoder(), [0])], remainder = 'passthrough')
gender = ohe.fit_transform(gender)
gender = pd.DataFrame(data = gender, index = range(22480), columns=['male', 'female' ])
print(gender)

anatom_site = sonuc.iloc[:, 2:3].values
anatom_site[:, 0] = le.fit_transform(anatom_site[:, 0])
ohe = ColumnTransformer([('anatom_site', OneHotEncoder(), [0])], remainder = 'passthrough')
anatom_site = ohe.fit_transform(anatom_site).toarray()
anatom_site = pd.DataFrame(data = anatom_site, index = range(22480), columns=['anterior_torso', 'head_neck','lateral_torso','lower_extremity','oral_genital','palms_soles','posterior_torso','upper_extremity'])
print(anatom_site)

sonuc.drop(['gender', 'anatom_site_general'], axis = 1, inplace = True)
sonuc = pd.concat([sonuc,anatom_site,gender], axis = 1)
sonuc = sonuc.dropna()
sinif = sonuc.iloc[:, 2:3].values
sinif = pd.DataFrame(data = sinif, index = range(19683), columns = ['Class'])
image = sonuc.iloc[:, 0:1].values
image = pd.DataFrame(data = image, index = range(19683), columns = ['image'])
sonuc.drop(['image', 'Class'], axis = 1, inplace = True)


# ------ CEVAP-11  ------  
# Age Kolonu İçin Veriyi ölçekleyen kod
age = sonuc.iloc[:, 0:1].values
age = pd.DataFrame(data = sonuc, index = range(19683), columns = ['age_approx'])
sc = StandardScaler()
age = sc.fit_transform(age)
print(age)


# ------ CEVAP -10  ------
# Veriyi %80 eğitim %20 test olacak şekilde bölen kod
x_train, x_test,y_train, y_test = train_test_split(sonuc, sinif, test_size = 0.2, random_state = 0)


# ------ CEVAP-13  ------
# Logistic Regression
model = LogisticRegression(solver='liblinear', random_state = 0)
model.fit(x_train, y_train)
print(confusion_matrix(y_test, model.predict(x_test)))
print(classification_report(y_test, model.predict(x_test)))


# ------ CEVAP-14  ------ 
# Support Vector Machine :
svClassifier = SVC(kernel = 'linear')
svClassifier.fit(x_train, y_train)
y_pred = svClassifier.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ------ CEVAP-15  ------  
# K-NN Algorithm :
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ------ CEVAP-16  ------ 
# Naive Bayes Algorithm : 
nbClassifier = MultinomialNB()
nbClassifier.fit(x_train, y_train)
y_predict = nbClassifier.predict(x_test)
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))


# ------ CEVAP-17  ------ 
# Decision Tree Classification :
dtClassifier = DecisionTreeClassifier()
dtClassifier.fit(x_train, y_train)
y_pred = dtClassifier.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ------ CEVAP-18  ------ 
# Neural Network Classifier :
clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (7, 2), random_state = 1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ------ CEVAP-4  ------ 
# Verideki lezyon türlerinin dağılımını gösteren grafiği çizdiren kod :
countClasses = dict(
    NV = 0,
    MEL = 0,
    BKL = 0,
    BCC = 0,
    VASC = 0,
    DF = 0,
    AK = 0,
    SCC = 0
)

classList = np.array(sinif)

for image in classList:
    countClasses[image[0]] += 1

print(countClasses)
countClasses = pd.DataFrame.from_dict(countClasses, orient = 'index', columns = ['Deger'])

y = countClasses.Deger
x = countClasses.index
plt.title("Lezyon Türlerinin Dağılımı")
plt.xlabel("Lezyonlar")
plt.ylabel("Lezyon Sayısı")
plt.bar(x,y, color = ('#fde0dd', '#ece7f2', '#d0d1e6', '#a6bddb', '#74a9cf', '#3690c0', '#0570b0', '#034e7b'))
plt.legend()
plt.show()


# ------ CEVAP-5  ------ 
# Verideki lezyonun bulunduğu bölgeye göre dağılımını gösteren grafiği çizdiren kod :
plt.title("Lezyonun Bulunduğu Bölgeye Göre Dağılımı")
plt.xlabel("Anatom Site General")
plt.ylabel("Lezyonlar")
x = sonuc2.Class
y = sonuc2.anatom_site_general
plt.bar(x,y, color = ('#fde0dd', '#ece7f2', '#d0d1e6', '#a6bddb', '#74a9cf', '#3690c0', '#0570b0', '#034e7b'))
plt.legend()
plt.show()


# ------ CEVAP-6  ------ 
# Verideki cinsiyet dağılımını gösteren grafiği çizdiren kodu :
genderClasses = dict(
    male = 0,
    female = 0
)
classList = np.array(sonuc2)

for image in classList:
    genderClasses[image[3]] += 1
print(genderClasses)

genderClasses = pd.DataFrame.from_dict(genderClasses, orient = 'index', columns = ['Deger'])

plt.title("Lezyonun Cinsiyete Göre Dağılımı")
plt.xlabel("Cinsiyet")
plt.ylabel("Lezyon Sayısı")
x = genderClasses.index
y = genderClasses.Deger
plt.bar(x,y, color = ('#a6bddb', '#74a9cf'))
plt.legend()
plt.show()


# ------ CEVAP-8  ------ 
# Verideki sınıfı MEL (melanoma) olup cinsiyet dağılımını gösteren grafiği çizdiren kod :
melonoma = sonuc2[sonuc2['Class'] == 'MEL']
melgenderClasses = dict(
    male = 0,
    female = 0
)

melClassList = np.array(melonoma)

for image in melClassList:
    melgenderClasses[image[3]] += 1
print(melgenderClasses)

melgenderClasses = pd.DataFrame.from_dict(melgenderClasses, orient = 'index', columns = ['Deger'])
plt.title("Lezyon Türü Melonoma Olup Cinsiyete Göre Dağılım Grafiği")
plt.xlabel("Cinsiyet")
plt.ylabel("Lezyon Değeri")
x = melgenderClasses.index
y = melgenderClasses.Deger
plt.bar(x,y, color = ('#ece7f2', '#d0d1e6'))
plt.legend()
plt.show()


# ------ CEVAP-9  ------ 
# Verideki sınıfı MEL (melanoma) olup lezyonun bulunduğu bölgeye göre dağılımını gösteren grafiği çizdiren kod :
melonoma = sonuc2[sonuc2['Class'] == 'MEL']

anatomCountClasses = dict(
    anteriortorso = 0,
    headneck = 0,
    lateraltorso = 0,
    lowerextremity = 0,
    oralgenital = 0,
    palmssoles = 0,
    posteriortorso = 0,
    upperextremity = 0
)

anatomClassList = np.array(melonoma)

for image in anatomClassList:
    anatomCountClasses[image[2].replace(' ', '').replace("/", "")] += 1

print(anatomCountClasses)

anatomCountClasses = pd.DataFrame.from_dict(anatomCountClasses, orient = 'index',columns = ['Deger'])
x = anatomCountClasses.index
y = anatomCountClasses.Deger

plt.title("Lezyon Türü Melonoma Olup Bulunduğu Bölgeye Göre Dağılım Grafiği")
plt.xlabel("Değerler")
plt.ylabel("Lezyonun Bulunduğu Bölge")
plt.barh(x,y, color = ('#fde0dd', '#ece7f2', '#d0d1e6', '#a6bddb', '#74a9cf', '#3690c0', '#0570b0', '#034e7b'))
plt.legend()
plt.show()
