#!/usr/bin/env python
# coding: utf-8

# ## CRAWLING DATA BERITA

# In[1]:


import scrapy
class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
            'https://nasional.sindonews.com/',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # print(response.url)

        for i in range(1, 30):
            for berita in response.css('body > div:nth-child(6) > section > div.grid_24 > div.homelist-new.scroll'):
                yield{
                    'Topik': berita.css(' ul > li.latest-event.latest-track-' + str(i) + ' > div.homelist-box > div.homelist-top > div.homelist-channel::text').extract(),
                    'Tanggal': berita.css(' ul > li.latest-event.latest-track-' + str(i) + ' > div.homelist-box > div.homelist-top > div.homelist-date::text').extract(),
                    'Judul': berita.css(' ul > li.latest-event.latest-track-' + str(i) + ' > div.homelist-box > div.homelist-title > a::text').extract(),
                    # 'link': response.css('body > div:nth-child(6) > section > div.grid_24 > div.homelist-new.scroll > ul > li.latest-event.latest-track-0 > div.homelist-box > div.homelist-title > a::@href').extract(),
                    'gambar': berita.css('ul > li.latest-event.latest-track-' + str(i) + ' > div.homelist-pict > a > img::text').extract(),
                    'Deskripsi': berita.css(' ul > li.latest-event.latest-track-' + str(i) + ' > div.homelist-box > div.homelist-desc::text').extract(),

                }


# # Import Library yang Diperlukan

# In[2]:


# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

#preprocessing
from nltk.corpus import stopwords  #stopwords
from nltk import word_tokenize,sent_tokenize # tokenizing
from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet

# for named entity recognition (NER)
from nltk import ne_chunk

# vectorizers for creating the document-term-matrix (DTM)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#stop-words
stop_words=set(nltk.corpus.stopwords.words('indonesian'))


# # Load Dataset

# In[2]:


df=pd.read_csv('scrapy_berita.csv')


# In[3]:


df.head()


# In[4]:


# drop the publish date.
df.drop(['Topik', 'Tanggal', 'Judul'],axis=1,inplace=True)


# In[5]:


df.head(10)


# # DATA CLEANING & PRE-PROCESSING

# ## Menghapus Angka

# In[6]:


import string 
import re #regex library

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist

#remove number
def remove_number(text):
    return  re.sub(r"\d+", "", text)

df['hapus angka'] = df['isi'].apply(remove_number)
df.head(10)


# ## Menghapus Simbol dan tanda Baca

# In[7]:


#remove punctuation(simbol dan tanda baca)
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

df['hapus simbol'] = df['hapus angka'].apply(remove_punctuation)
df.head(10)


# ## Stopword

# Stopword adalah kata-kata umum yang sering muncul, yang tidak memberikan informasi penting (biasanya tidak diacuhkan atau dibuang misalnya dalam proses pembuatan indeks atau daftar kata)

# In[26]:


def clean_text(headline):
  le=WordNetLemmatizer()
  word_tokens=word_tokenize(headline)
  tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
  cleaned_text=" ".join(tokens)
  return cleaned_text


# In[9]:


# time taking
df['stopword']=df['hapus simbol'].apply(clean_text)


# In[10]:


df.head(10)


# In[11]:


df.drop(['isi', 'hapus angka', 'hapus simbol'],axis=1,inplace=True)
df.head(10)


# In[12]:


df['stopword'][0]


# ## MENGEKSTRAK FITUR DAN MEMBUAT DOCUMENT-TERM-MATRIX ( DTM )
# Dalam DTM nilainya adalah nilai TFidf.
# 
# Parameter dari vectorizer Tfidf.
# 
# Beberapa poin penting:-
# 
# 1) LSA umumnya diimplementasikan dengan nilai Tfidf di mana-mana dan tidak dengan Count Vectorizer.
# 
# 2) max_features tergantung pada daya komputasi Anda dan juga pada eval. metrik (skor koherensi adalah metrik untuk model topik). Coba nilai yang memberikan evaluasi terbaik. metrik dan tidak membatasi kekuatan pemrosesan.
# 
# 3) Nilai default untuk min_df & max_df bekerja dengan baik.
# 
# 4) Dapat mencoba nilai yang berbeda untuk ngram_range.

# In[13]:


vect =TfidfVectorizer(stop_words=stop_words,max_features=1000)


# In[14]:


vect_text=vect.fit_transform(df['stopword'])


# In[15]:


print(vect_text.shape)
print(vect_text)


# In[16]:


idf=vect.idf_


# In[17]:


dd=dict(zip(vect.get_feature_names(), idf))
l=sorted(dd, key=(dd).get)
# print(l)
print(l[0],l[-1])


# ## LSA
# LSA pada dasarnya adalah dekomposisi nilai tunggal.
# 
# SVD menguraikan DTM asli menjadi tiga matriks S=U.(sigma).(V.T). Di sini matriks U menunjukkan matriks dokumen-topik sementara (V) adalah matriks topik-term.
# 
# Setiap baris dari matriks U (matriks istilah dokumen) adalah representasi vektor dari dokumen yang sesuai. Panjang vektor ini adalah jumlah topik yang diinginkan. Representasi vektor untuk suku-suku dalam data kami dapat ditemukan dalam matriks V (matriks istilah-topik).
# 
# Jadi, SVD memberi kita vektor untuk setiap dokumen dan istilah dalam data kita. Panjang setiap vektor adalah k. Kami kemudian dapat menggunakan vektor-vektor ini untuk menemukan kata-kata dan dokumen serupa menggunakan metode kesamaan kosinus.
# 
# Kita dapat menggunakan fungsi truncatedSVD untuk mengimplementasikan LSA. Parameter n_components adalah jumlah topik yang ingin kita ekstrak. Model tersebut kemudian di fit dan ditransformasikan pada hasil yang diberikan oleh vectorizer.
# 
# Terakhir perhatikan bahwa LSA dan LSI (I untuk pengindeksan) adalah sama dan yang terakhir kadang-kadang digunakan dalam konteks pencarian informasi.

# In[18]:


from sklearn.decomposition import TruncatedSVD
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# In[19]:


print(lsa_top)
print(lsa_top.shape)  # (no_of_doc*no_of_topics)


# In[20]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)


# In[21]:


print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)


# Sekarang bisa mendapatkan daftar kata-kata penting untuk masing-masing dari 10 topik seperti yang ditunjukkan. Untuk kesederhanaan di sini saya telah menunjukkan 10 kata untuk setiap topik

# In[24]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:15]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")

