import pandas
import pandas as pd
import os, sys
import json
from bs4 import BeautifulSoup
#from pyhanlp import *
import jieba.analyse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from gensim.models import KeyedVectors
from sklearn.decomposition import LatentDirichletAllocation
import jieba.posseg as pseg
import numpy as np


def process_AI(file_path):
    with open(file_path, "r", encoding="utf-8") as f1, open("ai_data.txt", "w", encoding="utf-8") as f2:
        for line in f1:
            url, category, time, location, data = line.split("\t")
            data = json.loads(data)
            try:
                content = BeautifulSoup(data["data"]["内容"], features="lxml").text
            except Exception as e:
                print(e)
            res = data["data"]
            res["内容"] = "".join(content.split()).replace("/n", " ")
            f2.write(json.dumps(res) + "\n")


def process_news(root_path):
    with open("news_data.txt", "w", encoding="utf-8") as out:
        for file_path in os.listdir(root_path):
            file_path = os.path.join(root_path, file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    # data = json.load(f)
                    pass
                except Exception as e:
                    print(e)
                for line in f:
                    out.write("".join(line.split()).replace("/n", " ") + "\n")


def process_keyword(data, num=100):
    #return HanLP.extractKeyword(data[1], num)
    pass


def keyword_jieba(word_dict, k, stop_words):
    res = {}
    jieba.load_userdict(word_dict_path)
    text_rank = jieba.analyse.TextRank()
    # text_rank.tokenizer = jieba.Tokenizer( dictionary=open(word_dict_path, "r", encoding="utf-8"))
    if text_rank.tokenizer.makesure_userdict_loaded():
        print("load dictionary successful")
    else:
        print("failed to load dictionary")
    with open("/home/harry/document/WordVec/cover/ai_data.txt", "r", encoding="utf-8") as f, \
            open("ai_key_" + str(k) + ".txt", "w", encoding="utf-8") as f2:
        texts = ""
        for line in f:
            text = json.loads(line)["内容"]
            texts += text
            # tmp_ = tf_idf.extract_tags(text, topK=k, withWeight=True, allowPOS=('ns', 'n', 'nr', 'nt', 'nz', 'un'))
            tmp_ = text_rank.extract_tags(text, topK=k, withWeight=True, allowPOS=('ns', 'n', 'nr', 'nt', 'nz', 'un'))
            for word in tmp_:
                if word[1] < 0.1:
                    break
                if word[0] in res:
                    res[word[0]] += word[1]
                else:
                    res[word[0]] = word[1]
        res = list(res.items())
        res = sorted(res, key=lambda item: item[1])
        res.reverse()
        # res_ = text_rank.extract_tags(texts, topK=800, withWeight=True, allowPOS=('ns', 'n', 'nr', 'nt', 'nz', 'un'))
        json.dump(res, f2, ensure_ascii=False, indent=2, separators=(",", ":"))


def keyword_sklearn(word_dict_path, k, data_path, stop_words):
    # vectorizer = CountVectorizer()
    # transformer = TfidfTransformer()
    texts = []
    stop_words = list(stop_words)
    count = 0
    with open(data_path, "r", encoding="utf-8") as f, open("ai_key_sk_" + str(k) + ".txt", "w", encoding="utf-8") as f1:
        for line in f:
            text = " ".join(jieba.cut(json.loads(line)["内容"]))
            count += 1
            if count == 2000:
                break
            texts.append(text)
    # tf_idf = transformer.fit_transform(vectorizer.fit_transform(texts))
    print(count)
    tfidf_model = TfidfVectorizer(stop_words=list(stop_words))
    tf_idf = tfidf_model.fit_transform(texts)

    # for i in range(len(weightlist)):
    #     print("-------第", i, "段文本的词语tf-idf权重------")
    #     for j in range(len(wordlist)):
    #         print(wordlist[j], weightlist[i][j])
    # print(tf_idf)

    # print(tf_idf.todense())
    def get_topk(tfidf, vectorizer, k):
        tfidf_array = tfidf.toarray()
        tfidf_sorted = np.argsort(-tfidf_array, axis=1)[:, :k]
        names = vectorizer.get_feature_names()
        keywords = pandas.Index(names)[tfidf_sorted].values
        print(tfidf_sorted.shape)
        for i in range(5):
            #print("\n The {} document".format(i))
            for j in range(tfidf_sorted.shape[1]):
                print(keywords[i][j], tfidf_array[i, tfidf_sorted[i][j]], end="")
            print()
    get_topk(tf_idf, tfidf_model, 10)


def keyword_gensim(word_embedding_path, k):
    wv = KeyedVectors.load(word_embedding_path)
    print()
    pass


def lda(data_path, stop_word_path, k=2):
    with open(data_path, "r", encoding="utf8") as f, open(stop_word_path, "r", encoding="utf-8") as f1:
        texts = []
        stop_words = set()
        pos_seg = set(["n", "nt", "nz", "nr"])
        for word in f1:
            stop_words.add(word.strip())
        for line in f:
            text_ = json.loads(line)["内容"].strip()

            for word_ in pseg.cut(text_):
                text_ = []
                if word_.flag in pos_seg and word_.word not in stop_words:
                    text_.append(word_.word)
            texts.append(" ".join(text_))
        countvectorizer = CountVectorizer()
        countvector = countvectorizer.fit_transform(raw_documents=texts)
        lda = LatentDirichletAllocation(n_topics=k)
        res_doc = lda.fit_transform(countvector)

        def print_topk_words(model, feature_names, n_top_words):
            for topic_idx, topic in enumerate(model.components_):
                print("Topic #%d:" % topic_idx)
                print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
                print()

        print_topk_words(lda, countvectorizer.get_feature_names(), 20)


if __name__ == "__main__":
    # ai_file_path = r"/home/harry/Document/WordVec/Data/AbstractAi.data"
    # process_AI(ai_file_path)
    # news_path = "/home/harry/Document/WordVec/Data/pubnews_striped"
    # process_news(news_path)
    k = 30
    # tf_idf = jieba.analyse.TFIDF(idf_path=None)
    word_dict_path = "/home/harry/document/WordVec/cover/Data/tencent.w2v.words.txt"
    data_path = "/home/harry/document/WordVec/cover/Data/ai_data.txt"
    word_embedding_path = "/home/harry/document/WordVec/cover/Data/Tencent_AILab_ChineseEmbedding.txt"
    stop_word_path = "/home/harry/document/WordVec/cover/Data/stop_word/stop_words"
    stop_words = set()
    with open(stop_word_path, "r", encoding="utf-8") as f1:
        for word_ in f1:
            stop_words.add(word_.strip())
    jieba.load_userdict(word_dict_path)
    jieba.enable_parallel(4)
    # keyword_jieba(word_dict_path,k)
    keyword_sklearn(word_dict_path, k, data_path, stop_words)
    # keyword_gensim(word_embedding_path,k)
    # lda(data_path, stop_word_path, 8)
