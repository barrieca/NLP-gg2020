import json
import spacy
import time
import imdb
import pandas as pd

def find_imdb_person(df):
    imdb_obj = imdb.IMDb()
    noun = df.iloc[0]['word']
    result = imdb_obj.search_person(noun)
    if len(result) == 0 or result[0]['name'] != noun:
        return find_imdb_person(df.iloc[1:,])
    else:
        return noun

def aggregate_and_sort_df(df):
    df['freq'] = df.groupby('word')['word'].transform('count')
    return df.drop_duplicates().sort_values(by='freq', ascending=False)

def find_tweets_about_host(data, word_list = []):
    nlp = spacy.load('en_core_web_sm')
    for d in data:
        for i in d['text'].split(' '):
            if i[:4]=='host':
                var = nlp(d['text'])
                for word in [*var.noun_chunks]:
                    word = word.text.strip('•').strip(' ')
                    word_list.append(word)
                break
    return word_list

def main():
    data = [json.loads(line) for line in open('c:/Users/Acer/CSClasses/CS337/gg2020/gg2020.json','r',encoding='utf-8')]
    word_list = find_tweets_about_host(data)
    word_df = pd.DataFrame(word_list, columns=['word'])
    host = find_imdb_person(aggregate_and_sort_df(word_df))
    print(host)


t = time.time()
main()
print(time.time()-t)