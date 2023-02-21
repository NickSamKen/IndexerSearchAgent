# Simple indexer and search engine built on an inverted-index and the BM25 ranking algorithm.
import os
import code
from collections import defaultdict, Counter
import pickle
import math
import operator


from tqdm import tqdm
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from datasets import load_dataset


class Indexer:
    dbfile = "./ir.idx"

    def __init__(self):
        # TODO. You will need to create appropriate data structures for the following elements
        self.tok2idx = defaultdict(lambda: len(self.tok2idx))  # map (id to token)
        self.idx2tok = dict()  # self.idx2tok = None
        self.postings_lists = dict()  # postings for each word
        self.docs = []  # encoded document list
        self.raw_ds = None  # raw documents for result presentation
        self.corpus_stats = {'avgdl': 0}  # any corpus-level statistics
        self.stopwords = stopwords.words('english')  # self.stopwords = stopwords.words('english')

        if os.path.exists(self.dbfile):
            # TODO. If This exists a saved corpus index file, load it.
            # (You may use pickle to save and load a python object.)
            # Load
            index = pickle.load(open(self.dbfile, 'rb'))
            self.tok2idx = index['tok2idx']
            self.idx2tok = index['idx2tok']
            self.docs = index['docs']
            self.raw_ds = index['raw_ds']
            self.postings_lists = index['postings']
            self.corpus_stats['avgdl'] = index['avgdl']
        else:
            # TODO. Load CNN/DailyMail dataset, preprocess and create  posting lists.
            ds = load_dataset("cnn_dailymail", '3.0.0', split="test")
            self.raw_ds = ds['article']
            self.clean_text(self.raw_ds)
            self.create_postings_lists()

    def clean_text(self, lst_text, query=False):
        # TODO. this function will run in two modes: indexing and query mode.
        # TODO. run simple whitespace-based tokenizer (e.g., RegexpTokenizer)
        tokenizer = RegexpTokenizer(r'\w+')
        # TODO. run lemmatizer (e.g., WordNetLemmatizer)
        lemmatizer =WordNetLemmatizer()
        if query:
            lst_text = [lst_text]
        # TODO. read documents one by one and process
        for l in tqdm(lst_text):
            enc_doc = []
            tokens = tokenizer.tokenize(l.lower().strip())
            # lemmitize
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
            for t in tokens:
                if not query:
                    self.idx2tok[self.tok2idx[t]] = t
                enc_doc.append(self.tok2idx[t])
            if not query:
                self.docs.append(enc_doc)
            else:
                return enc_doc

    def create_postings_lists(self):
        avgdl = 0
        #TODO. This creates posting lists of your corpus
        for di, d in enumerate(tqdm(self.docs)):
            avgdl += len(d)
            df_inc = False
            for wi in d:
                if wi in self.postings_lists:
                    if not df_inc:
                        self.postings_lists[wi][0] += 1
                        df_inc = True
                    self.postings_lists[wi][1].append(di)
                else:
                    self.postings_lists[wi] = [1, [di]]

        for k in self.postings_lists:
            self.postings_lists[k][1] = Counter(self.postings_lists[k][1])

        #TODO. While indexing compute avgdl and documnet frequencies of your vocabulary
        self.corpus_stats["avgdl"] = avgdl / len(self.docs)

        # TODO. Save it, so you don't need to do this again in the next runs.
        index = {
            'avgdl': self.corpus_stats['avgdl'],
            'tok2idx': dict(self.tok2idx),
            'idx2tok': self.idx2tok,
            'docs': self.docs,
            'raw_ds': self.raw_ds,
            'postings': self.postings_lists
        }
        # Save
        pickle.dump(index, open(self.dbfile, 'wb'))


class SearchAgent:
    k1 = 1.5  # BM25 parameter k1 for tf saturation
    b = 0.75  # BM25 parameter b for document length normalization

    def __init__(self, indexer):
        # TODO. set necessary parameters
        self.i = indexer
        self.N = len(self.i.docs)
        self.avgdl = self.i.corpus_stats['avgdl']

    def query(self, q_str):
        results = {}
        #TODO. This is take a query string a start run the clean_text process,
        enc_query = self.i.clean_text(q_str, query=True)

        #TODO. calculate BM25 Scores
        for t in enc_query:
            if t in self.i.postings_lists:
                df = self.i.postings_lists[t][0]
                for di in self.i.postings_lists[t][1]:
                    dl = len(self.i.docs[di])
                    tf = self.i.postings_lists[t][1][di]
                    s = math.log2(self.N/df)*((self.k1 +1)*tf)/(self.k1 * (1-self.b) + self.b * dl/self.avgdl)
                    if di not in results:
                        results[di] = s
                    else:
                        results[di] += s
        # TODO. Sort  the results by the scores in decsending order
        results = sorted(results.items(), key=operator.itemgetter(1))
        results.reverse()  # this line reverse the result as per assignment output required

        # TODO. Display the result
        if len(results) == 0:
            return None
        else:
            self.display_results(results)

    def display_results(self, results):
        # Decode
        # TODO, the following is an example code, you can change however you would like.
        for docid, score in results[:5]:  # print top 5 results
            print(f'\nDocID: {docid}')
            print(f'Score: {score}')
            print('Article:')
            print(self.i.raw_ds[docid][:1000])


if __name__ == "__main__":
    i = Indexer()  # instantiate an indexer
    q = SearchAgent(i)  # document retriever
    code.interact(local=dict(globals(), **locals()))  # interactive shell
