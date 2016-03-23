'''
Simple web service wrapping a Word2Vec as implemented in Gensim
Example call: curl http://127.0.0.1:5000/wor2vec/n_similarity/ws1=Sushi&ws1=Shop&ws2=Japanese&ws2=Restaurant
@TODO: Add more methods
@TODO: Add command line parameter: path to the trained model
@TODO: Add command line parameters: host and port
'''

from flask import Flask, request, jsonify
from flask.ext.restful import Resource, Api, reqparse
from gensim.models.word2vec import Word2Vec as w
from gensim import utils, matutils
from numpy import exp, dot, zeros, outer, random, dtype, get_include, float32 as REAL,\
     uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis, ndarray, empty, sum as np_sum
import cPickle
import argparse
import base64
import sys
import pprint
import logging
from logging.handlers import RotatingFileHandler

parser = reqparse.RequestParser()


def filter_words(words):
    if words is None:
        return
    return [word for word in words if word in model.vocab]


class N_Similarity(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('ws1', type=str, required=True, help="Word set 1 cannot be blank!", action='append')
        parser.add_argument('ws2', type=str, required=True, help="Word set 2 cannot be blank!", action='append')
        args = parser.parse_args()
        return model.n_similarity(filter_words(args['ws1']),filter_words(args['ws2']))


class Similarity(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('w1', type=str, required=True, help="Word 1 cannot be blank!")
        parser.add_argument('w2', type=str, required=True, help="Word 2 cannot be blank!")
        args = parser.parse_args()
        return model.similarity(args['w1'], args['w2'])


class Similarity_NxN(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('word', type=str, required=True, action='append', help="Word 0 cannot be blank!")
        args = parser.parse_args()
        words = filter_words(args.get('word', []))
        data=[]
        str_list=[]
        str_list.append("{'results':[")
        c=0
        if(len(words)<2): 
            return "Need more than a single word."
        else:
            for i in range((len(words)-1)):
                  for j in range((i+1),len(words)):
                       if(c>0):
                            str_list.append( ",{'w1':'"+words[i]+"','w2':'"+words[j]+"','sim':'"+str(model.similarity(words[i],words[j]))+"'}"  )
                            str_list.append( "{'w1':'"+words[i]+"','w2':'"+words[j]+"','sim':'"+str(model.similarity(words[i],words[j]))+"'}"  )
                       c=c+1

        str_list.append("]}")
        result="".join(str_list)

        return result

class Similarity_NxM(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('w1', type=str, required=True, action='append', help="Word 1 cannot be blank!")
        parser.add_argument('w2', type=str, required=True, action='append', help="Word 2 cannot be blank!")
        args = parser.parse_args()
        words1 = filter_words(args.get('w1', []))
        words2 = filter_words(args.get('w2', []))
        data=[]
        str_list=[]
        str_list.append("{'results':[")
        c=0
        for i in range((len(words1))):
             for j in range(len(words2)):
                if(c>0):
                  str_list.append(",{'w1':'"+words1[i]+"','w2':'"+words2[j]+"','sim':'"+str(model.similarity(words1[i],words2[j]))+"'}")
                else:
                  str_list.append("{'w1':'"+words1[i]+"','w2':'"+words2[j]+"','sim':'"+str(model.similarity(words1[i],words2[j]))+"'}")
                c=c+1

        str_list.append("]}")
        result="".join(str_list)

        return result



class MostSimilar(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('positive', type=str, required=False, help="Positive words.", action='append')
        parser.add_argument('negative', type=str, required=False, help="Negative words.", action='append')
        parser.add_argument('topn', type=int, required=False, help="Number of results.")
        args = parser.parse_args()
        pos = filter_words(args.get('positive', []))
        neg = filter_words(args.get('negative', []))
        t = args.get('topn', 10)
        pos = [] if pos == None else pos
        neg = [] if neg == None else neg
        t = 10 if t == None else t
        print "positive: " + str(pos) + " negative: " + str(neg) + " topn: " + str(t)
        try:
            res = model.most_similar_cosmul(positive=pos,negative=neg,topn=t)
            return res
        except Exception, e:
            print e
            print res
            return

class MostSimilar2(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('positive', type=str, required=False, help="Positive words.", action='append')
        parser.add_argument('negative', type=str, required=False, help="Negative words.", action='append')
	#parser.add_argument('class', type=str, required=False, help="CPC class", action='append')
	#parser.add_argument('section', type=str, required=False, help="CPC class section", action='append')
        parser.add_argument('topn', type=int, required=False, help="Number of results.")
        args = parser.parse_args()
        pos = filter_words(args.get('positive', []))
        neg = filter_words(args.get('negative', []))
        t = args.get('topn', 10)
        pos = [] if pos == None else pos
        neg = [] if neg == None else neg
        t = 10 if t == None else t
        print "positive: " + str(pos) + " negative: " + str(neg) + " topn: " + str(t)
        try:
            res = model.most_similar(positive=pos,negative=neg,topn=t)
            return res
        except Exception, e:
            print e
            print res
            return


class Dataset(Resource):
    def get(self):
        try:
            splitted = model_path.split("/")
            return splitted[len(splitted)-1].split("_")[0]
        except Exception, e:
            print e
            return


class Model(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('word', type=str, required=True, help="word to query.")
        args = parser.parse_args()
        try:
            res = model[args['word']]
            res = base64.b64encode(bytes(res))
            return res
        except Exception, e:
            print e
            return

class ModelWordSet(Resource):
    def get(self):
        try:
            res = base64.b64encode(cPickle.dumps(set(model.index2word)))
            return res
        except Exception, e:
            print e
            return

class LoggingMiddleware(object):
    def __init__(self, app):
        self._app = app

    def __call__(self, environ, resp):
        errorlog = environ['wsgi.errors']
        pprint.pprint(('REQUEST', environ), stream=errorlog)

        def log_response(status, headers, *args):
            pprint.pprint(('RESPONSE', status, headers), stream=errorlog)
            return resp(status, headers, *args)

        return self._app(environ, log_response)


app = Flask(__name__)
api = Api(app)

@app.errorhandler(404)
def pageNotFound(error):
    return "page not found"

@app.errorhandler(500)
def raiseError(error):
    return error

if __name__ == '__main__':
    global model

    #----------- Parsing Arguments ---------------
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="Path to the trained model")
    p.add_argument("--binary", help="Specifies the loaded model is binary")
    p.add_argument("--host", help="Host name (default: localhost)")
    p.add_argument("--port", help="Port (default: 5000)")
    p.add_argument("--path", help="Path (default: /word2vec)")
    args = p.parse_args()

    model_path = args.model if args.model else "./model.bin.gz"
    binary = True if args.binary else False
    host = args.host if args.host else "localhost"
    path = args.path if args.path else "/word2vec"
    port = int(args.port) if args.port else 5000
    if not args.model:
        print "Usage: word2vec-apy.py --model path/to/the/model [--host host --port 1234]"
    model = w.load_word2vec_format(model_path, binary=binary)
    api.add_resource(N_Similarity, path+'/n_similarity')
    api.add_resource(Similarity, path+'/similarity')
    api.add_resource(Similarity_NxN, path+'/similarity_NxN')
    api.add_resource(Similarity_NxM, path+'/similarity_NxM')
    api.add_resource(MostSimilar, path+'/most_similar')
    api.add_resource(MostSimilar2, path+'/most_similar2')
    api.add_resource(Dataset, path+'/dataset')
    api.add_resource(Model, path+'/model')
    api.add_resource(ModelWordSet, '/word2vec/model_word_set')

    handler = RotatingFileHandler("/tmp/flask_"+str(port)+".log", maxBytes=10000, backupCount=5)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.DEBUG)
    log.addHandler(handler)
    
   # app.wsgi_app = LoggingMiddleware(app.wsgi_app)
    app.run(host=host, port=port)

