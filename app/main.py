import os
import sys
import traceback

import bz2
import json
import pandas as pd
import numpy as np

import nltk
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, render_template

from app.preprocessor import Preprocessor


app = Flask(__name__)
corpus = pd.DataFrame
corpus_embeddings = np.array
embedder = ""
pp = ""
knn = ""


@app.before_first_request
def setup():
	global corpus, corpus_embeddings, embedder, pp, knn

	# Load corpus contents to dataframe
	print("Loading corpus from file...")
	corpus_file = './app/res/RC_2008-04.bz2'
	contents = []
	with bz2.open(corpus_file, mode='rt', encoding='utf-8') as f:
		for row in f.readlines():
			contents.append(json.loads(row))
	corpus = pd.DataFrame(contents, columns=['body'])

	# Load embedder from model file, else, try to download it
	print("Loading embedder from model...")
	model = "multi-qa-MiniLM-L6-cos-v1"
	try:
		embedder = SentenceTransformer(f'./app/res/{model}')
	except:
		embedder = SentenceTransformer(f"{model}")

	# Load pre-encoded corpus vector embeddings
	print("Loading corpus vector embeddings...")
	with np.load('./app/res/cve.npz') as vec_file:
		corpus_embeddings = vec_file['arr_0']

	# Initialize Preprocessor
	print("Initializing preprocessor...")
	pp = Preprocessor(embedder)

	# Initialize nearest neighbors model
	print("Initializing knn model...")
	knn = NearestNeighbors(n_neighbors=5, metric='cosine')
	knn.fit(corpus_embeddings)

	print("Resources loaded successfully. Loading entrypoint.")


@app.route("/")
def index():
	return render_template('index.html')


@app.route("/search")
def search():
	return render_template('search.html')


@app.route("/results", methods=['POST', 'GET'])
def results():
	global knn, corpus
	try:
		if request.method == 'POST':
			query = request.form['query']
			query_vec = pp.text2vec(query)

			distance, neighbors = knn.kneighbors(query_vec, return_distance=True)

			results = {}
			for n, d in zip(neighbors[0], distance[0]):
				print(f"n = {n}, d = {d}")
				results[f"{corpus.iloc[n]['body'].strip()}"] = f" (score: {1-d:.3f})"
			print(f"results = {results}")
			return render_template('results.html', results=results, query=query)
	except Exception:
		print("error")
		traceback.print_exc()
		return 'Something went wrong :('


@app.route("/add-doc")
def add_doc():
	return render_template('add-doc.html')


@app.route("/update-corpus", methods=['POST', 'GET'])
def update_corpus():
	global corpus, corpus_embeddings, knn
	try:
		if request.method == 'POST':
			doc = request.form['doc']
			corpus.loc[corpus.shape[0]] = [doc]
			doc_vec = pp.text2vec(doc)
			corpus_embeddings = np.concatenate((corpus_embeddings, doc_vec))
			knn.fit(corpus_embeddings)
			return render_template('update-corpus.html', doc=doc)
	except Exception:
		print("error")
		traceback.print_exc()
		return 'Something went wrong :('


if __name__ == "__main__":
	app.run(host='0.0.0.0', debug=True)
	# app.run(host='0.0.0.0', debug=True, port=80)
