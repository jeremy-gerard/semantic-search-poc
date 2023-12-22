# Semantic Search

A simple proof-of-concept for keyword search across a corpus of ~500k reddit docs.

To start, simply run  `docker compose up`

Once the env and resources have loaded and the flask app boots up, you can access the application in a browser @ <http://localhost:5000>

If you've reached the homepage, navigation from there should be self-evident.

- **Note**: This application is a bit bulky so if you're running it locally you may need to go into the Docker resource settings and up the memory or at least the swap limit if you're getting any errors trying to build.

- **Prospective troubleshooting on build**: I've noticed that sometimes after the container has been built and the wsgi app has compiled, the intialization may get hung up before the app starts running, which I have remedied by making a request to the homepage. This will likely throw a timeout error at first but should return a 200 once the `before_first_request` function has finished loading all the resources.

## Solution and Framework

To avoid the overhead of having to run a whole pytorch/cuda process on a GPU, I went ahead and pre-encoded the corpus vector embeddings which are stored in `app/res/cve.npz` as a compressed numpy ndarray.

For the encoding, I used `multi-qa-MiniLM-L6-cos-v1`, a pre-trained model based on BERT architecture and optimized for assymetric semantic search on sentence to paragraph length text.

For finding nearest neighbors, I used `sklearn`'s optimizer with cosine similarity as the distance metric.

The text cleaning and parsing uses the `nltk` language toolkit.
