import re
import nltk


class Preprocessor:
    """Handles text preprocessing and vectorization in just one easy payment
    """
    def __init__(self, embedder):
        self.embedder = embedder

        # Load stemmer
        try:
            self.stemmer = nltk.stem.porter.PorterStemmer()
        except:
            self.stemmer = False

        # Load the lemmatizer or download from nltk wordnet
        try:
            self.lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
        except:
            print("Downloading wordnet...")
            nltk.download('wordnet')
            self.lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
        else:
            self.lemmatizer = False

        # Load the list of stopwords or download from nltk
        try:
            self.stopwords = nltk.corpus.stopwords.words('english')
        except:
            print("Downloading stopwords...")
            nltk.download('stopwords')
            self.stopwords = nltk.corpus.stopwords.words('english')
        else:
            self.stopwords = False

    def _clean_text(self, text, stem=False, lemmatize=True, remove_stopwords=True):
        # Convert to lowercase, remove punctuations and characters and then strip
        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

        # Tokenize
        words = text.split()

        # Remove word stems (-ing, -ly, ...)
        if stem and self.stemmer:
            words = [self.stemmer.stem(word) for word in words]

        # Lemmatize (convert words into root words)
        if lemmatize and self.lemmatizer:
            words = [self.lemmatizer.lemmatize(word) for word in words]

        # Remove stopwords (and, the, ...)
        if remove_stopwords and self.stopwords:
            words = [word for word in words if word not in self.stopwords]

        # Detokenize
        text = " ".join(words)
        return text

    def _vectorize(self, text, embedder):
        # Vectorizes text into set dimensions defined by the embedder
        embedding = embedder.encode([text])
        return embedding

    def text2vec(self, text):
        # Run the whole thing in one shot
        return self._vectorize(self._clean_text(text), embedder=self.embedder)
