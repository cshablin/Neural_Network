import spacy
from gensim.models import KeyedVectors, word2vec
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Dense, Activation


class LyricsGenerator(object):

    def __init__(self):
        # TODO: first time run in terminal python -m spacy download en_core_web_sm
        self.nlp = spacy.load('en_core_web_sm')
        model = Sequential()
        # model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size,
        #                     weights=[pretrained_weights]))
        # model.add(LSTM(units=emdedding_size))
        # model.add(Dense(units=vocab_size))
        # model.add(Activation('softmax'))
        # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')


    def fit(self, x, y, hyper_parameters):
        pass

    def embedding_word2vec(self, word):
        # Load vectors directly from the file
        model = KeyedVectors.load_word2vec_format('data/GoogleGoogleNews-vectors-negative300.bin', binary=True)
        model = KeyedVectors.load_word2vec_format('C:\\Users\\cshablin\\Downloads\\GoogleNews-vectors-negative300.bin.gz', binary=True)
        # Access vectors for specific words with a keyed lookup:
        vector = model[word]
        # see the shape of the vector (300,)
        print(vector.shape)
        return vector
        # Processing sentences is not as simple as with Spacy:
        # vectors = [model[x] for x in "This is some text I am processing with Spacy".split(' ')]

    def embedding_spacy(self, word):

        # process a sentence using the model
        doc = self.nlp("This is some text that I am processing with Spacy")
        # It's that simple - all of the vectors and words are assigned after this point
        # Get the vector for 'text':
        print(doc[3].vector)
        # Get the mean vector for the entire sentence (useful for sentence classification etc.)
        print(doc.vector)