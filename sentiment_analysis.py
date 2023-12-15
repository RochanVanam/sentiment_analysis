import numpy as np
import math
import os
import re
import pandas as pd
from bs4 import BeautifulSoup
import tensorflow as tf
from keras import layers, preprocessing, Model
from keras.models import save_model, load_model
import tensorflow_datasets as tfds

path_to_model = 'models/saved_model'
path_to_tokenizer = 'tokenizer'

# Deep convolutional neural network
class DCNN(Model):

    def __init__(
            self,
            vocab_size,
            emb_dim=128, # Number of dimensions for embedding
            nb_filters=50, # Number of filters for each filter type
            FFN_units=512,
            nb_classes=2, # Number of classifications at the end
            dropout_rate=0.1,
            training=False,
            name="dcnn"):
        
        super(DCNN, self).__init__(name=name)

        # Embedding layer
        self.embedding = layers.Embedding(vocab_size, emb_dim)
        
        # Filter of width 2
        self.bigram = layers.Conv1D(
            filters=nb_filters,
            kernel_size=2,
            padding="valid",
            activation="relu")
        self.pool_1 = layers.GlobalMaxPool1D()

        # Filter of width 3
        self.trigram = layers.Conv1D(
            filters=nb_filters,
            kernel_size=3,
            padding="valid",
            activation="relu")
        self.pool_2 = layers.GlobalMaxPool1D()

        # Filter of width 4
        self.fourgram = layers.Conv1D(
            filters=nb_filters,
            kernel_size=4,
            padding="valid",
            activation="relu")
        self.pool_3 = layers.GlobalMaxPool1D()

        self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate) # Dropouts avoid overfitting

        if nb_classes == 2:
            self.last_dense = layers.Dense(units=1, activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=nb_classes, activation="softmax")
    
    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x_1 = self.bigram(x)
        x_1 = self.pool_1(x_1)
        x_2 = self.trigram(x)
        x_2 = self.pool_2(x_2)
        x_3 = self.fourgram(x)
        x_3 = self.pool_3(x_3)

        merged = tf.concat([x_1, x_2, x_3], axis=-1) # Shape of merged: (batch_size, 3 * nb_filters)
        merged = self.dense_1(merged)
        merged = self.dropout(merged, training=training)
        output = self.last_dense(merged)

        return output
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 73], dtype=tf.int32, name='input_1')])
    def serve(self, inputs):
        return self.call(inputs, training=False)

############################

dcnn = None
if (not os.path.exists(path_to_model)):
    # Load files
    cols = ["sentiment", "id", "date", "query", "user", "text"]
    train_data = pd.read_csv(
        "data/train.csv",
        header=None,
        names=cols,
        engine="python",
        encoding="latin1"
    )

    data = train_data

    # Cleaning
    data.drop(["id", "date", "query", "user"],
            axis=1,
            inplace=True)

    def clean_tweet(tweet):
        tweet = BeautifulSoup(tweet, "lxml").get_text()
        tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
        tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet)
        tweet = re.sub(r"[^a-zA-Z.!?']", ' ', tweet)
        tweet = re.sub(r" +", ' ', tweet)
        return tweet

    data_clean = [clean_tweet(tweet) for tweet in data["text"]]
    data_labels = data["sentiment"].values
    data_labels[data_labels == 4] = 1
    set(data_labels)

    # Tokenizer
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        data_clean,
        target_vocab_size=2**16)
    data_inputs = [tokenizer.encode(sentence) for sentence in data_clean]

    # Pad sentences (make each sentence same length by adding 0 padding)
    MAX_LEN = max([len(sentence) for sentence in data_inputs])
    data_inputs = preprocessing.sequence.pad_sequences(
        data_inputs,
        value=0,
        padding="post",
        maxlen=MAX_LEN)

    # Split into training/testing datasets
    # Our data is sorted. First 8 million rows are negative sentiment, second half is positive sentiment
    test_indexes = np.random.randint(0, 800000, 8000)
    test_indexes = np.concatenate((test_indexes, test_indexes+800000))

    test_inputs = data_inputs[test_indexes]
    test_labels = data_labels[test_indexes]
    train_inputs = np.delete(data_inputs, test_indexes, axis=0)
    train_labels = np.delete(data_labels, test_indexes)

    # Config variables
    VOCAB_SIZE = tokenizer.vocab_size
    NB_CLASSES = len(set(train_labels))
    EMB_DIM = 200
    NB_FILTERS = 100
    FFN_UNITS = 256
    DROPOUT_RATE = 0.2
    BATCH_SIZE = 32
    NB_EPOCHS = 5

    dcnn = DCNN(
        vocab_size=VOCAB_SIZE,
        emb_dim=EMB_DIM,
        nb_filters=NB_FILTERS,
        FFN_units=FFN_UNITS,
        nb_classes=NB_CLASSES,
        dropout_rate=DROPOUT_RATE)

    if NB_CLASSES == 2:
        dcnn.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy"])
    else:
        dcnn.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["sparse_categorical_accuracy"])
        
    checkpoint_path = "checkpoint"
    ckpt = tf.train.Checkpoint(Dcnn=dcnn)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!")

    dcnn.fit(
        train_inputs,
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=NB_EPOCHS)
    ckpt_manager.save()

    tokenizer.save_to_file(path_to_tokenizer)
    dcnn.save(path_to_model, save_format='tf', signatures={'serving_default': dcnn.serve}) ########

dcnn = load_model(path_to_model)
loaded_tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(path_to_tokenizer) 
while True:
    user_input = input("Enter a phrase for sentiment analysis: ")
    tokenized_input = loaded_tokenizer.encode(user_input)
    padded_input = preprocessing.sequence.pad_sequences([tokenized_input], maxlen=73, dtype="int32", padding="post", truncating="post")
    input_array = np.array(padded_input)
    
    array = dcnn(input_array, training=False).numpy()
    sentiment = array[0]
    
    print(f'Sentiment: {sentiment} --- ', end="")
    if sentiment > 0.5:
        print("POSITIVE")
    elif sentiment < 0.5:
        print("NEGATIVE")
    elif sentiment == 0.5:
        print("NEUTRAL")
    print()
