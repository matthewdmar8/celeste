# importing the required modules.
import json
import pickle
import numpy as np
import nltk
from keras.models import load_model
from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
from keras.layers import Dense, Dropout, Embedding, LSTM, LeakyReLU, BatchNormalization
from keras import optimizers, regularizers
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils, pad_sequences
import h5py
from sklearn.model_selection import KFold
import tensorflow as tf
from keras.callbacks import EarlyStopping

# MODULE: training.py
# LAST UPDATED: 03/25/2023
# AUTHOR: MATTHEW MAR
# FUNCTION : Construct CELESTE' Neural Network Architecture, Train CELESTE on data set, save model, tokenizer, vocabulary, and intents to local files.

# Set up GPU options
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.888)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
# Create a TensorFlow session with GPU options
sess = tf.compat.v1.Session(config=config)
# Prompt user to enter a username and password
flag = True
# while(flag==True):
#     username = input("Enter your username: ")
#     password = input("Enter your password: ")
#     if (AIDAO.login('', username, password)):
#         print("Login Succesfull. Beginning training...")
#         flag=False

# Define WordNetLemmatizer to lemmatize words.
lemmatizer = WordNetLemmatizer()
# reading the json.intense file
intents = json.loads(open("training.json", encoding="utf8").read())
# Define Nadam optimizer. This optimizer works best for CELESTE' Long Short-Term Memory (LSTM) network.
Nadam = optimizers.Nadam(
    learning_rate=0.001,
    clipvalue=1.0,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    name="Nadam",
)


# Function to built CELESTE ( NEURAL NETWORK ARCHITECTURE )
# Using Tensorflow, Keras
# Using Embedding Vectors
# Using LSTM RNN, Leaky Dropout Regularization, Batch Normalization
# PARAMETERS: Model
# RETURNS: built model
def reinit(model):
    print("Re-initializing CELESTE...")
    ########################################################################################################
    model = Sequential()  # Keras Sequential Model
    print("NEURAL NET WORDS")
    print(len(words))
    model.add(
        Embedding(tokenizer_vocab_size, 32, input_length=len(words))
    )  # Embedding layer to process embedding vectors
    model.add(
        LSTM(25, kernel_initializer="he_uniform", recurrent_initializer="he_uniform")
    )  # LSTM Layer. Scale neurons with size of training set.
    model.add(BatchNormalization())  # Batch normalization layer
    model.add(
        Dense(
            128,
            kernel_regularizer=regularizers.l2(0.3),
            kernel_initializer="variance_scaling",
        )
    )  # First Dense at 128 neurons. Drop this layer out.
    model.add(BatchNormalization())  # Batch normalization layer
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.3))
    model.add(
        Dense(
            64,
            kernel_regularizer=regularizers.l2(0.3),
            kernel_initializer="variance_scaling",
        )
    )  # Second Dense at 64 neurons. Don't drop this layer out.
    model.add(LeakyReLU(alpha=0.01))
    model.add(
        Dense(len(classes), activation="softmax")
    )  # Last Dense with one neuron for each class to make a prediction on. Softmax activation function.
    # compile the model
    model.compile(
        loss="categorical_crossentropy", optimizer=Nadam, metrics=["accuracy"]
    )
    print("CELESTE Re-initialized.")
    print(model.summary())
    return model


# K-Fold CROSS VALIDATION =================================================================================================

scores = []  # Initialize a list to store the evaluation metrics


# Function to execute training
# Using K-Fold Cross Validation
# PARAMETERS: Model
# RETURNS: CELESTE Model with best scores
def train(model):
    # Define a callback for early stopping. Early stop will stop training if the model stops improving.
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        mode="max",
        min_delta=0.01,
        verbose=1,
        restore_best_weights=True,
    )
    num_folds = 5  # Number of folds, we'll use 5
    best_score = float("-inf")
    best_loss = float("inf")
    tars = None
    # K-Fold Cross Validator comes from Sklearn
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Loop over the folds
    for train_index, test_index in kf.split(train_x_encoded_padded_words):
        # Get the training and test data for this fold
        x_train, x_test = (
            train_x_encoded_padded_words[train_index],
            train_x_encoded_padded_words[test_index],
        )
        y_train, y_test = train_y[train_index], train_y[test_index]

        # Train CELESTE on current fold.
        hist = model.fit(
            x_train,
            y_train,
            epochs=30,
            batch_size=7,
            callbacks=early_stop,
            validation_data=(x_test, y_test),
        )

        # Evaluate the model on the test data for this fold
        score = model.evaluate(x_test, y_test, verbose=0)
        scores.append(score)
        # Update tars and best score with best score
        if float(score[0]) < best_loss:
            best_score = float(score[1])
            best_loss = float(score[0])
            tars = hist
    print("BEST SCORE", best_score)
    print("BEST LOSS", best_loss)
    return tars


# creating empty lists to store data: words, intents, documents, and ignore letters
words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ",", "'", ":", ";", "(", ")", "-", "_"]
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # separating words from patterns
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)  # and adding them to words list

        # associating patterns with respective tags
        # Here we are ignoring the word "was" specifically because NLTK's wordnet doesn't have a proper lemma for it lol
        documents.append(
            (
                [
                    word
                    if word
                    == "was"  # <-- if we discover more words like this, we can add them here
                    else lemmatizer.lemmatize(word).lower()
                    for word in word_list
                    if word not in ignore_letters
                ],
                intent["tag"],
            )
        )

        # appending the tags to the class list
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

train_x = [document[0] for document in documents]  # train_x is list of patterns
train_x = np.array(train_x, dtype=object)  # Converting train_x to a numpy array
print("LENGTH OF CLASSES")  # Debugging, print amount of classes
print(len(classes))
# saving the words and classes list to binary files
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))
# one-hot encoding output train = essentially converting to binary
train_y = np_utils.to_categorical(
    [classes.index(document[1]) for document in documents], num_classes=len(classes)
)
# PREPROCESSING
# Store train_x into text variable
text = train_x
print(text)  # Debugging - print text
tokenizer = Tokenizer(
    oov_token="<OOV>"
)  # Initialize tokenizer and set Out of Vocab token to "<OOV>"
# fit the tokenizer on our text
tokenizer.fit_on_texts(text)  # Train the tokenizer on the text
tokenizer_vocab_size = (
    len(tokenizer.word_index) + 1
)  # Get the size of the tokenizer's vocabulary
# OOV tokens placed out of index to tell NN they are OOV

# Debugging...
print("VOCAB SIZE")
print(tokenizer_vocab_size)
print("X TRAIN SHAPE")
print(train_x.shape)
print("TRAIN_Y SHAPE")
print(train_y.shape)

# Store encoded and padded encoded words train_x words.
train_x_encoded_words = tokenizer.texts_to_sequences(train_x)
train_x_encoded_padded_words = pad_sequences(train_x_encoded_words, maxlen=len(words))

# Open CELESTE memory from .h5 file
with h5py.File("tars.h5", "r") as tars:
    if (
        len(tars.keys()) != 0 and 0 == 1
    ):  # Load TARS if memory is found. Currently, this is disabled.
        print("CELESTE MEMORY FOUND. DOWNLOADING CELESTE...")
        print(tars.keys())
        model = load_model(tars)
    else:  # Otherwise create a new neural network. We're experimenting with Neural Network architecture engineering so we will create from scratch every time.
        print("CELESTE MEMORY NOT FOUND. RE-CREATING NEURAL NETWORK...")
        model = reinit(0)

# Activate training
try:
    tars = train(model)  # Train CELESTE
except (
    ValueError
):  # This will raise if the dataset is different than the one CELESTE was trained on.
    print(ValueError)
    print(
        "Input data difference detected. Re-initializing CELESTE."
    )  # CELESTE's neural network will need to be re-created if the data set is different.
    model = reinit(model)
    print("Re-fitting CELESTE.")
    tars = train(model)
# saving the model
print("Saving CELESTE model...")
model.save("tars.h5", tars)
# Upload CELESTE to database upon finished training

print("Saving tokenizer...")
with open("tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print statement to show the
# successful training of the model
# Print the final evaluation metrics
print("Final evaluation metrics:", scores)
print("CELESTE Training Successful.")
