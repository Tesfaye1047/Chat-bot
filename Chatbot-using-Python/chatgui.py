import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
import telegram.ext
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
TOKEN="6057636260:AAFiLJJ-ekeiuD45cLKYU88p_CeE5YMEAd0"

print("done1")


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))
print("done2")

def predict_class(sentence):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
print("done3")

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result
print("done4")

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

print("done5")


def start(update,context):
    update.message.relpy_text(
    """
    /start->welcome to JIT AI center how can i help you
    """
    )
print("done6")


def handel_message(update,context):
    message=update.message.text
    ints=predict_class(message)
    res= getResponse(ints,intents)
    update.message.reply_text(res)

print("done7")


updater= telegram.ext.Updater(TOKEN,use_context=True)
disp=updater.dispatcher
disp.add_handler(telegram.ext.MessageHandler(telegram.ext.Filters.text, handel_message))
disp.add_handler(telegram.ext.CommandHandler('start', start))

updater.start_polling( 
    # timeout = 100
  
  )
updater.idle()



print("done8")



