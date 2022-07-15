# Based on https://github.com/SubhamIO/Build-and-Deploy-an-Machine-Learning-Model-using-AWS-and-API-s

#####################################################################################################

from flask import Flask, jsonify, request
import joblib
import re
from sklearn.feature_extraction.text import CountVectorizer


# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)




signos = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\¿)|(\@)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")

def signs_tweets(tweet):
    return signos.sub('', tweet.lower())

def del_emojis(tweet):
    return tweet.str.replace(r'[^\x00-\x7F]+', '', regex=True)

def remove_links(tweet):
    return " ".join(['{link}' if ('http') in word else word for word in df.split()])


def clean_text(sentence):
    #sentence = signs_tweets(sentence)
    #sentence = del_emojis(sentence)
    #sentence = remove_links(sentence)
    return  sentence


###################################################


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    clf = joblib.load('finished_model.pkl')
    count_vect = joblib.load('count_vect.pkl')
    to_predict_list = request.form.to_dict()
    review_text = clean_text(to_predict_list['review_text'])
    #pred = clf.predict(count_vect.transform([review_text]))
    pred = clf.predict([review_text])
    if pred[0]:
        prediction = "Positive"
    else:
        prediction = "Negative"

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
