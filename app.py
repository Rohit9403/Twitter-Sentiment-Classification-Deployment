from flask import Flask,render_template,url_for,request


import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.models import Sequential
from keras.models import load_model
maxlen = 358
def preprocess_text(sen):
  #removing HTML Tags
  #sentence= remove_tags(sen)

  #Remove punctuations and numbers
  sentence= re.sub('[^a-zA-z]',' ', str(sen))

  #Remove Single character
  sentence=re.sub(r"\s+[a-zA-z]\s+",' ', sentence)

  #remove multiple spaces
  sentence=re.sub(r'\s+',' ',sentence)

  return sentence




# load the model from disk
#filename = 'nlp_model.pkl'
#model= pickle.load(open(filename, 'rb'))
model=load_model('Nlp_Model.h5')
#cv=pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
  if request.method=='POST':
    message=request.form['message']
    data=[message]
    dat1=preprocess_text(data)
    tokenizer = Tokenizer(num_words=5000)
    dat1=tokenizer.texts_to_sequences(dat1)
    dat1=pad_sequences(dat1, padding='post', maxlen=358)
    my_prediction=model.predict(dat1)
  return render_template('return.html',prediction=my_prediction[0])


if __name__ == '__main__':
	app.run(debug=True)
