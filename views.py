import flask
import pickle5 as pickle

# NLTK
from nltk.tokenize import RegexpTokenizer

# Stop-words
from nltk.corpus import stopwords

# Lemmatisation
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

#==================

# Charger les objets pickel:

# Encodage de y
f = open('model/data_multilabelbin_coder.ser', 'rb')
mlabelbin_coder = pickle.load(f)
f.close()        
    
# Classifieur Reg Log
f = open('model/data_ovrLOGREG.ser', 'rb')
ovrLOGREG = pickle.load(f)
f.close()

# Load vectorizer
f = open('model/data_vect.ser', 'rb')
vectorizer = pickle.load(f)
f.close()

def sentence_to_tokens(sentence):
    tokenizer = RegexpTokenizer('( r | c\+\+ | c# | c | Objective-c | \.net|[a-z]{2,})')
    return tokenizer.tokenize(sentence)


def filter_stopwords(tokens, stopwords, flag):
    isstopword = []
    notstopword = []
    for w in tokens :
        if w in stopwords :
            isstopword.append(w)
        else:
            notstopword.append(w)       
    
    if (flag == "nostopword"):
        res = notstopword        
    else :
        res = isstopword
    return res

def word_lemmatizer(words):
    # Spécifier la nature des mots avec post-tag
    pt_words = pos_tag(words)
    
    # lemmatisation
    lemmatizer = WordNetLemmatizer()
    lemmes = []
    for word, tag in pt_words:
        wntag = tag[0].lower()
        wntag = wntag if wntag in ['n'] else None
        if wntag:
            lemma = lemmatizer.lemmatize(word, wntag)
            lemmes.append(lemma)   
    
    return lemmes

def text_preprocessing(text, vect):
    text1 = text.lower()
    text2 = text1.replace('c++',' cplusplus ')
    token1 = sentence_to_tokens(text2)
    token2 = filter_stopwords(token1, stopwords.words('english'), "nostopword")    
    token3 = word_lemmatizer(token2)
    token_str = " ".join(token3)
    token_cod = vect.transform([token_str]) # vectorization du texte  
    return token3, token_cod

def tags_prediction(clf, X, coder):
    tags_cod = clf.predict(X)
    #print("X shape : ", X.shape)
    tags = coder.inverse_transform(tags_cod)
    resp = " ".join(tags[0]).replace(" ",", ")
    return resp

#====================

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('main.html')
    if flask.request.method == 'POST':
        question = flask.request.form['question']
        text_token, text_cod = text_preprocessing(question, vectorizer)
        prediction = tags_prediction(ovrLOGREG, text_cod, mlabelbin_coder)
        return flask.render_template('main.html', original_input={'Your question':question}, result=prediction)

#====================

if __name__ == '__main__':
    app.run(debug=True)
