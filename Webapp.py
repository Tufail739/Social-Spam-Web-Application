from flask import Flask,render_template,request
import pandas as pan
from sklearn.feature_extraction.text import CountVectorizer

Webapp = Flask(__name__)

@Webapp.route('/')

def home():
	return render_template('MainContents.html')
	
@Webapp.route('/SMS',methods=['POST'])
def SMS():
	return render_template('SMSmain.html')

@Webapp.route('/Email',methods=['POST'])
def Email():
	return render_template('Emailmain.html')

@Webapp.route('/YouTube',methods=['POST'])
def YouTube():
	return render_template('YTmain.html')

@Webapp.route('/Twitter',methods=['POST'])
def Twitter():
	return render_template('Twittermain.html')


@Webapp.route('/smstest',methods=['POST'])
def smstest():
    X= pan.read_csv("Smsspam.csv", encoding="latin-1")
    X.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

    #Label Encoding
    X['Category'] = X['class'].map({'ham': 0, 'spam': 1})
    A = X['message']
    B = X['Category']

    #Feature Extraction using CountVectorizer
    cvr = CountVectorizer()
    A = cvr.fit_transform(A)
    from sklearn.model_selection import train_test_split
    TrainA, TestA, TrainB, TestB = train_test_split(A, B, test_size=0.33, random_state=42)

    #Deploying the Naive bayes model algorithm
    from sklearn.naive_bayes import MultinomialNB
    classifier = MultinomialNB()
    classifier.fit(TrainA,TrainB)
    classifier.score(TestA,TestB)


    if request.method == 'POST':
       message = request.form['message']
       PredData = [message]
       vector = cvr.transform(PredData).toarray()
       ExamineSMS = classifier.predict(vector)
    return render_template('SMSoutcome.html', prediction=ExamineSMS)



@Webapp.route('/emailtest',methods=['POST'])
def emailtest():
    E= pan.read_csv("e-maildata.csv", encoding="latin-1")

    #Assigning the columns to variables
    A = E['text']
    B = E['spam']

    #Feature Extraction using CountVectorizer
    cvr = CountVectorizer()
    A = cvr.fit_transform(A)
    from sklearn.model_selection import train_test_split
    TrainA, TestA, TrainB, TestB = train_test_split(A, B, test_size=0.33, random_state=42)

    #Deploying the SGD model algorithm
    from sklearn.linear_model import SGDClassifier
    classifier = SGDClassifier(max_iter=100)
    classifier.fit(TrainA,TrainB)
    classifier.score(TestA,TestB)


    if request.method == 'POST':
       message = request.form['message']
       PredData = [message]
       vector = cvr.transform(PredData).toarray()
       ExamineEmail = classifier.predict(vector)
    return render_template('Emailoutcome.html', prediction=ExamineEmail)

@Webapp.route('/YTcommentstest',methods=['POST'])
def YTcommentstest():
    Y= pan.read_csv("YT.csv")
    #Assigning the columns to variables
    A = Y['CONTENT']
    B = Y['CLASS']

    #Feature Extraction using CountVectorizer
    cvr = CountVectorizer()
    A = cvr.fit_transform(A)
    from sklearn.model_selection import train_test_split
    TrainA, TestA, TrainB, TestB = train_test_split(A, B, test_size=0.33, random_state=42)

    # Deploying the Logistic regression model algorithm
    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression()
    classifier.fit(TrainA,TrainB)
    classifier.score(TestA,TestB)


    if request.method == 'POST':
       message = request.form['message']
       PredData = [message]
       vector = cvr.transform(PredData).toarray()
       ExamineYT = classifier.predict(vector)
    return render_template('YToutcome.html', prediction=ExamineYT)


@Webapp.route('/Twittertest',methods=['POST'])
def Twittertest():
    T= pan.read_csv("Twitterdata.csv")
    T.drop(['Id', 'following', 'followers', 'actions', 'is_retweet'], axis=1, inplace=True)


    #Assigning the columns to variables
    T['Class'] = T['Type'].map({'Quality': 0, 'Spam': 1})
    A = T['Tweet']
    B = T['Class']

    #Feature Extraction using CountVectorizer
    cvr = CountVectorizer()
    A = cvr.fit_transform(A)
    from sklearn.model_selection import train_test_split
    TrainA, TestA, TrainB, TestB = train_test_split(A, B, test_size=0.33, random_state=42)

    # Deploying the Random Forest model algorithm
    from sklearn.ensemble import RandomForestClassifier

    classifier = RandomForestClassifier()
    classifier.fit(TrainA,TrainB)
    classifier.score(TestA,TestB)


    if request.method == 'POST':
       message = request.form['message']
       PredData = [message]
       vector = cvr.transform(PredData).toarray()
       ExamineTweet = classifier.predict(vector)
    return render_template('Twitteroutcome.html', prediction=ExamineTweet)



if __name__ == '__main__':
   Webapp.run(debug=True)