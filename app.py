from cProfile import label
import numpy as np
from flask import Flask,request,render_template
import pickle
model = pickle.load(open('model.pkl','rb'))
process = pickle.load(open('process.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])

def predict():
    label = {0:'not greater than 50k',1:'Greater than 50K'}
    d = {'Husband': 131.93,
        'Not-in-family': 83.04,
        'Own-child': 50.68,
        'Unmarried': 34.46,
        'Wife': 15.68,
        'Other-relative': 9.81}

    e = {'Doctorate':16,
'Masters':15,
'Assoc-acdm':14,
'Assoc-voc':13,
'Bachelors':12,
'Some-college':11,
'Prof-school':10,
'12th':9,
'11th':8,
'HS-grad':7,
'10th':6,
'9th':5,
'7th-8th':4,
'5th-6th':3,
'1st-4th':2,
'Preschool':1}
    if request.method == 'POST':
        age = int(request.form['age'])
        education = request.form['education']
        relationship= request.form['relationship']
        capital_gain = int(request.form['capital_gain'])
        hours_per_week = int(request.form['hours_per_week'])
        l = [age,e[education],d[relationship],capital_gain,hours_per_week]
        l1 = np.array(l)
        s = l1.reshape(1,-1)
        p = process.transform(s)
        model_eval = model.predict(p)[0]
        return  render_template('index.html',predection_eval = label[model_eval])

if __name__ == '__main__':
    app.run(debug=True)