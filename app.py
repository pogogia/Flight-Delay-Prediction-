import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle

def scale(data):
    return((data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0)))

def airline(data):
    if (data == 'American Airlines Inc.' or data == 'AA' ):
        return 0
    elif (data == 'Alaska Airlines Inc.' or data == 'AS'):
        return 1
    elif (data == 'JetBlue Airways' or data == 'B6'):
        return 2
    elif (data == 'Delta Air Lines Inc.' or data == 'DL'):
        return 3
    elif (data == 'Atlantic Southeast Airlines' or data == 'EV'):
        return 4
    elif (data == 'Frontier Airlines Inc.' or data == 'F9'):
        return 5
    elif (data == 'Hawaiian Airlines Inc.'or data == 'HA'):
        return 6
    elif (data == 'American Eagle Airlines Inc.'or data == 'MQ'):
        return 7
    elif (data == 'Spirit Air Lines'or data == 'NK'):
        return 8
    elif (data == 'Skywest Airlines Inc.'or data == 'OO'):
        return 9
    elif (data == 'United Air Lines Inc.' or data == 'UA'):
        return 10
    elif (data == 'US Airways Inc.' or data == 'US'):
        return 11
    elif (data == 'American Eagle Airlines Inc.' or data == 'VX'):
        return 12
    elif (data == 'Southwest Airlines Co.' or data == 'WN'):
        return 13

def result(data):
    if(data == 0):
        return 'There is no Departure Delay'
    elif(data == 1):
        return 'Delay should be in range of 1 to 5 mins'
    elif (data == 2):
        return 'Delay should be in range of 6 to 10 mins'
    elif (data == 3):
        return 'Delay should be in range of 11 to 20 mins'
    elif (data == 4):
        return 'Delay should be in range of 21 to 50 mins'
    elif (data == 5):
        return 'Delay should be in range of 51 to 100 mins'
    elif (data == 6):
        return 'Delay should be in range of 101 to 200 mins'
    elif (data == 7):
        return 'Delay should be in range of 201 to 500 mins'
    elif (data == 8):
        return 'Delay should be in range of 501 to 1000 mins'
    else:
        return 'Delay > 1000 mins'


def split_time(feat):
    x1 =[]
    for i in feat:
        if ':' in i:
            first, second = i.split(':', 1)
            x1.append(first)
            x1.append(second)
        else:
            j = int(i)
            x1.append(j)
    final_feat = [int(x) for x in x1]
    return final_feat

app = Flask(__name__) #Initialize the flask App


model = pickle.load(open('adaboost_model.pkl', 'rb'))
scaler = MinMaxScaler()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/test.html', methods = ['GET', 'POST'])
def single():
    return render_template('test.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('File Not Uploaded')
            return render_template('BatchPredict.html')
        f = request.files['file']
        if f.filename == '':
            print('No File')
            return render_template('BatchPredict.html')
        else:
            filename = secure_filename(f.filename)
            f.save(filename)
            print('File uploaded successfully')
        input_df = pd.read_csv(f.filename)
        result_df = input_df.copy()
        print(input_df['AIRLINE'])
        print(result_df['AIRLINE'])
        input_df['AIRLINE'] = input_df['AIRLINE'].apply(airline)
        input_df[['WHEELS_ON_HOUR', 'WHEELS_ON_MIN']] = input_df['WHEELS_ON'].str.split(":", expand=True)
        input_df[['WHEELS_OFF_HOUR', 'WHEELS_OFF_MIN']] = input_df['WHEELS_OFF'].str.split(":", expand=True)
        input_df[['SCHEDULED_DEPARTURE_HOUR', 'SCHEDULED_DEPARTURE_MIN']] = input_df['SCHEDULED_DEPARTURE'].str.split(":",expand=True)
        input_df[['SCHEDULED_ARRIVAL_HOUR', 'SCHEDULED_ARRIVAL_MIN']] = input_df['SCHEDULED_ARRIVAL'].str.split(":", expand=True)
        input_df[['DEPARTURE_TIME_HOUR', 'DEPARTURE_TIME_MIN']] = input_df['DEPARTURE_TIME'].str.split(":", expand=True)
        input_df[['ARRIVAL_TIME_HOUR', 'ARRIVAL_TIME_MIN']] = input_df['ARRIVAL_TIME'].str.split(":", expand=True)
        input_df.drop(['WHEELS_ON', 'WHEELS_OFF', 'SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL', 'DEPARTURE_TIME', 'ARRIVAL_TIME'], axis=1, inplace=True)
        input_df = input_df.astype('int64')
        scaler.fit(input_df)
        input_scale_feat = scaler.transform(input_df)
        input_predict = model.predict(input_scale_feat)
        result_df['Result'] = input_predict
        result_df['Result_Description'] = result_df['Result'].apply(result)
        print(result_df)
        print(result_df['AIRLINE'])
        result_df.to_csv('Prediction.csv')
        return redirect('/downloadfile/' + 'Prediction.csv')
    return render_template('BatchPredict.html')

@app.route('/downloadfile/<filename>', methods =['GET'])
def download_file(filename):
    return render_template('Download.html', value = filename)

@app.route('/return-files/<filename>')
def return_files_tut(filename):
    file_path = filename
    return send_file(file_path, as_attachment=True, attachment_filename='')

@app.route('/BatchPredict.html', methods = ['GET','POST'])
def batch():
    return render_template('BatchPredict.html')

#@app.route('/upload')
#def upload_file():
#   return render_template('BatchPredict.html')



@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    print(int_features)
    final_features = split_time(int_features)
    feat_df = pd.DataFrame(final_features, columns=['data'])
    print(feat_df)
    print(feat_df.info())
    scale_feat = feat_df.apply(scale)
    scale_list = scale_feat['data'].tolist()
    final_features = [np.array(scale_list)]
    print(final_features)
    prediction = model.predict(final_features)
    print(prediction)
    output = prediction[0]
    print(output)
    if output == 0:
        return render_template('test.html', prediction_text='There is no Departure Delay')
    elif output == 1:
        return render_template('test.html', prediction_text='Delay should be in range of 1 to 5 mins')
    elif output == 2:
        return render_template('test.html', prediction_text='Delay should be in range of 6 to 10 mins')
    elif output == 3:
        return render_template('test.html', prediction_text='Delay should be in range of 11 to 20 mins')
    elif output == 4:
        return render_template('test.html', prediction_text='Delay should be in range of 21 to 50 mins')
    elif output == 5:
        return render_template('test.html', prediction_text='Delay should be in range of 51 to 100 mins')
    elif output == 6:
        return render_template('test.html', prediction_text='Delay should be in range of 101 to 200 mins')
    elif output == 7:
        return render_template('test.html', prediction_text='Delay should be in range of 201 to 500 mins')
    elif output == 8:
        return render_template('test.html', prediction_text='Delay should be in range of 501 to 1000 mins')
    else:
        return render_template('test.html', prediction_text='Delay should be >1000 mins')

if __name__ == "__main__":
    app.run(debug=True)