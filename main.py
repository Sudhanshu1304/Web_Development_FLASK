from flask import Flask,request,render_template
import ML1 as ml


app = Flask(__name__)

@app.route('/')

def home():
    return render_template('tem2.html')


@app.route('/get',methods=['POST','GET'])

def get_input():

    if request.method=='POST':
        text=request.form['fname1']
        re = ml.output1(text)
        plturl=ml.history_plot()

        return render_template('tem2.html',output=re,img=plturl)



if __name__ == '__main__':
   app.run(debug = True)