from flask import Flask,render_template,redirect,url_for,flash,request
from summary import generate_summary

app = Flask(__name__)

app.secret_key = "HBCJSBJH454546SSCHJSBHCJBKSBKNCJAASnadn"

@app.route("/",methods=["POST","GET"])
def index():
    return render_template('index.html')

@app.route("/getchaptersummary",methods= ["POST","GET"])
def getsummary():
    if request.method == "POST":
        text = request.form["text"]
        data = generate_summary(text)
        return render_template("getchapsum.html",data= data,text=text)
    return render_template("getchapsum.html")
if __name__ == "__main__":
    app.run(debug=True)