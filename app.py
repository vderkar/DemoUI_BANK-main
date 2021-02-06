from flask import Flask, render_template,url_for, request,redirect
import pandas as pd
import pickle
import warnings
import joblib
from flask_cors import CORS, cross_origin
from predictionfolder.prediction import LA_predict,predict,Fraud_predict,LR_predict, LE_predict

def warns(*args, **kwargs):
    pass
warnings.warn = warns

ALLOWED_EXTENSIONS = set(['csv','xlsx','data'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# load the model from directory
#model = joblib.load('pickle_files/pickle_fraud.pkl')
#ss = joblib.load('pickle_files/scaler_fraud.pkl')
ss_LA = pickle.load(open('pickle_files/veena_LA_stan_scaler.pkl', 'rb'))
model_LA=pickle.load(open('pickle_files/LAOpt8model.sav', 'rb'))
model_Fraud = joblib.load('pickle_files/Fraud_new_model.pkl')
model_LR = pickle.load(open('pickle_files/loan_risk.pkl', 'rb'))
model_LE = pickle.load(open('pickle_files/XGBTunedModel-LE.pkl', 'rb'))

instance = predict()
LA_instance = LA_predict()
Fraud_instance = Fraud_predict()
LR_instance = LR_predict()
LE_instance = LE_predict()
app = Flask(__name__)

@app.route('/')
@cross_origin()
def home():
    return render_template('home.html')

@app.route('/LA',methods=['GET','POST'])
@cross_origin()
def LA():
    if request.method == 'POST':
        return render_template('neha_secondsample.html')

@app.route('/FD',methods=['GET','POST'])
@cross_origin()
def FD():
    if request.method == 'POST':
        return render_template('FD_new_home.html')

@app.route('/LR',methods=['GET','POST'])
@cross_origin()
def LR():
    if request.method == 'POST':
        return render_template('loan_risk.html')

@app.route('/LE',methods=['GET','POST'])
@cross_origin()
def LE():
    if request.method == 'POST':
        return render_template('loan_eligibility.html')

@app.route('/bulk_predict',methods=['GET','POST'])
@cross_origin()
def bulk_predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            data = Fraud_instance.predictor(file)
            return render_template('result_bulk.html', tables=[data.to_html(classes='data')], titles=data.columns.values)
        else:
            return redirect(request.url)

@app.route('/LA_bulk_predict',methods=['GET','POST'])
@cross_origin()
def LA_bulk_predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            data = LA_instance.predictor(file)
            return render_template('result_bulk.html', tables=[data.to_html(classes='data')], titles=data.columns.values)
        else:
            return redirect(request.url)

@app.route('/LE_bulk_predict',methods=['GET','POST'])
@cross_origin()
def LR_bulk_predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            data = LE_instance.predictor(file)
            return render_template('result_bulk.html', tables=[data.to_html(classes='data')], titles=data.columns.values)
        else:
            return redirect(request.url)

@app.route('/predict',methods=['GET','POST'])
@cross_origin()
def predict():
    if request.method == 'POST':

        Type = request.form.get("gender", False)
        if (Type == 'Male'):
            Type = 0
        elif (Type == 'Female'):
            Type = 1
        elif (Type == 'Enterprise'):  # random
            Type = 2
        elif (Type == 'Unknown'):
            Type = 3

        amount = float(request.form.get("amount", False))
        merchant = float(request.form.get("merchant", False))
        category = float(request.form.get("category", False))
        step = float(request.form.get("step", False))
        age = float(request.form.get("age", False))

        df = pd.DataFrame(
            {"step": step ,"age": age,'gender': Type,  "merchant": merchant,
                    "category": category,"amount": amount  }, index=[0])

        my_prediction = model_Fraud.predict(df)

        # check my_pred ..............................................
        return render_template('fraud_detect_result.html', prediction=my_prediction)

@app.route('/predict_LA',methods=['GET','POST'])
@cross_origin()
def predict_LA():
    if request.method == 'POST':
        married = request.form.get("Married", False)
        if (married=='Yes'):
            married=1
        elif(married=="No"):
            married=0

        property = request.form.get("Property", False)
        if (property == 'Semi-Urban'):
            property_semi = 1
        else:
            property_semi = 0
        if (property == "Urban"):
            property_urban = 1
        else:
            property_urban = 0

        ApplicantIncome = float(request.form.get("ApplicantIncome",False))
        CoapplicantIncome = float(request.form.get("CoapplicantIncome",False))
        LoanAmount = float(request.form.get("LoanAmount",False))
        Loan_Amount_Term = float(request.form.get("Loan_Amount_Term",False))
        Credit_History = float(request.form.get("Credit_History",False))

        LA_prediction = model_LA.predict(ss_LA.fit_transform([[ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History
                                                         ,married,property_semi,property_urban]]))

        return render_template('LA_result.html', prediction=LA_prediction)
    else:
        return render_template('LA_home.html')

@app.route('/predict_LE',methods=['GET','POST'])
@cross_origin()
def predict_LE():
    if request.method == 'POST':

        CurrentLoanAmount = float(request.form.get("CurrentLoanAmount",False))
        CreditScore = float(request.form.get("CreditScore",False))
        AnnualIncome = float(request.form.get("AnnualIncome",False))
        Yearsincurrentjob = float(request.form.get("Yearsincurrentjob",False))
        MonthlyDebt = float(request.form.get("MonthlyDebt",False))
        YearsofCreditHistory = float(request.form.get("YearsofCreditHistory",False))
        Monthssincelastdelinquent = float(request.form.get("Monthssincelastdelinquent",False))
        NumberofOpenAccounts = float(request.form.get("NumberofOpenAccounts",False))
        NumberofCreditProblems = float(request.form.get("NumberofCreditProblems",False))
        CurrentCreditBalance = float(request.form.get("CurrentCreditBalance",False))
        MaximumOpenCredit = float(request.form.get("MaximumOpenCredit",False))
        Term_LongTerm = float(request.form.get("Term_LongTerm",False))

        LE_prediction = model_LE.predict([
            [CurrentLoanAmount, CreditScore, AnnualIncome,
       Yearsincurrentjob, MonthlyDebt, YearsofCreditHistory,
       Monthssincelastdelinquent, NumberofOpenAccounts,
       NumberofCreditProblems, CurrentCreditBalance,
       MaximumOpenCredit, Term_LongTerm]
        ])

        return render_template('LE_result.html', prediction=LE_prediction)
    else:
        return render_template('LE_home.html')

@app.route('/predict_LR',methods=['GET','POST'])
@cross_origin()
def predict_LR():
    if request.method == 'POST':
        Loan_Amount = float(request.form.get("LoanAmount",False))
        Term = float(request.form.get("Loan_Amount_Term",False))
        Interest_Rate = float(request.form.get("Interest_Rate",False))
        Employment_Years = float(request.form.get("Employment_Years",False))
        Annual_Income = float(request.form.get("Annual_Income",False))
        Debt_to_Income = float(request.form.get("Debt_to_Income",False))
        Delinquent_2yr = float(request.form.get("Delinquent_2yr",False))
        Revolving_Cr_Util = float(request.form.get("Revolving_Cr_Util",False))
        Total_Accounts = float(request.form.get("Total_Accounts",False))
        Longest_Credit_Length = float(request.form.get("Longest_Credit_Length",False))
        
        Home_Ownership = request.form.get("Home_Ownership",False)
        if Home_Ownership == 'RENT':
            Home_Ownership = 5
        elif Home_Ownership == 'OWN':
            Home_Ownership = 4
        elif Home_Ownership == 'MORTGAGE':
            Home_Ownership = 1
        elif Home_Ownership == 'OTHER':
            Home_Ownership = 3
        elif Home_Ownership == 'NONE':
            Home_Ownership = 2
        elif Home_Ownership == 'ANY':
            Home_Ownership = 0

        Verification_Status = request.form.get("Verification_Status",False)
        if Verification_Status == 'VERIFIED - income':
            Verification_Status = 1
        elif Verification_Status == 'VERIFIED - income source':
            Verification_Status = 2
        elif Verification_Status == 'not verified':
            Verification_Status = 0
            
        Loan_Purpose = request.form.get("Loan_Purpose",False)
        if Loan_Purpose == 'credit_card':
            Loan_Purpose = 1
        elif Loan_Purpose =='car':
            Loan_Purpose = 0
        elif Loan_Purpose == 'small_business':
            Loan_Purpose = 11
        elif Loan_Purpose == 'other':
            Loan_Purpose = 9
        elif Loan_Purpose == 'wedding':
            Loan_Purpose = 13
        elif Loan_Purpose == 'debt_consolidation':
            Loan_Purpose = 2
        elif Loan_Purpose == 'home_improvement':
            Loan_Purpose = 4
        elif Loan_Purpose == 'major_purchase':
            Loan_Purpose = 6
        elif Loan_Purpose == 'medical':
            Loan_Purpose = 7
        elif Loan_Purpose == 'moving':
            Loan_Purpose = 8
        elif Loan_Purpose == 'renewable_energy':
            Loan_Purpose = 10
        elif Loan_Purpose == 'vacation':
            Loan_Purpose = 12
        elif Loan_Purpose == 'house':
            Loan_Purpose = 5
        elif Loan_Purpose == 'educational':
            Loan_Purpose = 3

        State = request.form.get("State",False)
        if State == 'AK':
            State = 0
        elif State == 'AL':
            State = 1
        elif State == 'AR':
            State = 2
        elif State == 'AZ':
            State = 3
        elif State == 'CA':
            State = 4
        elif State == 'CO':
            State = 5
        elif State == 'CT':
            State = 6
        elif State == 'DC':
            State = 7
        elif State == 'DE':
            State = 8
        elif State == 'FL':
            State = 9
        elif State == 'GA':
            State = 10
        elif State == 'HI':
            State = 11
        elif State == 'IA':
            State = 12
        elif State == 'ID':
            State = 13
        elif State == 'IL':
            State = 14
        elif State == 'IN':
            State = 15
        elif State == 'KS':
            State = 16
        elif State == 'KY':
            State = 17
        elif State == 'LA':
            State = 18
        elif State == 'MA':
            State = 19
        elif State == 'MD':
            State = 20
        elif State == 'ME':
            State = 21
        elif State == 'MI':
            State = 22
        elif State == 'MN':
            State = 23
        elif State == 'MO':
            State = 24
        elif State == 'MS':
            State = 25
        elif State == 'MT':
            State = 26
        elif State == 'NC':
            State = 27
        elif State == 'NE':
            State = 28
        elif State == 'NH':
            State = 29
        elif State == 'NJ':
            State = 30
        elif State == 'NM':
            State = 31
        elif State == 'NV':
            State = 32
        elif State == 'NY':
            State = 33
        elif State == 'OH':
            State = 34
        elif State == 'OK':
            State = 3
        elif State == 'OR':
            State = 36
        elif State == 'PA':
            State = 37
        elif State == 'RI':
            State = 38
        elif State == 'SC':
            State = 39
        elif State == 'SD':
            State = 40
        elif State == 'TN':
            State = 41
        elif State == 'TX':
            State = 42
        elif State == 'UT':
            State = 43
        elif State == 'VA':
            State = 44
        elif State == 'VT':
            State = 45
        elif State == 'WA':
            State = 46
        elif State == 'WI':
            State = 47
        elif State == 'WV':
            State = 48
        elif State == 'WY':
            State = 49


        LR_prediction = model_LR.predict([[Loan_Amount, Term, Interest_Rate, Employment_Years,
        Home_Ownership, Annual_Income, Verification_Status,
        Loan_Purpose, State, Debt_to_Income, Delinquent_2yr,
        Revolving_Cr_Util, Total_Accounts, Longest_Credit_Length]])

        return render_template('loan_risk_result.html', prediction=LR_prediction)
    else:
        return render_template('loan_risk.html')


if __name__ == '__main__':
    # To run on web ..
    #app.run(host='0.0.0.0',port=8080)
    # To run locally ..
    app.run(debug=True)