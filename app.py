from flask import *
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
df=pd.read_csv('cars.csv')
cars=pd.get_dummies(df['Car Model'],prefix='car')
new_df=pd.concat([df,cars],axis='columns')
new_df=new_df.drop(['Car Model','car_Mercedez Benz C class'],axis='columns')
x_data=new_df.drop(['Sell Price($)'],axis='columns')
y_data=new_df['Sell Price($)']
model=LinearRegression()
model.fit(x_data,y_data)
joblib.dump(model,'model')
md=joblib.load('model')
app=Flask(__name__)
@app.route('/')
def home():
	return render_template('index.html')
@app.route('/',methods=['POST'])
def result():
	car=request.form['model']
	mileage=float(request.form['mileage'])
	age=float(request.form['age'])
	if car=='BMW X5':
		pred=md.predict([[mileage,age,0,1]])
	elif car=="Audi A5":
		pred=md.predict([[mileage,age,1,0]])
	else:
		pred=md.predict([[mileage,age,0,0]])
	result=[car,mileage,age,round(float(pred))]
	cols=['Car Model','Mileage','Age','Predicted Price($)']
	return render_template('result.html',result=[cols,result])

if __name__=='__main__':
	app.run(debug=True)
