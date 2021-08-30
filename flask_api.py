# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:50:04 2020
Modified on Sun Aug 29 21:30:00 2021

@author: krish.naik
@modified by: debajyoti.das
"""

from flask import Flask, request
import numpy as np
import pandas as pd
import pickle
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("/Users/debajyotidas/Documents/GitHub/Deployment of ML Models using Cloud Frameworks/Bank Note Authentication model deployment using Flasgger/classifier.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/') #decorator
def welcome():
  #Defining the landing web page
  return "Welcome All"

#@app.route('/predict',methods=["Get"])
@app.route('/predict') #when no method is provided, it assumes methods=["Get"] by default
def predict_note_authentication():
  #The below block in """ """ is used to instruct Flasgger as to how to create the Web UI. 
  #The initial block is a description block
  #The parameters section informs of the parameters, which are 4 in our case: variance,skewness,curtosis,entropy. 
  #The parameter formats are provided in flasgger github
  #Similarly, with responses, if we receive 200, Flasgger will return the model prediction
  
  """Let's Authenticate the Banks Note
  This is using docstrings for specifications.
  ---
  parameters:
    - name: variance
      in: query
      type: number
      required: true
    - name: skewness
      in: query
      type: number
      required: true
    - name: curtosis
      in: query
      type: number
      required: true
    - name: entropy
      in: query
      type: number
      required: true
  responses:
      200:
          description: The output values
  """
  variance=request.args.get("variance")
  skewness=request.args.get("skewness")
  curtosis=request.args.get("curtosis")
  entropy=request.args.get("entropy")
  prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
  #print(prediction) #removing this print statement
  return "Hello The answer is: "+str(prediction)

@app.route('/predict_file',methods=["POST"]) #this time as we have a file and we can't pass all the features via a single URL. Hence we use the POST method.
def predict_note_file():
  
  """Let's Authenticate the Banks Note
  This is using docstrings for specifications.
  ---
  parameters:
    - name: file
      in: formData
      type: file
      required: true
  responses:
      200:
          description: The output values
  """
  df_test=pd.read_csv(request.files.get("file"))
  #print(df_test.head()) #removing this print statement
  prediction=classifier.predict(df_test)
  return "The predicted values for the csv file is: "+str(list(prediction))
  #return str(list(prediction))

if __name__=='__main__':
  app.run()
