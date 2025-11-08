#!/usr/bin/env python
# coding: utf-8

import pickle

import uvicorn

from fastapi import FastAPI

app=FastAPI(title='Churn-Prediction')

with open('model.bin','rb') as f_in:
    pipeline = pickle.load(f_in)
    
def predict_single(customer):
    result = pipeline.predict_proba(customer)[0,1]
    return result


@app.post("/predict")
def predict(customer):
    prob = predict_single(customer)
    
    return {
        'churn_probability': prob,
        'churn':bool(prob >=0.5)
    }

if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=9696)