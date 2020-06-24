# Diabetes Classifier Using Logistic Regression
Logistic regression from scratch in Python

Author: *Jomon Joshy George*

This example uses gradient descent to fit the model.

DataSet :
1. Context:

    This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

    Content:

    The dataset consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

2. Columns:

    - Pregnancies:  Number of times pregnant
    - Glucose:  Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    - BloodPressure:  Diastolic blood pressure (mm Hg)
    - SkinThickness:  Triceps skin fold thickness (mm)
    - Insulin:  2-Hour serum insulin (mu U/ml)
    - BMI:  Body mass index (weight in kg/(height in m)^2)
    - DiabetesPedigreeFunction:  Diabetes pedigree function
    - Age:  Age (years)
    - Outcome:  Class variable (0 or 1) 268 of 500 are 1, the others are 0
3. Variables :

    Each attribute is a potential risk factor. There are both demographic, behavioural and medical risk factors.
    
    Demographic: (Sex, Age)
    
    Behavioural: (currentSmoker, cigsPerDay)
    
    Medical( history): (BPMeds, prevalentStroke, prevalentHyp, diabetes)
    
    Medical(current): (totChol, sysBP, diaBP, BMI, heartRate, glucose)
    
    Predict variable (desired target): (10 year risk of coronary heart disease CHD)
    
    Outcome: Class variable (0 or 1)
