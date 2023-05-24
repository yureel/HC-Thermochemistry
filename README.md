# HC-Thermochemistry
Trained machine learning models for the prediction of the standard enthalpy of formation, standard molar entropy, and heat capacity of a wide range of hydrocarbons, hydrocarbon radicals, and carbenium ions. 

# Prerequisites
- Python 3.6 or higher
- Pytorch
- Pandas
- Numpy
- Scikit-learn

# Property Prediction
Run predict_carb.py and adapt the "type" in line 138 and the type of molecule encoder in line 1 of naphtha_props/features for prediction of carbenium ions, radicals, or regular hydrocarbons. The molecules to predict can be put in a .csv file according to the specified templates.
