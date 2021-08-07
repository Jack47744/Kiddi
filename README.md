
For backend developers
There are few steps that you must do before running the code.
1. pip3 install numpy
2. pip3 install pandas
3. pip3 install -U scikit-learn
4. pip3 install xgboost
5. pip3 install matplotlib
6. pip3 install openpyxl

To get patients score please run kiddi_get_score.py

Then you can change the parameter in kiddi_get_score.py
Args are model_path, feature_list, start_predict_date, end_predict_date
Return dataframe consists of 3 columns which are 'idd', 'ft_data_dt', 'score'

You may disable plot function in train_model() function
Example is in colab https://colab.research.google.com/drive/1x-liLNJJJWmTReaqjmqE68vgEVjngnFw?usp=sharing

For data scientist
You can obseve kiddi_ml.py as a main class for Kiddi.
Model training is coded in kiddi_train_model.py