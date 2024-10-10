User defined functions split into 2 parts:

**eda :**<br> 

| Function    | Desc |
| -------- | ------- |
| pie_chart()  | Used to plot the percentage of different change_status (I=increase,D=decrease,N=no change) and destinations   |
| plot_prices()  | Plots all the prices for a destination with time to trip    |
| price_change_plot() | Sum of change_status_num aggregated on time_to_trip, for all destinations    |


**model :**<br>  

| Function    | Desc |
| -------- | ------- |
| calculate_evaluation_metrics() | Calculates roc_auc and balances_acc for each model  |
| try_model() | Evaluates dummy model vs "smart" model   |
| create_interaction_features() | To create more features by using interactions (X1,X2 = X1*X2) with PolynomialFeatures(interaction_only=True)  |
| make_confusion_matrix() | https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py  |
|resample_data() | Since the problem is imbalanced, this function is used to create balanced datasets based on the least frequent                           class and the most frequent class.  |
