#%%
from sklearn.utils import resample
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score,balanced_accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer,PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from itertools import combinations
import seaborn as sns




################udf:s used to create classification model################################################



def calculate_evaluation_metrics(y_true,y_pred):
     
    #Calculate evaluation metics roc and acc for dummy and model

    le = LabelEncoder()
    y_true_encoded = le.fit_transform(y_true)
    y_pred_encoded = le.transform(y_pred)

    # One-hot encode the true labels for roc_auc_score
    onehot_encoder = OneHotEncoder()
    y_true_onehot = onehot_encoder.fit_transform(y_true_encoded.reshape(-1, 1)).toarray()

    # Binarize the predicted labels for roc_auc_score
    label_binarizer = LabelBinarizer()
    y_pred_binarized = label_binarizer.fit_transform(y_pred_encoded)

    # Ensure y_pred_binarized has the correct shape for roc_auc_score
    if y_pred_binarized.shape[1] == 1: 
         y_pred_binarized = np.hstack([1 - y_pred_binarized, y_pred_binarized])

    roc_auc = roc_auc_score(y_true_onehot, y_pred_binarized, multi_class='ovo')

    balanced_acc = balanced_accuracy_score(y_true_encoded, y_pred_encoded)

    return roc_auc,balanced_acc




def evaluate_model(y_true,y_pred_model,y_pred_dummy,title):


    # Calculate metrics
    roc_auc_dummy,balanced_acc_dummy=calculate_evaluation_metrics(y_true,y_pred_dummy)
    roc_auc_model,balanced_acc_model=calculate_evaluation_metrics(y_true,y_pred_model)
  
    groups = ['dummy', 'model']

    roc_auc = [round(roc_auc_dummy,4), round(roc_auc_model,4)]
    balanced_acc = [round(balanced_acc_dummy,4), round(balanced_acc_model,4)]

    plot_df = pd.DataFrame({
    'group': groups,
    'roc_auc': roc_auc,
    'balanced_acc': balanced_acc
    })

    fig, ax = plt.subplots()

    # Set the width of each bar
    bar_width = 0.35

    # Set the positions of the bars on the x-axis
    index = range(len(groups))

    # Plot roc_auc bars
    bar1 = ax.bar(index, plot_df['roc_auc'], bar_width, label='ROC AUC')

    # Plot balanced_acc bars next to roc_auc bars
    bar2 = ax.bar([i + bar_width for i in index], plot_df['balanced_acc'], bar_width, label='Balanced Accuracy')

    # Adding labels
    ax.set_xlabel('')
    ax.set_ylabel('Scores')
    ax.set_title(f"{title}: roc_auc-ovo and balanced accuracy")
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(plot_df['group'])

    # Adding the legend
    ax.legend()

    for bar in bar1:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    for bar in bar2:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


    # Display the plot
    plt.show()





def try_model(X_train, X_test, y_train, y_test,inter,smart_model,dummy_model,plot_title):
       

    # to remove DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().y = column_or_1d(y, warn=True)
      y_train = y_train.values.ravel()
      y_test = y_test.values.ravel()

   

      if inter=='Y':
            
                    X_inter_df_train = create_interaction_features(X_train)
                    X_inter_df_test = create_interaction_features(X_test)

                    # Concatenate original features with interaction features
                    X_inter_df_train = pd.concat([X_train, X_inter_df_train], axis=1)
                    X_inter_df_test = pd.concat([X_test, X_inter_df_test], axis=1)
                   

                    # SelectKBest on the training set
                    np.random.seed(42)
                    selector = SelectKBest(mutual_info_classif, k=15)
                    X_new = selector.fit_transform(X_inter_df_train, y_train)
                    selected_indices = selector.get_support(indices=True)
                    selected_feature_names_c_c = X_inter_df_train.columns[selected_indices]

                    # Ensure both train and test datasets have the same features
                    X_train = X_inter_df_train[selected_feature_names_c_c]
                    X_test = X_inter_df_test[selected_feature_names_c_c]

                    clf = smart_model.fit(X_train, y_train)



      else:
             
             clf = smart_model.fit(X_train, y_train)
             
            


      y_pred_model = clf.predict(X_test)

      cm = confusion_matrix(y_test, y_pred_model)


      make_confusion_matrix(cm, figsize=(8,6), categories = ['D','I','N'],cbar=False, title=f'CF Matrix: Model {plot_title}')
   


      y_pred_dummy = dummy_model.predict(X_test)

      cm = confusion_matrix(y_test, y_pred_dummy)
                   
      make_confusion_matrix(cm, figsize=(8,6), categories = ['D','I','N'],cbar=False, title='CF Matrix: Dummy Model')

      evaluate_model(y_test,y_pred_model,y_pred_dummy,plot_title)
      
      
      


    



def create_interaction_features(X):


    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    X_inter=poly.fit_transform(X)
    X_inter=pd.DataFrame(data=X_inter)
    original_feature_names = X.columns

    X_inter_df = pd.DataFrame(X_inter, columns=poly.get_feature_names_out(X.columns))
        
        
    interaction_feature_names = []
    for i, feature_name_i in enumerate(original_feature_names):
        for j, feature_name_j in enumerate(original_feature_names[i + 1:]):
            interaction_feature_names.append(f"{feature_name_i}*{feature_name_j}")


    interaction_features = []
    for feature_name_i, feature_name_j in combinations(original_feature_names, 2):
        interaction_features.append(X[feature_name_i] * X[feature_name_j])

    X_inter_df = pd.concat(interaction_features, axis=1)
    X_inter_df.columns = interaction_feature_names

    # Concatenate the original features with the interaction features DataFrame
    X_inter_df = pd.concat([X, X_inter_df], axis=1)

    #remove duplicate column names
    X_inter_df = X_inter_df.loc[:,~X_inter_df.columns.duplicated()].copy()



    return X_inter_df


def resample_data(Modeldata_train):
     
     #under-sample- least frequent:

    df_a = Modeldata_train[Modeldata_train['change_status'] == 'I']
    df_b = Modeldata_train[Modeldata_train['change_status'] == 'D']
    df_c = Modeldata_train[Modeldata_train['change_status'] == 'N']


    min_class_count = Modeldata_train['change_status'].value_counts().min()


    df_a_under = resample(df_a, replace=False, n_samples=min_class_count, random_state=42)
    df_b_under = resample(df_b, replace=False, n_samples=min_class_count, random_state=42)
    df_c_under = resample(df_c, replace=False, n_samples=min_class_count, random_state=42)

    # Combine the under-sampled dataframes
    df_balanced = pd.concat([df_a_under, df_b_under, df_c_under])



    y_train_least=df_balanced[['change_status']]
    X_train_least=df_balanced[['time_to_trip','weekend','run_date_day_number','price','price_level','to_nice','to_amsterdam']]



    max_class_count = Modeldata_train['change_status'].value_counts().max()

    # Over-sample each class to the maximum class count
    df_a_over = resample(df_a, replace=True, n_samples=max_class_count, random_state=42)
    df_b_over = resample(df_b, replace=True, n_samples=max_class_count, random_state=42)
    df_c_over = resample(df_c, replace=True, n_samples=max_class_count, random_state=42)

    # Combine the over-sampled dataframes
    df_balanced = pd.concat([df_a_over, df_b_over, df_c_over])


    y_train_most=df_balanced[['change_status']]
    X_train_most=df_balanced[['time_to_trip','weekend','run_date_day_number','price','price_level','to_nice','to_amsterdam']]



    return X_train_least,y_train_least,X_train_most,y_train_most



'''
Function taken from: https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
'''
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.

    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = ""#"\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)

    plt.show()


# %%
