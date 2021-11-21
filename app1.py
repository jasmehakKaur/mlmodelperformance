import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

# Calculates confusion matrix
def calc_confusion_matrix(input_data):
    Y_actual = input_data.iloc[:,0]
    Y_predicted = input_data.iloc[:,1]
    confusion_matrix_array = confusion_matrix(Y_actual, Y_predicted)
    confusion_matrix_df = pd.DataFrame(confusion_matrix_array, columns=['Actual','Predicted'], index=['Actual','Predicted'])
    return confusion_matrix_df

# Calculates performance metrics
def calc_metrics(input_data):
    Y_actual = input_data.iloc[:,0]
    Y_predicted = input_data.iloc[:,1]
    acc = accuracy_score(Y_actual, Y_predicted)
    acc_series = pd.Series(acc, name='Accuracy')
    balanced_accuracy = balanced_accuracy_score(Y_actual, Y_predicted)
    balanced_accuracy_series = pd.Series(balanced_accuracy, name='Balanced_Accuracy')
    precision = precision_score(Y_actual, Y_predicted, average='weighted')
    recall = recall_score(Y_actual, Y_predicted, average='weighted')
    f1 = f1_score(Y_actual, Y_predicted, average='weighted')
    precision_series = pd.Series(precision, name='Precision')
    recall_series = pd.Series(recall, name='Recall')
    f1_series = pd.Series(f1, name='F1')

    df = pd.concat( [acc_series, balanced_accuracy_series, precision_series,
                     recall_series, f1_series], axis=1 )
    return df

# Load example data
def load_example_data():
    df = pd.read_csv('eg_data.csv')
    return df


# Sidebar - Header
st.sidebar.header('Input panel')
st.sidebar.markdown("""
[Example CSV file](https://github.com/jasmehakKaur/ocr/files/7576588/eg_data.csv)
""")

# Sidebar panel - Upload input file
uploaded_file = st.sidebar.file_uploader('Upload your input CSV file', type=['csv'])

# Sidebar panel - Performance metrics
performance_metrics = ['Accuracy', 'Balanced_Accuracy', 'Precision', 'Recall', 'F1']
selected_metrics = st.sidebar.multiselect('Performance metrics', performance_metrics, performance_metrics)

# Main panel

st.title('ML Model Performance Calculator App')
st.markdown("""
This app calculates the model performance metrics given the actual and predicted values.
""")

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    confusion_matrix_df = calc_confusion_matrix(input_df)
    metrics_df = calc_metrics(input_df)
    selected_metrics_df = metrics_df[ selected_metrics ]
    st.header('Input data')
    st.write(input_df)
    st.header('Confusion matrix')
    st.write(confusion_matrix_df)
    st.header('Performance metrics')
    st.write(selected_metrics_df)
    
else:
    st.info('Awaiting the upload of the input file.')
    if st.button('Use Example Data'):
        input_df = load_example_data()
        confusion_matrix_df = calc_confusion_matrix(input_df)
        metrics_df = calc_metrics(input_df)
        selected_metrics_df = metrics_df[ selected_metrics ]
        st.header('Input data')
        st.write(input_df)
        st.header('Confusion matrix')
        st.write(confusion_matrix_df)
        st.header('Performance metrics')
        st.write(selected_metrics_df)
        

