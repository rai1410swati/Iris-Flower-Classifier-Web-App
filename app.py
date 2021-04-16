import pandas as pd
import numpy as np 
import pickle
import streamlit as st 
from PIL import Image

# loading in the model to predict on the data
pickle_in = open('knn.pkl','rb')
classifier = pickle.load(pickle_in)
pickle_in1 = open('log_reg.pkl','rb')
classifier1 = pickle.load(pickle_in1)

# defining the function which will make the prediction using the data which the user inputs
def prediction(sepal_length,sepal_width,petal_length,petal_width):
    prediction = classifier.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    print(prediction)
    return prediction

def prediction1(sepal_length,sepal_width,petal_length,petal_width):
    prediction1 = classifier1.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    print(prediction1)
    return prediction1

# this is the main function in which we define our webpage
def main():
    st.title('Iris Flower Prediction')

    # here we define some of the front end elements of the web page like font and background color etc...
    html_temp = """
    <div style = "background-color:yellow;
                  padding:13px">
    <h1 style = "color:black; text-align:center;">
    Streamlit Iris Flower Classifier ML App </h1>
    </div>
    """

    st.sidebar.header('ML Algorithms')
    algorithm = st.sidebar.selectbox("Choose Algorithm",["Logistic Regression","K Nearest Neighbour"])


    # this line allows us to display the front end aspects we have defined in above code
    st.markdown(html_temp,unsafe_allow_html=True)

    sepal_length = st.text_input("Sepal Length", "Type Here")
    sepal_width = st.text_input("Sepal Width", "Type Here")
    petal_length = st.text_input("Petal Length", "Type Here")
    petal_width = st.text_input("Petal Width", "Type Here")
    result = " "

    if (algorithm == "Logistic Regression"):
        if st.sidebar.button("Predict"):
            result = prediction(sepal_length,sepal_width,petal_length,petal_width)
    elif (algorithm == "K Nearest Neighbour"):
        if st.sidebar.button("Predict"):
            result = prediction1(sepal_length,sepal_width,petal_length,petal_width)

    st.success('The output is {}'.format(result))

if __name__ == '__main__':
    main()
    
    

