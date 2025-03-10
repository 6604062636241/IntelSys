import streamlit as st

st.title("SVM Model for Income Level Prediction")

st.header("1. Data Preparation")
st.write("""The dataset used for the model development is the Adult Census Income Dataset. 
This dataset contains economic and social factors, such as age, education level, 
occupation, and nationality, that influence income level.""")

st.header("2. SVM Algorithm Theory")
st.write("""Support Vector Machine (SVM) is an algorithm that uses the principle of a hyperplane 
to separate data into distinct classes. It selects the hyperplane that maximizes 
the margin between the nearest data points of each class, which helps improve 
the model’s ability to generalize. SVM can be used for problems where data is linearly 
separable or for more complex problems where the Kernel Trick is applied to transform 
the data into a higher-dimensional space for easier separation.""")

st.header("3. SVM Model Development")
st.subheader("3.1 Data Splitting into Training and Testing Sets")
st.write("""The data is split into a Training Set and a Test Set. The training set allows the model 
to learn from labeled examples, while the test set is used to evaluate the model’s performance 
on unseen data.""")

st.subheader("3.2 Training the SVM Model")
st.write("""The Linear Kernel SVM is chosen because the income level classification is expected to be 
linearly separable. The model learns from the training data and constructs the best hyperplane 
to classify new data.""")

st.subheader("3.3 Model Evaluation")
st.write("""After training, the model's accuracy is evaluated by comparing its predictions against 
the actual values in the test set. This helps determine how well the model generalizes to 
new, unseen data.""")

st.subheader("3.4 Saving the Model")
st.write("""Once the model is trained, the SVM model and the associated transformers are saved 
as a file for future use.""")
