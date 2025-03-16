import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('adult.csv')

selected_columns = ['age', 'workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'native.country', 'income']
df = df[selected_columns]

categorical_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'native.country']
encoder_dict = {}
for col in categorical_columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoder_dict[col] = encoder

target_encoder = LabelEncoder()
df['income'] = target_encoder.fit_transform(df['income'])

scaler = StandardScaler()
df[['age']] = scaler.fit_transform(df[['age']])

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['income']), df['income'], test_size=0.2, random_state=42)


svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

train_predictions = svm_model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

test_predictions = svm_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder_dict, 'encoders.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')
joblib.dump(X_train.columns, 'columns.pkl')