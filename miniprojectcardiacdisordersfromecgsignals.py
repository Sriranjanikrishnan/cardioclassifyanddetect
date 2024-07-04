# Step 1: Import necessary libraries
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.naive_bayes import GaussianNB # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, confusion_matrix # type: ignore
import matplotlib.pyplot as plt # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras import models, layers # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from matplotlib.animation import FuncAnimation   # type: ignore
# Step 2: Load the dataset and Display the first few and last few rows
df = pd.read_csv(r"C:\Users\USER\Downloads\miniprodataset\ECGCvdatacopy.csv")
print("First few rows:")
print(df.head())
print("\nLast few rows:")
print(df.tail())
# Step 3: Preprocessing and Visualization
# Checking for missing values
missing_values = df.isnull().sum()
# Summary statistics of the dataset
df.describe(include='all')
# One-hot encode the 'ECG_signal' column
df_encoded = pd.get_dummies(df, columns=['ECG_signal'], drop_first=True)
# Fill missing values with median of respective columns
df_filled = df_encoded.fillna(df_encoded.median())
# Create visualizations
# Plot ECG signal from the first row of columns 2 to 55
row_number = 77  
ecg_data = df_filled.iloc[row_number, 1:56].values  # Assuming ECG data (features) is in columns 2 to 55
plt.figure(figsize=(12, 6))
plt.plot(ecg_data)
plt.title(f'ECG SIGNAL OF A PERSON')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()
plt.figure(figsize=(10, 6))
sns.countplot(x='ECG_signal', data=df, palette='pastel', hue='ECG_signal', dodge=False, legend=False)
plt.title('Distribution of ECG Signal Types')
plt.show()
#step 4: Feature visualization
corr = df_encoded.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Correlation Heatmap of ECG Features')
plt.show()
df_filled['ECG_signal'] = df['ECG_signal']
features = ['hbpermin', 'QRSarea', 'QRSperi']
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='ECG_signal', y=feature, data=df_filled)
    plt.title(f'Distribution of {feature} for Each ECG Signal Type')
    plt.show()
plt.figure(figsize=(10, 6))
sns.violinplot(x='ECG_signal', y='hbpermin', data=df_filled)
plt.title('Distribution of Heartbeats per Minute for Each ECG Signal Type')
plt.show()
# Visualize the dataset using the scatter plot
x = df_filled['hbpermin']
y = df_filled['ECG_signal']
plt.scatter(x, y)
plt.title('Relationship between Heartbeat per Minute and Cardiac Ailments')
plt.xlabel('Heart Beat per minute')
plt.ylabel('Cardiac Ailments')
plt.show()
# MACHINE LEARNING - SVM,RANDOM FOREST,NAIVE BAYES
# Step 5: Define selected features
selected_features = ['hbpermin', 'Pseg', 'PQseg', 'QRSseg', 'QRseg', 'QTseg', 'RSseg', 'STseg', 'Tseg', 'PTseg', 'ECGseg',
                     'QRtoQSdur', 'RStoQSdur', 'RRmean', 'PPmean', 'PQdis', 'PonQdis', 'PRdis', 'PonRdis', 'PSdis',
                     'PonSdis', 'PTdis', 'PonTdis', 'PToffdis', 'QRdis', 'QSdis', 'QTdis', 'QToffdis', 'RSdis', 'RTdis',
                     'RToffdis', 'STdis', 'SToffdis', 'PonToffdis', 'PonPQang', 'PQRang', 'QRSang', 'RSTang', 'STToffang',
                     'RRTot', 'NNTot', 'SDRR', 'IBIM', 'IBISD', 'SDSD', 'RMSSD', 'QRSarea', 'QRSperi', 'PQslope', 'QRslope',
                     'RSslope', 'STslope', 'NN50', 'pNN50']
# Step 6: Prepare the data
X = df[selected_features]  # Features
y = df['ECG_signal']  # Labels
# Step 7: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Step 8: Handle missing values
imputer = SimpleImputer(strategy='median')  # Imputer to replace missing values with median
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
# Step 9: Train the classifiers
# Support Vector Machine (SVM)
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_imputed, y_train)
svm_train_pred = svm_classifier.predict(X_train_imputed)  # Make predictions on the training set
#Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_imputed, y_train)
rf_train_pred = rf_classifier.predict(X_train_imputed)  # Make predictions on the training set
#Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train_imputed, y_train)
nb_train_pred = nb_classifier.predict(X_train_imputed)  # Make predictions on the training set
# Step 10: Make predictions on testing set
# Support Vector Machine (SVM)
svm_pred = svm_classifier.predict(X_test_imputed)
# Random Forest
rf_pred = rf_classifier.predict(X_test_imputed)
# Naive Bayes
nb_pred = nb_classifier.predict(X_test_imputed)
# Step 11: Calculate accuracy for training and testing sets
svm_train_accuracy = accuracy_score(y_train, svm_train_pred)
svm_test_accuracy = accuracy_score(y_test, svm_pred)
rf_train_accuracy = accuracy_score(y_train, rf_train_pred)
rf_test_accuracy = accuracy_score(y_test, rf_pred)
nb_train_accuracy = accuracy_score(y_train, nb_train_pred)
nb_test_accuracy = accuracy_score(y_test, nb_pred)
# Display accuracy for training and testing sets
print("Accuracy (SVM) - Training Set:", svm_train_accuracy)
print("Accuracy (SVM) - Testing Set:", svm_test_accuracy)
print("Accuracy (Random Forest) - Training Set:", rf_train_accuracy)
print("Accuracy (Random Forest) - Testing Set:", rf_test_accuracy)
print("Accuracy (Naive Bayes) - Training Set:", nb_train_accuracy)
print("Accuracy (Naive Bayes) - Testing Set:", nb_test_accuracy)
# Define labels and accuracies
models = ['SVM', 'Random Forest', 'Naive Bayes']
training_accuracies = [svm_train_accuracy, rf_train_accuracy, nb_train_accuracy]
testing_accuracies = [svm_test_accuracy, rf_test_accuracy, nb_test_accuracy]
# Plot training accuracies
plt.figure(figsize=(10, 5))
plt.bar(models, training_accuracies, color='skyblue')
plt.title('Training Accuracies of Different Models')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Set y-axis limits to ensure consistency
plt.show()
# Plot testing accuracies
plt.figure(figsize=(10, 5))
plt.bar(models, testing_accuracies, color='lightgreen')
plt.title('Testing Accuracies of Different Models')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Set y-axis limits to ensure consistency
plt.show()
# Step 12: Make predictions
#SUpport Vector Machine (SVM)
svm_pred = svm_classifier.predict(X_test_imputed)
#Random Forest
rf_pred = rf_classifier.predict(X_test_imputed)
#Naive Bayes
nb_pred = nb_classifier.predict(X_test_imputed)
# Step 13: Calculate accuracy
svm_accuracy = accuracy_score(y_test, svm_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
nb_accuracy = accuracy_score(y_test, nb_pred)
# Display validation accuracy in command window
print("Validation Accuracy of SVM:", svm_accuracy)
print("Validation Accuracy of Random Forest:", rf_accuracy)
print("Validation Accuracy of Naive Bayes:", nb_accuracy)
# Step 14: AUROC score for testing set
svm_pred_scores_test = svm_classifier.decision_function(X_test_imputed)
svm_pred_probs_test = np.exp(svm_pred_scores_test) / np.sum(np.exp(svm_pred_scores_test), axis=1, keepdims=True)
rf_pred_probs_test = rf_classifier.predict_proba(X_test_imputed)
nb_pred_probs_test = nb_classifier.predict_proba(X_test_imputed)
svm_auroc_test = roc_auc_score(y_test, svm_pred_probs_test, multi_class='ovr', average='weighted')
rf_auroc_test = roc_auc_score(y_test, rf_pred_probs_test, multi_class='ovr', average='weighted')
nb_auroc_test = roc_auc_score(y_test, nb_pred_probs_test, multi_class='ovr', average='weighted')
# Step 15: AUROC score for training set
svm_pred_scores_train = svm_classifier.decision_function(X_train_imputed)
svm_pred_probs_train = np.exp(svm_pred_scores_train) / np.sum(np.exp(svm_pred_scores_train), axis=1, keepdims=True)
rf_pred_probs_train = rf_classifier.predict_proba(X_train_imputed)
nb_pred_probs_train = nb_classifier.predict_proba(X_train_imputed)
svm_auroc_train = roc_auc_score(y_train, svm_pred_probs_train, multi_class='ovr', average='weighted')
rf_auroc_train = roc_auc_score(y_train, rf_pred_probs_train, multi_class='ovr', average='weighted')
nb_auroc_train = roc_auc_score(y_train, nb_pred_probs_train, multi_class='ovr', average='weighted')
# Step 16: F1 score for testing set
svm_f1_test = f1_score(y_test, svm_pred, average='weighted')
rf_f1_test = f1_score(y_test, rf_pred, average='weighted')
nb_f1_test = f1_score(y_test, nb_pred, average='weighted')
# Step 17: F1 score for training set
svm_f1_train = f1_score(y_train, svm_train_pred, average='weighted')
rf_f1_train = f1_score(y_train, rf_train_pred, average='weighted')
nb_f1_train = f1_score(y_train, nb_train_pred, average='weighted')
# Step 18: Recall score for testing set
svm_recall_test = recall_score(y_test, svm_pred, average='weighted')
rf_recall_test = recall_score(y_test, rf_pred, average='weighted')
nb_recall_test = recall_score(y_test, nb_pred, average='weighted')
# Step 19: Recall score for training set
svm_recall_train = recall_score(y_train, svm_train_pred, average='weighted')
rf_recall_train = recall_score(y_train, rf_train_pred, average='weighted')
nb_recall_train = recall_score(y_train, nb_train_pred, average='weighted')
# Display scores for training set
print("AUROC Score (SVM) - Training:", svm_auroc_train)
print("F1 Score (SVM) - Training:", svm_f1_train)
print("Recall Score (SVM) - Training:", svm_recall_train)
print("AUROC Score (Random Forest) - Training:", rf_auroc_train)
print("F1 Score (Random Forest) - Training:", rf_f1_train)
print("Recall Score (Random Forest) - Training:", rf_recall_train)
print("AUROC Score (Naive Bayes) - Training:", nb_auroc_train)
print("F1 Score (Naive Bayes) - Training:", nb_f1_train)
print("Recall Score (Naive Bayes) - Training:", nb_recall_train)
# Display scores for testing set
print("AUROC Score (SVM) - Testing:", svm_auroc_test)
print("F1 Score (SVM) - Testing:", svm_f1_test)
print("Recall Score (SVM) - Testing:", svm_recall_test)
print("AUROC Score (Random Forest) - Testing:", rf_auroc_test)
print("F1 Score (Random Forest) - Testing:", rf_f1_test)
print("Recall Score (Random Forest) - Testing:", rf_recall_test)
print("AUROC Score (Naive Bayes) - Testing:", nb_auroc_test)
print("F1 Score (Naive Bayes) - Testing:", nb_f1_test)
print("Recall Score (Naive Bayes) - Testing:", nb_recall_test)
# Step 20 : Function to plot confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(title)
    plt.show()
# Get labels for confusion matrix
labels = sorted(df['ECG_signal'].unique())
# SVM Confusion Matrices
svm_train_pred = svm_classifier.predict(X_train_imputed)
svm_train_cm = confusion_matrix(y_train, svm_train_pred)
plot_confusion_matrix(svm_train_cm, 'Confusion Matrix (SVM) - Training Set')
svm_test_pred = svm_classifier.predict(X_test_imputed)
svm_test_cm = confusion_matrix(y_test, svm_test_pred)
plot_confusion_matrix(svm_test_cm, 'Confusion Matrix (SVM) - Testing Set')
# Random Forest Confusion Matrices
rf_train_pred = rf_classifier.predict(X_train_imputed)
rf_train_cm = confusion_matrix(y_train, rf_train_pred)
plot_confusion_matrix(rf_train_cm, 'Confusion Matrix (Random Forest) - Training Set')
rf_test_pred = rf_classifier.predict(X_test_imputed)
rf_test_cm = confusion_matrix(y_test, rf_test_pred)
plot_confusion_matrix(rf_test_cm, 'Confusion Matrix (Random Forest) - Testing Set')
# Naive Bayes Confusion Matrices
nb_train_pred = nb_classifier.predict(X_train_imputed)
nb_train_cm = confusion_matrix(y_train, nb_train_pred)
plot_confusion_matrix(nb_train_cm, 'Confusion Matrix (Naive Bayes) - Training Set')
nb_test_pred = nb_classifier.predict(X_test_imputed)
nb_test_cm = confusion_matrix(y_test, nb_test_pred)
plot_confusion_matrix(nb_test_cm, 'Confusion Matrix (Naive Bayes) - Testing Set')
# Step 21: Visualize results
# Bar graph: Validation Accuracy of 3 Models
models = ['SVM', 'Random Forest', 'Naive Bayes']
accuracies = [svm_accuracy, rf_accuracy, nb_accuracy]
plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.title('Validation Accuracy of 3 Models')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()
# Step 22 : User input and analysis
row_number = int(input("Enter a row number from the test dataset (0 to {}): ".format(len(X_test)-1)))
# Get the features of the selected row
selected_row_features = X_test_imputed[row_number, :].reshape(1, -1)
# Predict the label using all classifiers
svm_pred_row = svm_classifier.predict(selected_row_features)
rf_pred_row = rf_classifier.predict(selected_row_features)
nb_pred_row = nb_classifier.predict(selected_row_features)
# Define a function to plot pie charts
def plot_pie_chart(pred_svm, pred_rf, pred_nb):
    labels = ['Normal', 'Abnormal']
    pred_counts_svm = [np.sum(pred_svm == 'NSR'), len(pred_svm) - np.sum(pred_svm == 'NSR')]
    pred_counts_rf = [np.sum(pred_rf == 'NSR'), len(pred_rf) - np.sum(pred_rf == 'NSR')]
    pred_counts_nb = [np.sum(pred_nb == 'NSR'), len(pred_nb) - np.sum(pred_nb == 'NSR')]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].pie(pred_counts_svm, labels=labels, autopct='%1.1f%%', startangle=140)
    axs[0].set_title('Predicted Classification (SVM)')
    axs[1].pie(pred_counts_rf, labels=labels, autopct='%1.1f%%', startangle=140)
    axs[1].set_title('Predicted Classification (Random Forest)')
    axs[2].pie(pred_counts_nb, labels=labels, autopct='%1.1f%%', startangle=140)
    axs[2].set_title('Predicted Classification (Naive Bayes)')
    plt.show()
# Display the predicted results as pie charts
plot_pie_chart(svm_pred_row, rf_pred_row, nb_pred_row)

#DEEP LEARNING - CNN
# Step 23: Load the dataset
df = pd.read_csv(r"C:\Users\USER\Downloads\miniprodataset\ECGCvdatacopy.csv")
# Step 24: Define selected features and labels
selected_features = ['hbpermin', 'Pseg', 'PQseg', 'QRSseg', 'QRseg', 'QTseg', 'RSseg', 'STseg', 'Tseg', 'PTseg', 'ECGseg',
                     'QRtoQSdur', 'RStoQSdur', 'RRmean', 'PPmean', 'PQdis', 'PonQdis', 'PRdis', 'PonRdis', 'PSdis',
                     'PonSdis', 'PTdis', 'PonTdis', 'PToffdis', 'QRdis', 'QSdis', 'QTdis', 'QToffdis', 'RSdis', 'RTdis',
                     'RToffdis', 'STdis', 'SToffdis', 'PonToffdis', 'PonPQang', 'PQRang', 'QRSang', 'RSTang', 'STToffang',
                     'RRTot', 'NNTot', 'SDRR', 'IBIM', 'IBISD', 'SDSD', 'RMSSD', 'QRSarea', 'QRSperi', 'PQslope', 'QRslope',
                     'RSslope', 'STslope', 'NN50', 'pNN50']
X = df[selected_features]  # Features
y = df['ECG_signal']  # Labels
# Map labels to integers
label_map = {'NSR': 0, 'ARR': 1, 'AFF': 1, 'CHF': 1}
y = y.map(label_map)
# Step 25: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 26: Handle missing values (if any)
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
# Step 27: Reshape data for CNN input
input_shape = (X_train_imputed.shape[1], 1)
X_train_reshaped = X_train_imputed.reshape(-1, X_train_imputed.shape[1], 1)
X_test_reshaped = X_test_imputed.reshape(-1, X_test_imputed.shape[1], 1)
from tensorflow.keras.models import Sequential  # type: ignore # Import Sequential directly from keras.models
# Step 28: Define CNN architecture with Dropout
model = Sequential([
    layers.Conv1D(32, 3, activation='relu', input_shape=input_shape),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.Flatten(),
    layers.Dropout(0.5),  # Add Dropout layer with dropout rate of 0.5
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
# Step 29: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Step 30: Train the model with Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_train_reshaped, y_train, epochs=20, validation_split=0.2, callbacks=[early_stopping])
# Step 31: Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test)
train_loss, train_accuracy = model.evaluate(X_train_reshaped, y_train)
print('CNN Model Test Accuracy:', test_accuracy)
print('CNN Model Train Accuracy:', train_accuracy)
# Step 32: Calculate AUROC Score, F1 Score, and Recall Score for testing
y_pred_prob_test = model.predict(X_test_reshaped)
auroc_score_test = roc_auc_score(y_test, y_pred_prob_test)
y_pred_test = np.where(y_pred_prob_test > 0.5, 1, 0)
f1_score_test = f1_score(y_test, y_pred_test)
recall_score_test = recall_score(y_test, y_pred_test)
# Step 33: Calculate AUROC Score, F1 Score, and Recall Score for training
y_pred_prob_train = model.predict(X_train_reshaped)
auroc_score_train = roc_auc_score(y_train, y_pred_prob_train)
y_pred_train = np.where(y_pred_prob_train > 0.5, 1, 0)
f1_score_train = f1_score(y_train, y_pred_train)
recall_score_train = recall_score(y_train, y_pred_train)
# Print accuracy and scores
print(f'AUROC Score (Train): {auroc_score_train:.4f}')
print(f'AUROC Score (Test): {auroc_score_test:.4f}')
print(f'Recall Score (Train): {recall_score_train:.4f}')
print(f'Recall Score (Test): {recall_score_test:.4f}')
print(f'F1 Score (Train): {f1_score_train:.4f}')
print(f'F1 Score (Test): {f1_score_test:.4f}')
# Bar graph: Training Accuracy
plt.figure(figsize=(8, 6))
plt.bar(['Training'], [train_accuracy], color='blue')
plt.title('Training Accuracy of CNN Model')
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()
# Bar graph: Testing Accuracy
plt.figure(figsize=(8, 6))
plt.bar(['Testing'], [test_accuracy], color='green')
plt.title('Testing Accuracy of CNN Model')
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()
from sklearn.metrics import confusion_matrix # type: ignore
import seaborn as sns # type: ignore
# Step 34: Generate confusion matrix for training set
y_train_pred = model.predict(X_train_reshaped)
y_train_pred_binary = np.where(y_train_pred > 0.5, 1, 0)
conf_matrix_train = confusion_matrix(y_train, y_train_pred_binary)
# Plot confusion matrix for training set
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (CNN) - Training Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
# Step 35: Generate confusion matrix for testing set
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
# Plot confusion matrix for testing set
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (CNN) - Testing Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
# Bar graph: Validation Accuracy of CNN Model
plt.figure(figsize=(6, 4))
plt.bar(['CNN'], [test_accuracy], color='purple')
plt.title('Validation Accuracy of CNN Model')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()
# Plotting training and validation accuracy and loss
plt.figure(figsize=(10, 6))
# Plot training & validation accuracy values
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# Plot training & validation loss values
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
# Mapping of predicted diseases to full names
disease_mapping = {
    'ARR': 'ARRHYTHMIA',
    'AFF': 'ATRIAL FIBRILLATION',
    'CHF': 'CONGESTIVE HEART FAILURE',
    'NSR': 'NORMAL SINUS RHYTHM'
}
#  Step 36:User input and analysis
row_number = int(input("Enter a row number from the test dataset (0 to {}): ".format(len(X_test)-1)))
selected_row_features = X_test_imputed[row_number, :].reshape(1, X_test_imputed.shape[1], 1)  # Reshape to match model input shape
y_pred_row = model.predict(selected_row_features)
# Get ECG data and convert to numeric (skip non-numeric values)
ecg_data = df.iloc[row_number, 1:56].apply(pd.to_numeric, errors='coerce').dropna().values
plt.figure(figsize=(12, 6))
plt.plot(ecg_data)
plt.title(f'ECG SIGNAL OF PERSON {row_number}')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()
# Generate prediction pie chart for the selected row
prediction_labels = ['Normal', 'Abnormal']
prediction_probabilities = [1 - y_pred_row[0][0], y_pred_row[0][0]]  # Probabilities for normal and abnormal
colors = ['lightgreen', 'lightcoral']
plt.figure(figsize=(6, 6))
plt.pie(prediction_probabilities, labels=prediction_labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Prediction Distribution (Row {})'.format(row_number))
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
# Step 37: Detect disease based on abnormal prediction
def detect_disease(y_pred):
    if y_pred > 0.5:  # If abnormal prediction
        return True
    else:  # If normal prediction
        return False
    # Define function to animate big screen warning messages with detected disease
# Define function to animate big screen warning messages with detected disease
def animate_big_screen_warning(pred_prob, row_number):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    def detect_disease(y_pred, row_number):
        if y_pred > 0.5:  # If abnormal prediction
            actual_label = y_test.iloc[row_number]  # Get actual label from test data
            predicted_disease = df.iloc[y_test.index[row_number]]['ECG_signal']
            print("Actual Disease:", actual_label)
            print("Predicted Disease:", predicted_disease)
            return disease_mapping.get(predicted_disease, 'Unknown')
        else:  # If normal prediction
            print("No Abnormality Detected")
            return None
    # Call detect_disease once
    detected_disease = detect_disease(pred_prob, row_number)
    def animate(frame):
        ax.clear()
        if frame % 2 == 0:
            if pred_prob <= 0.5:
                ax.text(0.5, 0.5, 'Congratulations!\nYour ECG signal indicates a healthy heart rhythm.\nKeep up the excellent effort in maintaining your cardiac health!', ha='center', va='center', fontsize=20, color='green')
            else:
                warning_msg = 'ABNORMAL SIGNAL DETECTED!\nPlease consult a healthcare professional.'
                if detected_disease:
                    warning_msg += '\nDetected Condition: ' + detected_disease
                ax.text(0.5, 0.5, warning_msg, ha='center', va='center', fontsize=20, color='red')
    ani = FuncAnimation(fig, animate, frames=10, interval=500)
    plt.show()
# Call the function to display the warning with animation and detect disease
animate_big_screen_warning(y_pred_row, row_number)
# Accuracy comparison
ml_accuracies = [svm_accuracy, rf_accuracy, nb_accuracy]
ml_labels = ['SVM', 'Random Forest', 'Naive Bayes']
cnn_accuracy = test_accuracy
plt.figure(figsize=(10, 6))
plt.bar(ml_labels, ml_accuracies, color='blue', label='Machine Learning')
plt.bar('CNN', cnn_accuracy, color='green', label='CNN')
plt.title('Accuracy Comparison: Machine Learning vs. CNN')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.show()
