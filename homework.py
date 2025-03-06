#1b: write code (with AI assistant) to build a naive Bayes and KNN classifier. You can use the hamspam.csvto test it out. 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_path = r"C:\Users\shris\OneDrive\Desktop\homework\hamspam.csv.csv"
df = pd.read_csv(file_path)

# Display the first few rows
print("Dataset loaded successfully!")
print(df.head())

# Convert categorical values into numerical values
df['Contains Link'] = df['Contains Link'].map({'Yes': 1, 'No': 0})
df['Contains Money Words'] = df['Contains Money Words'].map({'Yes': 1, 'No': 0})
df['Length'] = df['Length'].map({'Long': 1, 'Short': 0})
df['Class'] = df['Class'].map({'Spam': 1, 'Ham': 0})  # Target variable

# Features (X) and Target (y)
X = df[['Contains Link', 'Contains Money Words', 'Length']]  # Input features
y = df['Class']  # Target variable (Spam or Ham)

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naïve Bayes Classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)

# Train KNN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)

# Evaluate Models
print("\n📌 Naïve Bayes Classifier Performance:")
print(classification_report(y_test, y_pred_nb))
print("Accuracy:", accuracy_score(y_test, y_pred_nb))

print("\n📌 KNN Classifier Performance:")
print(classification_report(y_test, y_pred_knn))
print("Accuracy:", accuracy_score(y_test, y_pred_knn))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load the dataset
file_path = r"C:\Users\shris\OneDrive\Desktop\homework\roc_data.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Define thresholds
thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]

# Prepare a list to store results
roc_data = []

# Compute TP, FP, TN, FN, TPR, and FPR for each threshold
for threshold in thresholds:
    df['prediction_label'] = (df['Prediction'] >= threshold).astype(int)
    
    TP = ((df['prediction_label'] == 1) & (df['True_Label'] == 1)).sum()
    FP = ((df['prediction_label'] == 1) & (df['True_Label'] == 0)).sum()
    TN = ((df['prediction_label'] == 0) & (df['True_Label'] == 0)).sum()
    FN = ((df['prediction_label'] == 0) & (df['True_Label'] == 1)).sum()
    
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensitivity / Recall
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # Fall-out
    
    roc_data.append([threshold, TP, FP, TN, FN, TPR, FPR])

# Convert to DataFrame
roc_df = pd.DataFrame(roc_data, columns=['Threshold', 'TP', 'FP', 'TN', 'FN', 'TPR', 'FPR'])

# Plot ROC Curve
plt.figure(figsize=(7, 5))
plt.plot(roc_df['FPR'], roc_df['TPR'], marker='o', linestyle='-', color='blue', label='ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing')  # Diagonal reference line
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Display results
from IPython.display import display
display(roc_df)


#Write code (with AI assistant) to fit the model using your favorite classifier (NB, KNN, or Decision tree); using the hamspam.csv, ask to output an ROC curve and AUC score. (Hint: if you fit a decision tree, you might want to reduce max_depth) 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

# Load dataset
file_path = r"C:\Users\shris\OneDrive\Desktop\homework\hamspam.csv.csv"  # Update the file path if needed
df = pd.read_csv(file_path)

# Display first few rows
print("Dataset loaded successfully!")
print(df.head())

# Convert categorical values into numerical format
df['Contains Link'] = df['Contains Link'].map({'Yes': 1, 'No': 0})
df['Contains Money Words'] = df['Contains Money Words'].map({'Yes': 1, 'No': 0})
df['Length'] = df['Length'].map({'Long': 1, 'Short': 0})
df['Class'] = df['Class'].map({'Spam': 1, 'Ham': 0})  # Convert labels to binary format

# Features (X) and Target (y)
X = df[['Contains Link', 'Contains Money Words', 'Length']]  # Use actual features
y = df['Class']  # Target variable

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose Classifier: Naïve Bayes, KNN, or Decision Tree
classifier = GaussianNB()  # Change to KNeighborsClassifier() or DecisionTreeClassifier(max_depth=5) as needed
classifier.fit(X_train, y_train)

# Predict probabilities for ROC curve
y_pred_prob = classifier.predict_proba(X_test)[:, 1]
y_pred = classifier.predict(X_test)

# Evaluate Model
print("\n📌 Classifier Performance:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# ROC Curve and AUC Score
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
