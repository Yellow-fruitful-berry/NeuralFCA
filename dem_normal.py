import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score, classification_report, matthews_corrcoef

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier



plt.rcParams['figure.facecolor'] = (1,1,1,1)

df = pd.read_csv('Car_Insurance_Claim.csv')

df = df.drop(columns=['ID', 'CREDIT_SCORE', 'POSTAL_CODE', 'ANNUAL_MILEAGE', 'SPEEDING_VIOLATIONS', 'DUIS', 'PAST_ACCIDENTS'])

df = df.drop(columns=[
    'VEHICLE_YEAR',
    'VEHICLE_TYPE',
    'VEHICLE_OWNERSHIP',
    'EDUCATION',
    'INCOME'
])

one_hot_encoded_df = pd.get_dummies(df.drop(columns=['OUTCOME']))
one_hot_encoded_df = one_hot_encoded_df.replace({0.0: False, 1.0: True})
one_hot_encoded_df['not_MARRIED'] = ~one_hot_encoded_df['MARRIED']
one_hot_encoded_df['not_CHILDREN'] = ~one_hot_encoded_df['CHILDREN']
one_hot_encoded_df['OUTCOME'] = df['OUTCOME'].astype(int)
one_hot_encoded_df.index = one_hot_encoded_df.index.astype(str)

df = one_hot_encoded_df.copy()
df['<25'] = df['AGE_16-25'] | df['AGE_26-39'] | df['AGE_40-64'] | df['AGE_65+']
df['<39'] = df['AGE_26-39'] | df['AGE_40-64'] | df['AGE_65+']
df['<64'] = df['AGE_40-64'] | df['AGE_65+']
df['>65'] = df['AGE_65+']

map = {
    'RACE_majority': 'maj',
    'RACE_minority': 'min',
    'EDUCATION_none': 'no_ed',
    'EDUCATION_university': 'uni',
    'EDUCATION_high school': 'high',
    'INCOME_poverty': 'pov',
    'INCOME_upper class': 'up',
    'INCOME_working class': 'work',
    'INCOME_middle class': 'mid',
    'VEHICLE_TYPE_sedan': 'sedan',
    'VEHICLE_TYPE_sports car': 'sport',
    'GENDER_male': 'male',
    'GENDER_female': 'female',
    'VEHICLE_OWNERSHIP': 'veh',
    'not_VEHICLE_OWNERSHIP': 'no_veh',
    'MARRIED': 'mar',
    'not_MARRIED': 'nmar',
    'CHILDREN': 'cld',
    'not_CHILDREN': 'ncld'
}

df.rename(columns=map, inplace=True)

df = df.drop(columns=['DRIVING_EXPERIENCE_0-9y', 'DRIVING_EXPERIENCE_10-19y',
       'DRIVING_EXPERIENCE_20-29y', 'DRIVING_EXPERIENCE_30y+', 'AGE_16-25', 'AGE_26-39', 'AGE_40-64', 'AGE_65+'])


one_hot_encoded_df = df.copy()



X = df.drop(['OUTCOME'], axis=1)
y = df['OUTCOME']

print(df.columns)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
# Initialize the classifiers
hist_clf = HistGradientBoostingClassifier(
    l2_regularization=10,
    learning_rate=0.1,
    max_depth=5,
    max_iter=100,
    random_state=42,
    # class_weight='balanced',
)

rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    class_weight='balanced',
    random_state=42
)

# Create a voting classifier
voting_clf = VotingClassifier(estimators=[
    ('hgb', hist_clf),
    ('rf', rf_clf),
], voting='soft')

# Define the parameter grid for GridSearchCV
param_grid = {
    'hgb__max_iter': [100],
    'hgb__l2_regularization': [10],
    'hgb__max_depth': [5],
    'hgb__learning_rate': [0.1],
    'rf__n_estimators': [200],
    'rf__max_depth': [5],
    'rf__max_features': [None],
}

# Create StratifiedKFold object
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create GridSearchCV object with StratifiedKFold
grid_search = GridSearchCV(voting_clf, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)

# Fit the grid search model
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Print the best parameters and test f1 score
print("Best parameters:", grid_search.best_params_)
print("Test F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("Test accuracy Score:", accuracy_score(y_test, y_pred))
print("Test MCC:", matthews_corrcoef(y_test, y_pred))

# sns.heatmap(df.corr())
# plt.show()

acc = []
mcc = []
f1 = []

for k in range(3,10):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    # Evaluate the classifier
    acc.append(accuracy_score(y_test, y_pred))
    f1.append(f1_score(y_test, y_pred, average='weighted'))
    mcc.append(matthews_corrcoef(y_test, y_pred))



print(acc)
print(f1)
print(mcc)


knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the classifier
acc.append(accuracy_score(y_test, y_pred))
f1.append(f1_score(y_test, y_pred, average='weighted'))
mcc.append(matthews_corrcoef(y_test, y_pred))

print('KNN')
print(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'), matthews_corrcoef(y_test, y_pred))


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

new_feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
X.columns = new_feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

print("Best parameters:", grid_search.best_params_)
print("Test F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("Test accuracy Score:", accuracy_score(y_test, y_pred))
print("Test MCC:", matthews_corrcoef(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))