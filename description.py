import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, accuracy_score, classification_report, matthews_corrcoef

from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice

from fcapy.visualizer import LineVizNx
import matplotlib.pyplot as plt
import networkx as nx
import neural_lib as nl
import seaborn as sns


plt.rcParams['figure.facecolor'] = (1,1,1,1)

df = pd.read_csv('./Car_Insurance_Claim.csv')
df = df.drop(columns=['ID', 'CREDIT_SCORE', 'POSTAL_CODE', 'ANNUAL_MILEAGE', 'SPEEDING_VIOLATIONS', 'DUIS', 'PAST_ACCIDENTS'])

df = df.drop(columns=[
    'VEHICLE_OWNERSHIP',
    'MARRIED',
    'CHILDREN',
    'VEHICLE_YEAR'
])


# print(df.columns)

one_hot_encoded_df = pd.get_dummies(df.drop(columns=['OUTCOME']))
one_hot_encoded_df = one_hot_encoded_df.replace({0.0: False, 1.0: True})
# one_hot_encoded_df[['not_VEHICLE_OWNERSHIP', 'not_MARRIED', 'not_CHILDREN']] = one_hot_encoded_df[['VEHICLE_OWNERSHIP', 'MARRIED', 'CHILDREN']]
one_hot_encoded_df['OUTCOME'] = df['OUTCOME'].astype(int)
one_hot_encoded_df.index = one_hot_encoded_df.index.astype(str)
# print(one_hot_encoded_df.columns)

df = one_hot_encoded_df.copy()

df['<25'] = df['AGE_16-25'] | df['AGE_26-39'] | df['AGE_40-64'] | df['AGE_65+']
df['<39'] = df['AGE_26-39'] | df['AGE_40-64'] | df['AGE_65+']
df['<64'] = df['AGE_40-64'] | df['AGE_65+']
df['>65'] = df['AGE_65+']

df = df.drop(columns=['AGE_16-25', 'AGE_26-39', 'AGE_40-64', 'AGE_65+'])

# df['DRIVING_EXPERIENCE_<9y'] = (
#     df['DRIVING_EXPERIENCE_0-9y']
#     | df['DRIVING_EXPERIENCE_10-19y']
#     | df['DRIVING_EXPERIENCE_20-29y']
#     | df['DRIVING_EXPERIENCE_30y+']
# )
# df['DRIVING_EXPERIENCE_<19y'] = (
#     df['DRIVING_EXPERIENCE_10-19y']
#     | df['DRIVING_EXPERIENCE_20-29y']
#     | df['DRIVING_EXPERIENCE_30y+']
# )
# df['DRIVING_EXPERIENCE_<29y'] = (
#     df['DRIVING_EXPERIENCE_20-29y']
#     | df['DRIVING_EXPERIENCE_30y+']
# )

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
    'GENDER_female': 'female'
}

df.rename(columns=map, inplace=True)

df = df.drop(columns=['DRIVING_EXPERIENCE_0-9y', 'DRIVING_EXPERIENCE_10-19y',
       'DRIVING_EXPERIENCE_20-29y', 'DRIVING_EXPERIENCE_30y+'])

# Create a barplot for each feature
num_features = len(df.columns)
fig, axes = plt.subplots(nrows=num_features, figsize=(10, 5 * num_features), sharex=False)

# Loop through each feature to plot its distribution
for i, column in enumerate(df.columns):
    # Count the unique values for categorical or discretized numeric data
    value_counts = df[column].value_counts()

    # Plot
    sns.barplot(
        x=value_counts.index,
        y=value_counts.values,
        ax=axes[i] if num_features > 1 else axes
    )
    axes[i].set_title(f'Distribution of {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Frequency')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()