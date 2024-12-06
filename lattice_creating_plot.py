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

one_hot_encoded_df = df.copy()

y_feat = 'OUTCOME'
df_train, df_test = train_test_split(one_hot_encoded_df, train_size=0.7, random_state=0)

X_train, y_train = df_train.drop(columns=['OUTCOME']), df_train[y_feat]
X_test, y_test = df_test.drop(columns=['OUTCOME']), df_test[y_feat]

K_train = FormalContext.from_pandas(X_train)

L = ConceptLattice.from_context(K_train,algo='Sofia', is_monotone=True)
print("Amount of concepts in lattice: ", len(L))

for c in L:
    y_preds = np.zeros(K_train.n_objects)
    y_preds[list(c.extent_i)] = 1
    c.measures['f1_score'] = f1_score(y_train, y_preds)
    c.measures['recall_score'] = recall_score(y_train, y_preds)

n_concepts = 3
best_concepts = list(L.measures['f1_score'].argsort()[::-1][:n_concepts])
# assert len({g_i for c in L[best_concepts] for g_i in c.extent_i})==K_train.n_objects, "Selected concepts do not cover all train objects"

cn = nl.ConceptNetwork.from_lattice(L, best_concepts, sorted(set(y_train)))
vis = LineVizNx(node_label_font_size=14, node_label_func=lambda el_i, P: nl.neuron_label_func(el_i, P, set(cn.attributes))+'\n\n')

descr = {''}

traced = cn.trace_description(descr, include_targets=False)

fig, ax = plt.subplots(figsize=(20,5))

vis.draw_poset(
    cn.poset, ax=ax,
    flg_node_indices=False,
    node_label_func=lambda el_i, P: nl.neuron_label_func(el_i, P, set(cn.attributes), only_new_attrs=True)+'\n\n',
    node_color=['darkblue' if el_i in traced else 'lightgray' for el_i in range(len(cn.poset))
                ],
               node_size=100,
               node_label_font_size=6
)

# plt.subplots_adjust()
# plt.tight_layout()
# plt.savefig('nn_g_b_a.png')
# plt.show()

cn = nl.ConceptNetwork.from_lattice(L, best_concepts, sorted(set(y_train)))
cn.fit(X_train, y_train,  n_epochs = 10000)

y_pred = cn.predict(X_test).numpy()
print('Class prediction', y_pred[:10])
y_proba = cn.predict_proba(X_test).detach().numpy()
print('Class prediction with probabilities', y_proba[:10])
print('True class', y_test.values[:10])


print("TEST SCORES")
print('Recall score:', recall_score(y_test.values.astype('int'), y_pred))
print('F1     score:', f1_score(y_test.values.astype('int'), y_pred))
print('Accuracy score:', accuracy_score(y_test.values.astype('int'), y_pred))
print("Test MCC:", matthews_corrcoef(y_test.astype('int'), y_pred))

print(classification_report(y_test.values.astype('int'), y_pred))




edge_weights = cn.edge_weights_from_network()

fig, ax = plt.subplots(figsize=(15,5))

vis.draw_poset(
    cn.poset, ax=ax,
    flg_node_indices=False,
    node_label_func=lambda el_i, P: nl.neuron_label_func(el_i, P, set(cn.attributes), only_new_attrs=True)+'\n\n',
    edge_color=[edge_weights[edge] for edge in cn.poset.to_networkx().edges],
    edge_cmap=plt.cm.RdBu,
)
nx.draw_networkx_edge_labels(cn.poset.to_networkx(), vis.mover.pos, {k: f"{v:.1f}" for k,v in edge_weights.items()}, label_pos=0.7)

# plt.title('Neural network with fitted edge weights', size=24, x=0.05, loc='left')
# plt.tight_layout()
# plt.subplots_adjust()
# # plt.savefig('fitted_network_silly_baseline.png')
plt.show()

print(edge_weights)