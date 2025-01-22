#fonctions utilisés
import apriori_classement_rules
from pyxai import Learning, Explainer, Tools ,Builder
from sklearn.model_selection import train_test_split
import pandas as pd
from pysat.solvers import Glucose3
#import matplotlib
#matplotlib.use('Qt5Agg')  # Choisit le backend Qt5
import matplotlib.pyplot as plt
import time
import json
import os
##############################################################################################################
Tools.set_verbose(0)
glucose = Glucose3()
name=Tools.Options.dataset
n_max_rules = int(Tools.Options.types)
print("n_max_rules:", n_max_rules)

data = pd.read_csv(name+'.csv')
name=name
# Dividing the DataFrame into training, testing, and validation sets
train_df, validation_df = train_test_split(data, test_size=0.3, random_state=42)
# Save the DataFrames to CSV files
train_df.to_csv('train_data.csv', index=False)
#best_parameters = DecisionNode.tuning('train_data.csv')
# for i in range(2):
rf_learner = Learning.Scikitlearn(train_df, learner_type=Learning.CLASSIFICATION) # 70%
# I create a xgboost model: the expert
print("##################################################################")
rf_model1 = rf_learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF,seed=1)
instance, prediction = rf_learner.get_instances(rf_model1, n=1)
# I need an explainer BT
rf_explainer = Explainer.initialize(rf_model1, instance, features_type= name+'.types')
# I need the theory.... Currently, we collect clauses related to a binarised instance
clauses=[]
# I need to collect the theory related to boolean variables....
for clause in rf_model1.get_theory(rf_explainer.binary_representation):
    glucose.add_clause(clause)
    negated_clause = [[-clause[0]],[clause[1]]]
    clauses.append(negated_clause)

binarized_training   = []
raw_validation       = []
label_validation     = []
binarized_validation = []
nb_features = len(rf_explainer.binary_representation)  # nb binarized features
# Iterating through the training set to binarize it
for i, instance in enumerate(rf_learner.data):
    rf_explainer.set_instance(instance)
    binarized_training.append([0 if l < 0 else 1 for l in rf_explainer.binary_representation] +  [rf_learner.labels[i]])
training_data=pd.DataFrame(binarized_training, columns=[f"X_{i}" for i in range(1, nb_features + 1)] + ['y'])

# Iterating through the validation set to binarize it
for i, instance in validation_df.iterrows():
    rf_explainer.set_instance(instance[:-1])
    raw_validation.append(instance[:-1])
    label_validation.append(int(instance[-1]))
    binarized_validation.append([0 if l < 0 else 1 for l in rf_explainer.binary_representation] +  [instance[-1]])
labels=(rf_learner.labels_to_values(label_validation))
##############################################################################################################

rf_learner1 = Learning.Scikitlearn(training_data, learner_type=Learning.CLASSIFICATION)
#I add the negation of the features to extract negative association rules.
# We create K fold cross validation models
rf_models = rf_learner1.evaluate(method=Learning.K_FOLDS, output=Learning.RF,seed=0,n_estimators=4)
#print("2")
feature_names=rf_learner1.get_details()[0]['feature_names']
all_scikit = rf_learner1.get_raw_models()
# Declare the lists and dictionaries that we will use
number_of_nodes_random_forest=[]
precision_random_forest_before_correction_on_the_validation_set=[]
X_test=[]
y_test=[]
random_forest_depth=[]
precision_for_each_random_forest=[]
#print("3")
# Iterating through the 10 decision tree models created with PyXAI
yo=[]
for i, rf_model in enumerate(rf_models) :
    yo.append(rf_model)
    #print("dj")
    # I take scikitLearn model
    clf = all_scikit[i]
    total_nodes = sum(tree.tree_.node_count for tree in clf.estimators_)
    # Calculer la profondeur moyenne des arbres
    depth_for_all_trees = [tree.tree_.max_depth for tree in clf.estimators_]
    average_depth = sum(depth_for_all_trees) / len(depth_for_all_trees)
    random_forest_depth.append(average_depth)
    number_of_nodes_random_forest.append(total_nodes)
    # I collect  all instances from training set
    instances_dt_training = rf_learner1.get_instances(rf_model, n=None, indexes=Learning.TRAINING, details=True)
    # I collect  all instances from test set
    instances_dt_test= rf_learner1.get_instances(rf_model, n=None, indexes=Learning.TEST, details=True)
    X_train1=[]
    y_train1=[]
    X_1=[]
    y_1=[]
    ert = Explainer.initialize(rf_model)
    # Store instances and their labels in the lists X1 and Y1 of the test set
    for instance_dict in instances_dt_test:
        instance_dt = instance_dict["instance"]
        label_dt = instance_dict["label"]
        ert.set_instance(instance_dt)
        X_1.append(instance_dt)
        y_1.append(label_dt)
    # Store instances and their labels in the lists X_train1 and y_train1 of the training set that we will use for retraining
    for instance_dicti in instances_dt_training:
        instance_dt1 = instance_dicti["instance"]
        label_dt = instance_dicti["label"]
        X_train1.append(instance_dt1)
        y_train1.append(label_dt)
# Calculate the accuracy of each decision tree
    X_test1 = []
    y_test1 = []
    # Store instances and their labels in the lists X_test1 and y_test1 of the validation set
    for id_instance,instance_dict in enumerate(binarized_validation):
        instance_dt = instance_dict[:-1]
        label_dt = labels[id_instance]
        X_test1.append(instance_dt)
        ert.set_instance(instance_dt)
        y_test1.append(label_dt)
    X_test.append(X_test1) # Store all instances from all decision trees
    y_test.append(y_test1)
    locf0=apriori_classement_rules.precision(rf_model, X_test1, y_test1)
    locf = clf.score(X_1, y_1)
    precision_random_forest_before_correction_on_the_validation_set.append(locf0)
    precision_for_each_random_forest.append(locf)

#apriori
##############################################################################################################

start_time = time.time()
len_rules_2, len_rules_3, len_rules_total, madelaine_time, rules = apriori_classement_rules.madelaine(training_data, time_limit=20, n_max_rules=n_max_rules, explainer=rf_explainer)
end_time = time.time()

print("len_rules_2:", len_rules_2)
print("len_rules_3:", len_rules_3)
print("len_rules_total:", len_rules_total)

elapsed_time_aprioris = (end_time - start_time)
association_dict = {}
antecedents=[]
consequents=[]
for antecedent, consequent in rules:
    association_dict[antecedent] = consequent
association_dict_copy = dict(association_dict)
new_association_dict = dict(association_dict)
#print(new_association_dict)
nb_rules=[]
#simplification
# Parcours du dictionnaire
#for antecedent, consequent in association_dict.items():
#    for clause in clauses:
#        if clause[0][0] in list((antecedent)) and clause[1][0] in list(antecedent):
            #print("antecedent",antecedent)
            #print("a supprimer",clause[1][0])
            #print(new_association_dict[antecedent])
#            new_association_dict = apriori_classement_rules.remove_element_from_key(new_association_dict, antecedent, clause[1][0])
#print("nombre de régles extraire",len(association_dict_copy))
#nb_rules.append(len(association_dict_copy))
#print("nombre de regles restante aprés simplification avec theorie",len(new_association_dict))

#généralisation
# keys_to_delete = []  # Liste pour stocker les clés à supprimer

# for key1 in new_association_dict:
#     for key2 in new_association_dict:
#         if key1 != key2:
#             if apriori_classement_rules.generelise((key1), (key2),new_association_dict[key1],new_association_dict[key2]):
#                 if key2 not in keys_to_delete:
#                     keys_to_delete.append(key2)

# Supprimer les clés genéralisé du dictionnaire
# for key in keys_to_delete:
#     del new_association_dict[key]
# Afficher les règles d'association de classement
# print("nb de regles apres généralisation:",len(new_association_dict))
##############################################################################################################
nb_rules.append(len(new_association_dict))
class_association_dict = {}
class_association_dict0={}
y=nb_features+1
for antecedent, consequent in new_association_dict.items():
    if (y == consequent):
        class_association_dict[antecedent] = consequent
    if (-y == consequent):
        class_association_dict0[antecedent] = consequent

print("nombre regles de classement:",len(class_association_dict)+len(class_association_dict0))
tuple_of_tuples = [(tuple(key), 1) for key in class_association_dict.keys()]
tuple_of_tuples0 = [(tuple(key), 0) for key in class_association_dict0.keys()]
##############################################################################################################

precision_random_forest_after_rectification_for_all_random_forest=[]
number_of_nodes_for_all_trees=[]
depth_rectification_for_each_rule=[]
time_rectification_for_each_rule=[]
for b in range(len(yo)):
    tree_rectified=yo[b]
    rf_model=yo[b]
    ert=Explainer.initialize(rf_model)
    ths=rf_model1.get_theory(rf_explainer.binary_representation)
    theorie=apriori_classement_rules.trasforme_list_tuple_to_binaire(ths,rf_model)
    theorie_clause=ert.condi(conditions=theorie)
    theorie_clause=apriori_classement_rules.list_to_tuple_pairs(theorie_clause)
    # # class_association_dict = {frozenset({1}): frozenset({'y'}), frozenset({4,2}): frozenset({1})}
    precisions=[precision_random_forest_before_correction_on_the_validation_set[b]]
    number_of_nodes=[number_of_nodes_random_forest[b]]
    depth_rectification_=[random_forest_depth[b]]
    time_unwind=[]
    eft=Explainer.initialize(rf_model)
    for j in tuple_of_tuples:
        conditions=apriori_classement_rules.trasforme_tuple_to_binaire(j[0],rf_model)
        start_time = time.time()
        rf_model = eft.rectify(conditions=conditions, label=1, tests=False,theory_cnf=theorie_clause)
        end_time = time.time()
        elapsed_time = (end_time - start_time)
        precision_tree_rectified=apriori_classement_rules.precision(rf_model, X_test[b], y_test[b])
        total_node=rf_model.n_nodes()
        random_forest_depth_rectification__=rf_model.depth()
        number_of_nodes.append(total_node)
        depth_rectification_.append(random_forest_depth_rectification__)
        precisions.append(precision_tree_rectified)
        time_unwind.append(elapsed_time)
    for i in tuple_of_tuples0:
        conditions=apriori_classement_rules.trasforme_tuple_to_binaire(i[0],rf_model)
        start_time = time.time()
        rf_model = eft.rectify(conditions=conditions, label=0, tests=False,theory_cnf=theorie_clause)
        end_time = time.time()
        elapsed_time = (end_time - start_time)
        precision_tree_rectified=apriori_classement_rules.precision(rf_model, X_test[b], y_test[b])
        total_node=rf_model.n_nodes()
        random_forest_depth_rectification__=rf_model.depth()
        number_of_nodes.append(total_node)
        depth_rectification_.append(random_forest_depth_rectification__)
        precisions.append(precision_tree_rectified)
        time_unwind.append(elapsed_time)
    precision_random_forest_after_rectification_for_all_random_forest.append(precisions)
    number_of_nodes_for_all_trees.append(number_of_nodes)
    depth_rectification_for_each_rule.append(depth_rectification_)
    time_rectification_for_each_rule.append(time_unwind)

data_ = {
    "dataset name":name,
    "n_instances":len(data),
    "n_instances_training":len(train_df),
    "n_instances_validation":len(validation_df),
    "n_max_rules": n_max_rules,
    "len_rules_2:": len_rules_2,
    "len_rules_3:": len_rules_3,
    "len_rules_total:": len_rules_total,    
    "accuracy_of_the_random_forest_on_test_set":rf_learner.get_details()[0]["metrics"]["accuracy"],
    "columns_of_the_non_binarized_data_set":rf_learner.get_details()[0]['feature_names'],
    "columns_of_the_binarized_data_set":rf_learner1.get_details()[0]['feature_names'],
    "accuracy_for_each_random_forest_on_test_set":precision_for_each_random_forest,
    "accuracy_for_each_decision_random_forest_before_correction_on_validation_set_relative_to_random_forest":precision_random_forest_before_correction_on_the_validation_set,
    "number_of_nodes_for_each_random_forest_before_correction":number_of_nodes_random_forest,
    "accuracy_after_rectification_for_each_rule": precision_random_forest_after_rectification_for_all_random_forest,
    "number_of_nodes_after_rectification_for_each_rule": number_of_nodes_for_all_trees,
    "depth_after_rectification_for_each_rule": depth_rectification_for_each_rule,
    "time_rectification_for_each_rule":time_rectification_for_each_rule,
    "apriori_classement_0": tuple_of_tuples0,
    "apriori_classement_1": tuple_of_tuples,
    "nb_rules": nb_rules,
    "elapsed_time_aprioris": elapsed_time_aprioris
}
# Writing the data to the JSON file
with open(name + ".json", 'w') as file_json:
    json.dump(data_, file_json)

print("accuracy_after_rectification_for_each_rule",precision_random_forest_after_rectification_for_all_random_forest)