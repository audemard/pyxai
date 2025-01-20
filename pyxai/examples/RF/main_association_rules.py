import json
from pyxai import Learning, Explainer, Tools
import pandas as pd
from pysat.solvers import Glucose3
from sklearn.model_selection import train_test_split
import apriori_association_rules
import time
##################################################################################################################################
# I load the dataset

n_max_rules = int(Tools.Options.types)
print(n_max_rules)
name=Tools.Options.dataset 
data = pd.read_csv(name+'.csv')
# Split the DataFrame into training and test sets
train_df, validation_df = train_test_split(data, test_size=0.3, random_state=42)
glucose = Glucose3()

print("Divide dataset")
#I train my model on the training set.
rf_learner = Learning.Scikitlearn(train_df, learner_type=Learning.CLASSIFICATION)
rf_model = rf_learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF,seed=1)
instance, prediction = rf_learner.get_instances(rf_model, n=1)
rf_explainer = Explainer.initialize(rf_model, instance, features_type= name+'.types')
#I binarize my dataset.
nb_features=len(rf_explainer.binary_representation)
binarized = []
raw_validation       = []
label_validation     = []
binarized_validation = []
for i, instance in enumerate(rf_learner.data):
    rf_explainer.set_instance(instance)
    binarized.append([0 if l < 0 else 1 for l in rf_explainer.binary_representation] +  [rf_learner.labels[i]])
training_data=pd.DataFrame(binarized, columns=[f"X_{i}" for i in range(1, nb_features + 1)] + ['y'])
for i, instance in validation_df.iterrows():
    rf_explainer.set_instance(instance[:-1])
    raw_validation.append(instance[:-1])
    label_validation.append(instance[-1])
    binarized_validation.append([0 if l < 0 else 1 for l in rf_explainer.binary_representation] +  [instance[-1]])
#I add the negation of the features to extract negative association rules.
#for i in range(1,training_data.shape[1]):
#    training_data[f'N_{i}']=training_data[f'X_{i}'].apply(lambda x: 1 if x == 0 else (0 if x == 1 else x))
#training_data['yy'] = training_data['y'].apply(lambda x: 1 if x == 0 else (0 if x == 1 else x))
#Displaying the final DataFrame.
print(training_data)

#Apriori
#######################################################################################################################################
#I use Apriori to extract association rules without the class variables.
#df_filtered = training_data.drop(columns=['y', 'yy'])
df_filtered = training_data.drop(columns=['y'])
min_support = 0.5
min_confidence = 1
max_length=3
print("Start madelaine ...")
madelaine_time, rules = apriori_association_rules.madelaine(df_filtered, time_limit=3600, n_max_rules=n_max_rules, explainer=rf_explainer)
print("End madelaine time: ", madelaine_time)

# Display the number of rules generated.
print(f"Nombre de règles: {len(rules)}")
print("###########################################")
theory_association_rules=apriori_association_rules.rules_to_clauses(rules)
theory_initial = rf_explainer.get_theory()

print('len theory_initial: ',len(theory_initial))
print("len theory_association_rules: ",len(theory_association_rules))
#theory_not_subsumed = apriori_association_rules.remove_subsumed(theory_association_rules)
#print("len theory_not_subsumed: ",len(theory_not_subsumed))

############################################################################################################"
# Choose 100 instances on which we will extract majority explanations and see the number of excluded instances.
print("Chooses instances")
good_instances = []
glucose = Glucose3()
nb_instances = 10
nb_instances_excluded = 0
for i in theory_association_rules:
    glucose.add_clause(i)
for id_instance,instance_dict in enumerate(binarized_validation):
    instance_dt = instance_dict[:-1]
    label_dt = instance_dict[-1]
    rf_explainer.set_instance(raw_validation[id_instance])
    if glucose.propagate(rf_explainer.binary_representation)[0] is False:
        nb_instances_excluded += 1
        continue
    good_instances.append(raw_validation[id_instance])
    #print(raw_validation[id_instance])
    if len(good_instances) >= nb_instances:
        break
len_reason=0
treasean=[]
nb_is_not_reason=0
majoritary_literal_reason1=[]
majoritary_feature_reason1=[]
elapsed_time_majoritary_reason1 = []
############################################################################################################"
#We extract the majority explanations on the instances chosen before adding the second theory.
print("Compute majoritary normal")
for i in good_instances:
    start_time = time.time()
    rf_explainer.set_instance(i)
    reason = rf_explainer.majoritary_reason(n_iterations=100,seed=1)
    majoritary_literal_reason1.append(len(reason))
    majoritary_feature_reason1.append(len(rf_explainer.to_features(reason)))
    if not(rf_explainer.is_majoritary_reason(reason)):
        nb_is_not_reason+=1
    treasean.append(len(reason))
    len_reason+=len(reason)
    end_time = time.time()
    print("time: ", time.time() -start_time, "classic reason: ", len(reason))
    elapsed_time_majoritary_reason1.append(end_time - start_time)

moreasen1=len_reason/len(good_instances)
#We add the second theory to our explainer
# rf_explainer = Explainer.initialize(rf_model)
for clause in theory_association_rules:
    rf_explainer.add_clause_to_theory(clause)

len_reason_theorie2=0
nb_is_not_reason2=0
treasean1=[]
majoritary_literal_reason2=[]
majoritary_feature_reason2=[]
elapsed_time_majoritary_reason2 = []

print("Compute Majoritary with additional theory")
# We extract the majority explanations on the instances selected after adding the second theory.
for i in good_instances:
    start_time = time.time()
    rf_explainer.set_instance(i)
    reason1 = rf_explainer.majoritary_reason(n_iterations=100,seed=1)
    majoritary_literal_reason2.append(len(reason1))
    majoritary_feature_reason2.append(len(rf_explainer.to_features(reason1)))
    if not(rf_explainer.is_majoritary_reason(reason1)):
        nb_is_not_reason2+=1
    treasean1.append(len(reason1))
    len_reason_theorie2+=len(reason1)
    print("time: ", time.time() -start_time, "additional theory reason: ", len(reason1), rf_explainer.to_features(reason1))
    end_time = time.time()
    elapsed_time_majoritary_reason2.append(end_time - start_time)

moreasen2=len_reason_theorie2/len(good_instances)


print("Compute stats")
#Traverse both lists simultaneously and compare the sizes of the explanations before and after adding the apriori theory.
count_inf = 0
count_sup = 0
count_eq = 0
for t, t1 in zip(treasean, treasean1):
    if t1 < t:
        count_inf += 1
    elif t1 > t:
        count_sup += 1
    else:  # t1 == t
        count_eq += 1

############################################################################################################"
#Recording the extracted logs.
data_ = {
    "dataset_name": name,
    "theory_initial": len(theory_initial),
    "theory_association_rules": len(theory_association_rules),
    "confidence":min_confidence,
    "support":min_support,
    "nb_instances_excluded":nb_instances_excluded,
    "size_majority_reason_literal_1": majoritary_literal_reason1,
    "size_majority_reason_literal_2": majoritary_literal_reason2,
    "size_majority_reason_feature_1": majoritary_feature_reason1,
    "size_majority_reason_feature_2": majoritary_feature_reason2,
    "number_of_reasons_reduced_after_adding_theory": count_inf,
    "Number_of_reasons_increased_after_adding_theory": count_sup,
    "Number_of_equal_reasons_after_adding_theory": count_eq,
    "Number_of_is_not_reason_before_adding_new_clauses": nb_is_not_reason,
    "Number_of_is_not_reason_after_adding_new_clauses ": nb_is_not_reason2,
    "elapsed_time_aprioris":madelaine_time,
    "elapsed_time_majoritary_reason1":elapsed_time_majoritary_reason1,
    "elapsed_time_majoritary_reason2":elapsed_time_majoritary_reason2,

}

print("###########################################")
print(data_)
# Writing to a JSON file.
file_name = name
with open(file_name + ".json", "w") as file_json:
    json.dump(data_, file_json, indent=4)


