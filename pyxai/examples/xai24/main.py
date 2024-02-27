from pyxai import Learning, Explainer, Tools
import cases
import user
import constants
import misc



# Create the user agent
learner_user = Learning.Xgboost(Tools.Options.dataset, learner_type=Learning.CLASSIFICATION)
model_user = learner_user.evaluate(method=Learning.HOLD_OUT, output=Learning.BT, test_size=1 - constants.training_size, seed=123)
instances = learner_user.get_instances(model_user, indexes=Learning.TEST, details=True)

# Change weights of BT
if constants.debug:
    print("Accuracy before", misc.get_accuracy(model_user, instances[0:100]))
misc.change_weights(model_user)
if constants.debug:
    print("Accuracy after ", misc.get_accuracy(model_user, instances[0:100]))


# Extract test instances and classified instances
threshold = int(len(instances) * constants.classified_size)
classified_instances = instances[0:threshold]
test_instances = instances[threshold:-1]
positive_instances, negative_instances, unclassified_instances = misc.partition_instances(model_user, classified_instances)

if constants.trace:
    print("nb positives:", len(positive_instances))
    print("nb negatives:", len(negative_instances))
    print("nb unclassified:", len(unclassified_instances))

# Create user
#explainer_user = Explainer.initialize(model_user, features_type=Tools.Options.types)


# create AI
learner_AI = Learning.Scikitlearn(Tools.Options.dataset, learner_type=Learning.CLASSIFICATION)
model_AI = learner_AI.evaluate(method=Learning.HOLD_OUT, output=Learning.RF, test_size=1 - constants.training_size, seed=123) # The same seed


# Create the global theory, enlarge AI in consequence change the representation for user
# Keep the same representation in AI but, increase the binary representation
#model_user => BT
#model_AI => RF
model_user, model_AI = misc.create_binary_representation(model_user, model_AI)


# Create the explainers
explainer_user = Explainer.initialize(model_user, features_type=Tools.Options.types)
explainer_AI = Explainer.initialize(model_AI, features_type=Tools.Options.types)

if constants.debug:
    explainer_user.set_instance(positive_instances[0])
    explainer_AI.set_instance(positive_instances[0])
    assert explainer_user._binary_representation == explainer_AI._binary_representation, "Big problem :)"


user = user.User(explainer_user, positive_instances, negative_instances)

if constants.trace:
    print("nb positive rules", len(user.positive_rules))
    print("nb negative rules", len(user.negative_rules))
print("WARNING: what about N")


# Iterate on all classified instances
nb_cases_1 = 0
nb_cases_2 = 0
nb_cases_3 = 0
nb_cases_4 = 0
nb_cases_5 = 0


for detailed_instance in classified_instances:
    instance = detailed_instance['instance']
    explainer_AI.set_instance(instance)
    prediction_AI = model_AI.predict_instance(instance)
    prediction_user = user.predict_instance(explainer_AI.binary_representation)  # no they have the same representation
    rule_AI = explainer_AI.majoritary_reason(n_iterations=1)
    # All cases
    print(prediction_user)
    if prediction_user is None:  # cases (3) (4) (5)
        match cases.cases_3_4_5(explainer_AI, rule_AI, user):
            case 3:
                nb_cases_3 += 1
            case 4:
                nb_cases_4 += 1
            case 5:
                nb_cases_5 += 1

    else:
        if prediction_AI != prediction_user:  # case (1)
            cases.case_1(explainer_AI, rule_AI, user)
            nb_cases_1 += 1
        if prediction_AI == prediction_user:  # case (2)
            cases.case_2(explainer_AI, rule_AI, user)
            nb_cases_2 += 1
    print(nb_cases_1, nb_cases_2, nb_cases_3)

# je demande à l'ia
# une des 5 cas arrive et je modifie l'ia en conséquence (et aussi le U)

# on evalue par paquet de 10
# on evalue sur le test set

# - accuracy de IA
# - couverture : combien d'instances U est il capable de classer
































