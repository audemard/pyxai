name='balance-scale_0'

from pyxai import Learning, Explainer, Tools

# Machine learning part
learner = Learning.Scikitlearn(name+'.csv', learner_type=Learning.CLASSIFICATION)

model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT)
instance, prediction = learner.get_instances(model, n=1, correct=True)

# Explanation part
explainer = Explainer.initialize(model, instance, features_type= name+'.types')
# explainer.add_clause_to_theory([1, -2])
# explainer.add_clause_to_theory([2, -3])
explainer.add_clause_to_theory([-3, -5])
print(explainer.get_theory())
print("instance:", instance)
print("binary: ", explainer.binary_representation)
reason = explainer.minimal_majoritary_reason()
print("reason: ", reason)
print("tofeature",explainer.to_features(reason))
print("is reason", explainer.is_reason(reason))