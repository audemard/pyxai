# Check V1.0: Ok
from pyxai import Learning, Explainer, Tools

# Machine learning part
learner = Learning.Scikitlearn(Tools.Options.dataset, learner_type=Learning.CLASSIFICATION)

model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT)
instance, prediction = learner.get_instances(model, n=1, correct=True)

# Explanation part
explainer = Explainer.decision_tree(model, instance)
explainer.add_clause_to_theory([-3, 4])
print(explainer.get_theory())
print("instance:", instance)
print("binary: ", explainer.binary_representation)
reason = explainer.minimal_majoritary_reason()
print("reason: ", reason)
print("is reason", explainer.is_reason(reason))


