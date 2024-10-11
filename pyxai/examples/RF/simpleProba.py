from pyxai import Learning, Explainer, Tools
from pyxai.sources.core.structure.type import Indexes

# usage
# python3 pyxai/examples/RF/Simple.py -dataset=path/to/dataset.csv
# Check V1.0: Ok 

# Machine learning part
learner = Learning.Scikitlearn(Tools.Options.dataset, learner_type=Learning.CLASSIFICATION)
model_proba = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RFP)
# model_classic = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF)
explainer = Explainer.initialize(model_proba)
instances = learner.get_instances(model_proba, indexes=Learning.TEST, details=True)

accuracy_classic = 0
accuracy_proba = 0
accuracy_proba_sure = 0
number_unknown = 0
model_proba.theta = 0.5
for instance in instances:
    # print(instance)
    inst = instance["instance"]
    explainer.set_instance(inst)
    if model_proba.predict_instance_default_rf(inst) == instance["label"]:
        accuracy_classic += 1
    predict_p = model_proba.predict_instance(inst)
    if predict_p is None:
        number_unknown += 1
    if predict_p == instance["label"]:
        accuracy_proba += 1
    if predict_p is not None and predict_p == instance["label"]:
        accuracy_proba_sure += 1

print("nb test instances: ", len(instances))
print("accuracy_classic", accuracy_classic / len(instances))
print("accuracy_proba", accuracy_proba / len(instances))
print("accuracy proba sure", accuracy_proba_sure / (len(instances) - number_unknown))
print("nb unknown: ", number_unknown / len(instances))
