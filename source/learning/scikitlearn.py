import copy
import pickle

import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from pyxai.source.core.structure.decisionTree import DecisionTree, DecisionNode, LeafNode
from pyxai.source.core.structure.randomForest import RandomForest
from pyxai.source.core.tools.utils import compute_accuracy
from pyxai.source.learning.Classifier import Classifier, NoneData


class Scikitlearn(Classifier):
    def __init__(self, data=NoneData):
        super().__init__(data)


    def get_solver_name(self):
        return str(self.__class__.__name__)


    def fit_and_predict_DT(self, instances_training, instances_test, labels_training, labels_test, max_depth=None, seed=0):
        # Training phase
        decision_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
        decision_tree.fit(instances_training, labels_training)

        # Test phase
        result = decision_tree.predict(instances_test)
        return (copy.deepcopy(decision_tree), compute_accuracy(result, labels_test))


    def fit_and_predict_RF(self, instances_training, instances_test, labels_training, labels_test, max_depth=None, seed=0):
        # Training phase
        decision_tree = RandomForestClassifier(max_depth=max_depth, random_state=seed)
        decision_tree.fit(instances_training, labels_training)

        # Test phase
        result = decision_tree.predict(instances_test)
        return (copy.deepcopy(decision_tree), compute_accuracy(result, labels_test))


    def fit_and_predict_BT(self, instances_training, instances_test, labels_training, labels_test, max_depth=None, seed=0):
        assert False, "Scikitlearn is not able to produce BT !"


    """
    Convert the Scikitlearn's decision trees into the program-specific objects called 'DecisionTree'.
    """


    def to_DT(self, classifier_information=None):
        if classifier_information is not None: self.classifier_information = classifier_information
        decision_trees = []
        for id_solver_results, _ in enumerate(self.classifier_information):
            sk_tree = self.classifier_information[id_solver_results].raw_model
            sk_raw_tree = sk_tree.tree_
            decision_trees.append(self.classifier_to_DT(sk_tree, sk_raw_tree, id_solver_results))
        return decision_trees


    def to_RF(self, classifier_information=None):
        if classifier_information is not None: self.classifier_information = classifier_information
        random_forests = []
        for id_solver_results, _ in enumerate(self.classifier_information):
            random_forest = self.classifier_information[id_solver_results].raw_model
            decision_trees = []
            for sk_tree in random_forest:
                sk_raw_tree = sk_tree.tree_
                decision_trees.append(self.classifier_to_DT(sk_tree, sk_raw_tree, id_solver_results))
            random_forests.append(
                RandomForest(decision_trees, n_classes=len(sk_tree.classes_), classifier_information=self.classifier_information[id_solver_results]))
        return random_forests


    def to_BT(self):
        assert False, "TODO"


    def save_model(self, classifier_information, filename):
        pickle.dump(classifier_information.raw_model, open(filename + ".model", 'wb'))


    """
    Convert a specific Scikitlearn's decision tree into a program-specific object called 'DecisionTree'.
    """


    def classifier_to_DT(self, sk_tree, sk_raw_tree, id_solver_results=0):
        nodes = {i: DecisionNode(int(feature + 1), threshold=sk_raw_tree.threshold[i], left=None, right=None)
                 for i, feature in enumerate(sk_raw_tree.feature) if feature >= 0}
        for i in range(len(sk_raw_tree.feature)):
            if i in nodes:
                # Set left and right of each node
                id_left = sk_raw_tree.children_left[i]
                id_right = sk_raw_tree.children_right[i]
                nodes[i].left = nodes[id_left] if id_left in nodes else LeafNode(numpy.argmax(sk_raw_tree.value[id_left][0]))
                nodes[i].right = nodes[id_right] if id_right in nodes else LeafNode(numpy.argmax(sk_raw_tree.value[id_right][0]))
        root = nodes[0] if 0 in nodes else DecisionNode(1, 0, sk_raw_tree.value[0][0])
        return DecisionTree(sk_tree.n_features_in_, root, sk_tree.classes_, id_solver_results=id_solver_results,
                            classifier_information=self.classifier_information[id_solver_results])


    def load_model(self, model_file):
        classifier = None
        with open(model_file, 'rb') as file:
            classifier = pickle.load(file)
        return classifier
