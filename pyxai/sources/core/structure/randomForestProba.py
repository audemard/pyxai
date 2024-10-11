from pyxai.sources.core.structure.randomForest import RandomForest
import numpy


class RandomForestProba(RandomForest):
    def __init__(self, forest, n_classes=2, learner_information=None, feature_names=None, theta=0.5):
        super().__init__(forest, n_classes, learner_information, feature_names)
        assert (n_classes == 2)
        self._theta = theta

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        if value < 0.5 or value >= 1:
            raise ValueError("vaule for theta is [0.5, 1[")
        self._theta = value

    def predict_instance(self, instance):
        sum_proba_positive = 0
        sum_proba_negative = 0
        for tree in self.forest:
            tmp = tree.predict_instance(instance)
            sum_proba_positive += tmp[1]
            sum_proba_negative += tmp[0]
        if sum_proba_positive / len(self.forest) >= self.theta:
            # assert (sum_proba_negative / len(self.forest) < self.theta)
            return 1
        if sum_proba_negative / len(self.forest) >= self.theta:
            # assert (sum_proba_positive / len(self.forest) < self.theta)
            return 0
        return None

    def predict_instance_default_rf(self, instance):
        n_votes = numpy.zeros(self.n_classes)
        for tree in self.forest:
            n_votes[numpy.argmax(tree.predict_instance(instance))] += 1
        return numpy.argmax(n_votes)
