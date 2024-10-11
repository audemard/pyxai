from pyxai.sources.core.structure.decisionTree import DecisionTree

class DecisionTreeProba(DecisionTree):
    def __init__(self, n_features, root, target_class=0, id_solver_results=0, learner_information=None,
                 force_features_equal_to_binaries=False, feature_names=None):
        super().__init__(n_features, root, target_class, id_solver_results, learner_information,
                 force_features_equal_to_binaries, feature_names)

