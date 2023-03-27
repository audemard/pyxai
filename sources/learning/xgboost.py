import copy
import json
import os
from numpy.random import RandomState

import xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pyxai import Tools
from pyxai.sources.core.structure.boostedTrees import BoostedTrees, BoostedTreesRegression
from pyxai.sources.core.structure.decisionTree import DecisionTree, DecisionNode, LeafNode
from pyxai.sources.core.tools.utils import compute_accuracy
from pyxai.sources.learning.Learner import Learner, LearnerInformation, NoneData
from pyxai.sources.core.structure.type import OperatorCondition

class Xgboost(Learner):
    """
    Load the dataset, rename the attributes and separe the prediction from the data
    """


    def __init__(self, data=NoneData, types=None):
        super().__init__(data, types)


    def get_solver_name(self):
        return str(self.__class__.__name__)


    def fit_and_predict_DT_CLS(self, instances_training, instances_test, labels_training, labels_test, learner_options):
        raise NotImplementedError("Xgboost does not have a Decision Tree Classifier.")
        

    def fit_and_predict_RF_CLS(self, instances_training, instances_test, labels_training, labels_test, learner_options):
        raise NotImplementedError("Xgboost does not have a Random Forest Classifier.")


    def fit_and_predict_BT_CLS(self, instances_training, instances_test, labels_training, labels_test, learner_options):
        if "eval_metric" not in learner_options.keys():
            learner_options["eval_metric"] = "mlogloss"
        # Training phase
        Tools.verbose("learner_options:", learner_options)
        xgb_classifier = xgboost.XGBClassifier(**learner_options)
        xgb_classifier.fit(instances_training, labels_training)
        # Test phase
        result = xgb_classifier.predict(instances_test)
        metrics = {
            "accuracy": compute_accuracy(result, labels_test)
        }

        extras = {
            "base_score": None
        }

        return (copy.deepcopy(xgb_classifier), metrics, extras)

    def fit_and_predict_DT_REG(self, instances_training, instances_test, labels_training, labels_test, learner_options):
        raise NotImplementedError("Xgboost does not have a Decision Tree Regressor.")
        
    def fit_and_predict_RF_REG(self, instances_training, instances_test, labels_training, labels_test, learner_options):
        raise NotImplementedError("Xgboost does not have a Random Forest Regressor.")

    def fit_and_predict_BT_REG(self, instances_training, instances_test, labels_training, labels_test, learner_options):
        #if "eval_metric" not in learner_options.keys():
        #    learner_options["eval_metric"] = "mlogloss"
        # Training phase
        Tools.verbose("learner_options:", learner_options)
        xgb_regressor = xgboost.XGBRegressor(**learner_options)
        
        xgb_regressor.fit(instances_training, labels_training)
        # Test phase
        
        result = xgb_regressor.predict(instances_test)
        metrics = {
            "mean_squared_error": mean_squared_error(labels_test, result),
            "root_mean_squared_error": mean_squared_error(labels_test, result, squared=False),
            "mean_absolute_error": mean_absolute_error(labels_test, result)
        }

        extras = {
            "base_score": float(0.5) if xgb_regressor.base_score is None else xgb_regressor.base_score,
        }
        
        return (copy.deepcopy(xgb_regressor), metrics, extras)

    def to_DT_CLS(self, learner_information=None):
        assert True, "Xgboost is only able to evaluate a classifier in the form of boosted trees"


    def to_RF_CLS(self, learner_information=None):
        assert True, "Xgboost is only able to evaluate a classifier in the form of boosted trees"


    def to_BT_CLS(self, learner_information=None):
        if learner_information is not None: self.learner_information = learner_information
        if self.n_features is None:
            self.n_features = learner_information[0].raw_model.n_features_in_
        if self.n_labels is None:
            self.n_labels = len(learner_information[0].raw_model.classes_)

        self.id_features = {"f{}".format(i): i for i in range(self.n_features)}
        BTs = [BoostedTrees(self.results_to_trees(id_solver_results), n_classes=self.n_labels, learner_information=learner_information) for
               id_solver_results, learner_information in enumerate(self.learner_information)]
        return BTs

    def to_DT_REG(self, learner_information=None):
        assert True, "Xgboost is only able to evaluate a classifier in the form of boosted trees"


    def to_RF_REG(self, learner_information=None):
        assert True, "Xgboost is only able to evaluate a classifier in the form of boosted trees"

    
    def to_BT_REG(self, learner_information=None):
        if learner_information is not None: self.learner_information = learner_information
        if self.n_features is None:
            self.n_features = learner_information[0].raw_model.n_features_in_
        
        self.id_features = {"f{}".format(i): i for i in range(self.n_features)}
        BTs = [BoostedTreesRegression(self.results_to_trees2(id_solver_results), learner_information=learner_information) for
               id_solver_results, learner_information in enumerate(self.learner_information)]
        return BTs


    def save_model(self, learner_information, filename):
        learner_information.raw_model.save_model(filename + ".model")

    def results_to_trees2(self, id_solver_results):
        dataframe = self.learner_information[id_solver_results].raw_model.get_booster().trees_to_dataframe()
        n_trees = self.learner_information[id_solver_results].raw_model.n_estimators
        #print("dataframe:", dataframe)
        #print("n_trees:", n_trees)
        decision_trees = []
        target_class = 0
        for i in range(n_trees) :
            dataframe_tree = dataframe.loc[dataframe['Tree'] == i]
            root = self.recuperate_nodes2(dataframe_tree, dataframe_tree.index.values[0])
            decision_trees.append(DecisionTree(self.n_features, root, target_class=[target_class], id_solver_results=id_solver_results))
            if self.n_labels > 2:  # Special case for a 2-classes prediction !
                target_class = target_class + 1 if target_class != self.n_labels - 1 else 0
        return decision_trees
        
    def recuperate_nodes2(self, dataframe_tree, id):
        if dataframe_tree["Feature"][id] == "Leaf":
            return LeafNode(dataframe_tree["Gain"][id])
        else:
            id_feature = self.id_features[dataframe_tree["Feature"][id]]
            
            threshold = round(dataframe_tree["Split"][id],2)
            #print("kiki", dataframe_tree["Split"][id].options.display.precision)
            #exit(0)
            decision_node = DecisionNode(int(id_feature + 1), threshold=threshold, operator=OperatorCondition.LT, left=None, right=None)

            id_right = dataframe_tree["Yes"][id]
            id_left = dataframe_tree["No"][id]
            id_right = dataframe_tree.index[dataframe_tree['ID'] == id_right].tolist()[0]
            id_left = dataframe_tree.index[dataframe_tree['ID'] == id_left].tolist()[0]
            
            decision_node.right = LeafNode(dataframe_tree["Gain"][id_right]) if dataframe_tree["Feature"][id_right] == "Leaf" else self.recuperate_nodes2(dataframe_tree, id_right)
            decision_node.left = LeafNode(dataframe_tree["Gain"][id_left]) if dataframe_tree["Feature"][id_left] == "Leaf" else self.recuperate_nodes2(dataframe_tree, id_left)
            return decision_node
        
    def results_to_trees(self, id_solver_results):
        
        xgb_BT = self.learner_information[id_solver_results].raw_model.get_booster()
        xgb_JSON = self.xgboost_BT_to_JSON(xgb_BT)
        decision_trees = []
        target_class = 0
        for i, tree_JSON in enumerate(xgb_JSON):
            #print(tree_JSON)
            tree_JSON = json.loads(tree_JSON)
            
            root = self.recuperate_nodes(tree_JSON)
            
            decision_trees.append(DecisionTree(self.n_features, root, target_class=[target_class], id_solver_results=id_solver_results))
            if self.n_labels > 2:  # Special case for a 2-classes prediction !
                target_class = target_class + 1 if target_class != self.n_labels - 1 else 0
            
        return decision_trees




    def recuperate_nodes(self, tree_JSON):
        if "children" in tree_JSON:
            assert tree_JSON["split"] in self.id_features, "A feature is not correct during the parsing from xgb_JSON to DT !"
            id_feature = self.id_features[tree_JSON["split"]]
            threshold = tree_JSON["split_condition"]
            decision_node = DecisionNode(int(id_feature + 1), threshold=threshold, left=None, right=None)
            id_right = tree_JSON["no"]  # It is the inverse here, right for no, left for yes
            for child in tree_JSON["children"]:
                if child["nodeid"] == id_right:
                    decision_node.right = LeafNode(child["leaf"]) if "leaf" in child else self.recuperate_nodes(child)
                else:
                    decision_node.left = LeafNode(child["leaf"]) if "leaf" in child else self.recuperate_nodes(child)
            return decision_node
        elif "leaf" in tree_JSON:
            # Special case when the tree is just a leaf, this append when no split is realized by the solver, but the weight have to be take into account
            return LeafNode(tree_JSON["leaf"])


    def xgboost_BT_to_JSON(self, xgboost_BT):
        save_names = xgboost_BT.feature_names
        xgboost_BT.feature_names = None
        xgboost_JSON = xgboost_BT.get_dump(with_stats=True)
        xgboost_BT.feature_names = save_names
        return xgboost_JSON


    def load_model(self, model_file):
        classifier = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        classifier.load_model(model_file)
        return classifier
