from pyxai.sources.core.structure.type import OperatorCondition, Theory
from pyxai.sources.core.tools.encoding import CNFencoding

class BinaryMapping():

    def __init__(self, map_id_binaries_to_features, map_features_to_id_binaries, learner_information):
        self.map_id_binaries_to_features = map_id_binaries_to_features
        self.map_features_to_id_binaries = map_features_to_id_binaries
        self.map_numerical_features = {} #dict[id_feature] -> [id_binaries of the feature]
        self.map_categorical_features_ordinal = {} #dict[id_feature] -> [id_binaries of the feature]
        self.map_categorical_features_one_hot = {} #dict[overall_name] -> [id_binaries of the set of features representing the overall name of the feature that was one hot encoded]
        self.learner_information = learner_information



    @property
    def accuracy(self):
        return self.learner_information.accuracy

    @property
    def features_name(self):
        return self.learner_information.feature_names

    @property
    def numerical_features(self):
        return self.map_numerical_features

    @property
    def categorical_features_ordinal(self):
        return self.map_categorical_features_ordinal

    @property
    def categorical_features_one_hot(self):
        return self.map_categorical_features_ordinal
    
    def get_used_features(self):
        used_features = set()
        for key in self.map_features_to_id_binaries.keys():
            used_features.add(key[0])
        return len(used_features)
        
    def add_numerical_feature(self, id_feature):
        if id_feature in self.map_numerical_features:  #".keys() ???? bug here"
          raise ValueError("The given id_feature (" + str(id_feature) + ") is already considered as numerical.")
        if id_feature in self.map_categorical_features_ordinal:
          raise ValueError("The given id_feature (" + str(id_feature) + ") is already considered as categorical ordinal.")
        
        id_binaries_of_the_feature = []
        for key in self.map_features_to_id_binaries.keys():
          if key[0] == id_feature:
            id_binaries_of_the_feature.append(self.map_features_to_id_binaries[key][0])

        self.map_numerical_features[id_feature] = id_binaries_of_the_feature

    def add_categorical_feature_one_hot(self, overall_name, id_features):
        #print("overall_name:", overall_name)
        #print("id_features:", id_features)
        id_binaries_of_the_feature = []
        for key in self.map_features_to_id_binaries.keys():
            for id_feature in id_features:
                if key[0] == id_feature:
                    id_binaries_of_the_feature.append(self.map_features_to_id_binaries[key][0])
        self.map_categorical_features_one_hot[overall_name] = id_binaries_of_the_feature

    def add_categorical_feature_ordinal(self, id_feature):
        if id_feature in self.map_numerical_features: #".keys() ???? bug here"
            raise ValueError("The given id_feature (" + str(id_feature) + ") is already considered as numerical.")
        if id_feature in self.map_categorical_features_ordinal:
            raise ValueError("The given id_feature (" + str(id_feature) + ") is already considered as categorical ordinal.")
        
        id_binaries_of_the_feature = []
        for key in self.map_features_to_id_binaries.keys():
            if key[0] == id_feature:
                id_binaries_of_the_feature.append(self.map_features_to_id_binaries[key][0])

        self.map_categorical_features_ordinal[id_feature] = id_binaries_of_the_feature


    def get_theory(self, map_id_binary_sign, map_is_represented_by_new_variables, theory, max_id_binary_cnf):
        clauses = []
        new_variables = []
        id_new_var = max_id_binary_cnf
        

        #print("self.map_numerical_features:", self.map_numerical_features)
        if theory == Theory.ORDER or theory == Theory.ORDER_NEW_VARIABLES:
            # The > and >= operators
            
            for key in self.map_numerical_features.keys():
                id_binaries = self.map_numerical_features[key]
                conditions = [tuple(list(self.map_id_binaries_to_features[id])+[id]) for id in id_binaries]
                conditions = sorted(conditions, key=lambda t: t[2], reverse=True)
                id_binaries_sorted = tuple(condition[3] for condition in conditions)
                
                for i in range(len(id_binaries_sorted)-1): #To not takes the last
                    clauses.append((-id_binaries_sorted[i], id_binaries_sorted[i+1]))

                if theory == Theory.ORDER_NEW_VARIABLES:
                    id_new_var = id_new_var + 1
                    #associated_literals = []
                    for id_binary in id_binaries_sorted:
                        clauses.append((-id_new_var, map_id_binary_sign[id_binary]*id_binary))
                        map_is_represented_by_new_variables[id_binary] = True
                        # associated_literals.append(map_id_binary_sign[id_binary]*id_binary)
                    new_variables.append(id_new_var)
            
            # For catgorical features that was one hot encoded
            for key in self.map_categorical_features_one_hot.keys(): 
                id_binaries = self.map_categorical_features_one_hot[key]
                for i, id_1 in enumerate(id_binaries):
                    for j, id_2 in enumerate(id_binaries):
                        if i != j:
                            # we code a => not b that is equivalent to not a or not b (material implication)
                            clauses.append((-id_1, -id_2))    
                #for i, condition in enumerate(conditions):
                #    if i < len(conditions)-1:
                #        # we code a => b that is equivalent to not a or b (material implication)
                #        a = condition[3]
                #        b = conditions[i+1][3]
                #        clauses.append((-a, b))
                #if theory == Theory.ORDER_NEW_VARIABLES:
                #    id_new_var = id_new_var + 1
                #    associated_literals = []
                #    for condition in conditions:
                #        # we code not id_new_var or condition
                #        id_binary = condition[3]
                #        sign = 1 if id_binary in binary_representation else -1
                #        clauses.append((-id_new_var, sign*condition[3]))
                #        associated_literals.append(sign*condition[3])
                #    new_variables.append((id_new_var, associated_literals))
            # The == and != operators: 1 == 10 => 1 > 5 (not possible), 1 == 20 => 1 != 30 ? (possible) 
            #for key in self.map_categorical_features_ordinal.keys(): 
            #    id_binaries = self.map_categorical_features_ordinal[key]
            #    conditions = [list(self.map_id_binaries_to_features[id])+[id] for id in #id_binaries if self.map_id_binaries_to_features[id][1] == OperatorCondition.EQ]
                #print("id_binaries:", id_binaries)
                #print("conditions:", conditions)
             #   for i, condition_1 in enumerate(conditions):
             #       for j, condition_2 in enumerate(conditions):
             #           if i != j:
                            # we code a => not b that is equivalent to not a or not b (material implication)
             #               a = condition_1[3]
             #               b = condition_2[3]
             #               clauses.append((-a, -b))

        # The < and <= operators ? (not possible with xgboost but possible in the builder ...)
        return clauses, new_variables, map_is_represented_by_new_variables

    def compute_id_binaries(self):
        assert False, "Have to be implemented in a child class."


    def instance_to_binaries(self, instance, preference_order=None):
        """
        map_id_binaries_to_features: list[id_binary] -> (id_feature, operator, threshold)
        map_features_to_id_binaries: dict[(id_feature, operator, threshold)] -> [id_binary, n_appears, n_appears_per_tree]
        Transform a instance into a cube (conjunction of literals) according to the tree
        """
        output = []
        if preference_order is None:
            for key in self.map_features_to_id_binaries.keys():  # the keys are of the type: (id_feature, operator, threshold)
                id_feature = key[0]
                operator = key[1]
                threshold = key[2]

                if operator == OperatorCondition.GE:
                    sign = 1 if instance[id_feature - 1] >= threshold else -1
                elif operator == OperatorCondition.GT:
                    sign = 1 if instance[id_feature - 1] > threshold else -1
                elif operator == OperatorCondition.LE:
                    sign = 1 if instance[id_feature - 1] <= threshold else -1
                elif operator == OperatorCondition.LT:
                    sign = 1 if instance[id_feature - 1] < threshold else -1
                elif operator == OperatorCondition.EQ:
                    sign = 1 if instance[id_feature - 1] == threshold else -1
                elif operator == OperatorCondition.NEQ:
                    sign = 1 if instance[id_feature - 1] != threshold else -1
                else:
                    raise NotImplementedError("The operator " + str(self.operator) + " is not implemented.")

                output.append(sign * self.map_features_to_id_binaries[key][0])  # map_features_to_id_binaries[key][0] => id_binary
            return CNFencoding.format(output)
        assert True, "To implement"
        return output


    def get_id_features(self, binary_representation):
        return tuple(self.map_id_binaries_to_features[abs(lit)][0] for lit in binary_representation)


    def to_features(self, binary_representation, eliminate_redundant_features=True, details=False, inverse=False):
        """
        Convert an implicant into features. Return a tuple of features.
        Two types of features are available according to the details parameter.
        If details is False, each feature is represented by a string "name >= threshold" or "name < threshold".
        Else, if details is True, each feature is a python dict with the keys "id", "name", "threshold", "sign" and "weight".
        "id": The id of the feature (from 1 to the number of features) of the condition node represented by the literal of the implicant.
        "name": The name of the feature. It is None if it is not available.
        "threshold": The threshold of the condition node represented by the literal of the implicant.
        "sign": If the associated literal was positive in the implicant (resp. negative), it is True (resp. False).
        "weight": To give a weight to the feature (optional)

        Remark: call self.eliminate_redundant_features()
        """
        result = []
        if eliminate_redundant_features:
            binary_representation = self.eliminate_redundant_features(binary_representation, inverse)
        for lit in binary_representation:
            feature = dict()
            feature["id"] = self.map_id_binaries_to_features[abs(lit)][0]

            if self.learner_information is None:
                feature["name"] = "f" + str(feature["id"])
            elif self.learner_information.feature_names is None:
                feature["name"] = "f" + str(feature["id"])
            else:
                feature["name"] = self.learner_information.feature_names[feature["id"] - 1]

            feature["operator"] = self.map_id_binaries_to_features[abs(lit)][1]
            feature["threshold"] = self.map_id_binaries_to_features[abs(lit)][2]
            feature["sign"] = True if lit > 0 else False
            feature["weight"] = binary_representation[lit] if isinstance(binary_representation, dict) else None
            if details:
                result.append(feature)
            else:
                if feature["operator"] == OperatorCondition.GE:
                    str_sign = " >= " if feature["sign"] else " < "
                elif feature["operator"] == OperatorCondition.GT:
                    str_sign = " > " if feature["sign"] else " <= "
                elif feature["operator"] == OperatorCondition.LE:
                    str_sign = " <= " if feature["sign"] else " > "
                elif feature["operator"] == OperatorCondition.LT:
                    str_sign = " < " if feature["sign"] else " >= "
                elif feature["operator"] == OperatorCondition.EQ:
                    str_sign = " == " if feature["sign"] else " != "
                elif feature["operator"] == OperatorCondition.NEQ:
                    str_sign = " != " if feature["sign"] else " == "
                else:
                    raise NotImplementedError("The operator " + str(feature["operator"]) + " is not implemented.")

                result.append(str(feature["name"]) + str_sign + str(feature["threshold"]))
        return tuple(result)


    def extract_excluded_features(self, implicant, excluded_features):
        """Return index of the implicant to exclude.

        Args:
            implicant (_type_): _description_
            excluded_features (_type_): _description_

        Returns:
            :obj:`list` of int: List of index of the implicant to exclude
        """
        return [i for i, feature in enumerate(self.to_features(implicant, eliminate_redundant_features=False, details=True)) if
                feature["id_feature"] in excluded_features]


    def eliminate_redundant_features(self, binary_representation, inverse=False):
        """
        A implicant without redundant features i.e. If we have 'feature_a > 3' and 'feature_a > 2', we keep only the id_binary linked to the boolean
        corresponding to the 'feature_a > 3' Warning, the 'implicant' parameter can be a list of literals or a dict of literals.
        In the last case, it is a map literal -> weight.
        """
        positive = {}
        positive_weights = {}
        negative = {}
        negative_weights = {}
        n_redundant_features = 0
        no_done = []
        # Search redundant features
        for lit in binary_representation:
            key = self.map_id_binaries_to_features[abs(lit)]
            id_feature = key[0]
            operator = key[1]
            threshold = key[2]
            if operator == OperatorCondition.GE:
                if lit > 0:
                    if (id_feature, operator) not in positive:
                        positive[(id_feature, operator)] = [threshold]
                        if isinstance(binary_representation, dict):
                            positive_weights[(id_feature, operator)] = binary_representation[lit]
                    else:
                        n_redundant_features += 1
                        positive[(id_feature, operator)].append(threshold)
                        if isinstance(binary_representation, dict):
                            positive_weights[(id_feature, operator)] += binary_representation[lit]
                else:
                    if (id_feature, operator) not in negative:
                        negative[(id_feature, operator)] = [threshold]
                        if isinstance(binary_representation, dict):
                            negative_weights[(id_feature, operator)] = binary_representation[lit]
                    else:
                        n_redundant_features += 1
                        negative[(id_feature, operator)].append(threshold)
                        if isinstance(binary_representation, dict):
                            negative_weights[(id_feature, operator)] += binary_representation[lit]
            else:
                no_done.append(lit)
        # Copy the new implicant without these redundant features

        # Replace min(negative[(idx, operator)]))][0] by max(negative[(idx, operator)]))][0], to check !
        if not isinstance(binary_representation, dict):
            output = [self.map_features_to_id_binaries[(idx, operator, max(positive[(idx, operator)]) if inverse is False else min(positive[(idx, operator)]))][0] for (idx, operator) in positive.keys()]
            output += [-self.map_features_to_id_binaries[(idx, operator, min(negative[(idx, operator)]) if inverse is False else max(negative[(idx, operator)]))][0] for (idx, operator) in negative.keys()]
            output += no_done
            return tuple(output)
        
        output = [(self.map_features_to_id_binaries[(idx, operator, max(positive[(idx, operator)]) if inverse is False else min(positive[(idx, operator)]))][0], positive_weights[(idx, operator)]) for
                  (idx, operator) in positive.keys()]
        output += [(-self.map_features_to_id_binaries[(idx, operator, min(negative[(idx, operator)]) if inverse is False else max(negative[(idx, operator)]))][0], negative_weights[(idx, operator)]) for
                   (idx, operator) in negative.keys()]
        output += no_done
        output = {t[0]: t[1] for t in output}  # To dict
        return output
