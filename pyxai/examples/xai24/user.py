import math
import constants
from pysat.solvers import Glucose4

class User:
    def __init__(self, explainer, positive_instances, negative_instances):
        self.explainer = explainer
        explainer.set_instance(positive_instances[0])
        self.nb_variables = len(explainer.binary_representation)
        n_total = len(positive_instances) + len(negative_instances)
        n_positives = round((len(positive_instances) / n_total)*constants.N)
        n_negatives = round((len(negative_instances) / n_total)*constants.N)
        self.positive_rules = self._create_rules(explainer, positive_instances, constants.theta)[0:n_positives]
        self.negative_rules = self._create_rules(explainer, negative_instances, -constants.theta)[0:n_negatives]

    def predict_instance(self, binary_representation):
        """
        Take in parameter the binary representation of an instance
        return 1 if it is classified 1
        return 0 if it is classified 0
        return None otherwise
        """
        for rule in self.positive_rules:
            if generalize(self.explainer, rule, binary_representation):
                return 1
        for rule in self.negative_rules:
            if generalize(self.explainer, rule, binary_representation):
                return 0
        return None

    def get_rules_predict_instance(self, binary_representation, prediction):
        tmp = []
        if prediction:
            for rule in self.positive_rules:
                if generalize(self.explainer, rule, binary_representation):
                    tmp.append(rule)
        else:
            for rule in self.negative_rules:
                if generalize(self.explainer, rule, binary_representation):
                    tmp.append(rule)
        return tmp

    def remove_specialized(self, reason, positive):
        rules = self.positive_rules if positive else self.negative_rules
        tmp = [r for r in rules if not generalize(self.explainer, reason, r)]
        if len(tmp) != len(rules):
            tmp.append(reason)
            constants.statistics["generalisations"] += 1
            if positive:
                self.positive_rules = tmp
            else:
                self.negative_rules = tmp

    # -------------------------------------------------------------------------------------
    #  Create the rules for a given set of instances
    def _create_rules(self, explainer, instances, theta):
        result = []
        for instance in instances:
            explainer.set_instance(instance)

            reason = explainer.tree_specific_reason(n_iterations=constants.n_iterations, theta=theta)

            new_rule = True
            for rule in result:  # reason does not specialize existing rule
                if generalize(self.explainer, rule, reason):
                    # print("\n---\nreason:", reason, "\nspecial:", rule)
                    new_rule = False
                    break
            if new_rule:  # if not
                tmp = []  # can be done more efficiently
                for rule in result:  # remove specialized rules
                    if not generalize(self.explainer, reason, rule):
                        tmp.append(rule)
                    else:
                        pass
                        # print("\n---\nrule:", rule, "\nspecial:", reason)

                tmp.append(reason)  # do not forget to add this one
                result = tmp
        return sorted(result)


    def accurary(self, test_set):
        nb = 0
        total = 0
        for instance in test_set:
            self.explainer.set_instance(instance["instance"])
            prediction = self.predict_instance(self.explainer.binary_representation)
            if prediction is not None:
                nb += 1 if prediction == instance['label'] else 0
                total += 1
        if total == 0:
            return None
        return nb / total


# -------------------------------------------------------------------------------------

# c statistics {'rectifications': 21, 'generalisations': 13, 'cases_1': 5, 'cases_2': 20, 'cases_3': 0, 'cases_4': 0, 'cases_5': 5, 'n_positive': 0, 'n_negatives': 113, 'n_positives': 74}

def generalize(explainer_AI, rule1, rule2):
    """
    Return True if rule1 generalizes rule2
    a generalize ab
    """
    tmp1 = explainer_AI.extend_reason_with_theory(rule1)
    tmp2 = explainer_AI.extend_reason_with_theory(rule2)

    if len(tmp1) > len(tmp2):
        return False

    for lit in tmp1:
        if lit not in tmp2:
            return False

    # occurences = [0 for _ in range(len_binary + 1)]
    # for lit in rule1:
    #    occurences[abs(lit)] = lit
    # for lit in rule2:
    #    if occurences[abs(lit)] != lit:
    #        return False
    return True


def specialize(explainer_AI, rule1, rule2):
    return generalize(explainer_AI, rule2, rule1)


from pysat.solvers import Glucose4

gluglu = None
def conflict(explainer_AI, rule1, rule2):
    """
    Check if two rules are in conflict
    """
    global gluglu
    if gluglu is None:
        gluglu = Glucose4()
        gluglu.append_formula(explainer_AI.get_model().get_theory([]))  # no need of binary representation

    return gluglu.solve(assumptions=rule1 + rule2)  # conflict if SAT

    """
    Old version
    tmp1 = explainer_AI.extend_reason_with_theory(rule1)
    tmp2 = explainer_AI.extend_reason_with_theory(rule2)
    for lit in tmp1:
        if -lit in tmp2:
            return False
    return True
"""


