import pandas as pd
from itertools import combinations
import time
from mybitset import BitSet
from pyxai.sources.core.tools.encoding import CNF, CNFencoding
from pyxai import Builder
def list_to_tuple_pairs(lst):
    if len(lst) % 2 != 0:
        raise ValueError("La liste doit contenir un nombre pair d'éléments.")
    
    return [(lst[i], lst[i+1]) for i in range(0, len(lst), 2)]
def trasforme_list_tuple_to_binaire(tupl,rf_model):
    s=[]
    # print(tupl)
    for k in tupl:
        for n in k:
            # print(n)
            # print(bt_model.map_id_binaries_to_features[abs(n)])
            #s.append(tuple(bt_model.map_id_binaries_to_features[abs(n)]) + (True if n < 0 else False,))
            #s.append(bt_model.map_features_to_id_binaries[n])
            is_inside=False
            for e in rf_model.map_features_to_id_binaries:
                #print(dt_model.map_features_to_id_binaries)
                if e[0]==n: 
                    s.append((rf_model.map_features_to_id_binaries[e])[0])
                    is_inside=True
                elif e[0]==-n:
                    s.append(-(rf_model.map_features_to_id_binaries[e])[0])
                    is_inside=True
            if is_inside is False:
                #print("n not found",n)
                s.append((abs(n),Builder.GT,0.5, True if n < 0 else False))
    return s
def precision(model, X_test1, y_test1):
    s = 0
    predictions = []
    for instance in X_test1:
        predicted_label = model.predict_instance(instance)
        predictions.append(predicted_label)
        s += 1

    correct_predictions = sum(1 for pred, true_label in zip(predictions, y_test1) if pred == true_label)

    accuracy = correct_predictions / len(X_test1)
    return accuracy
def trasforme_tuple_to_binaire(tupl,dt_model):
    s=[]
    # print(tupl)
    for n in tupl:
        # print(n)
        # print(bt_model.map_id_binaries_to_features[abs(n)])
        #s.append(tuple(bt_model.map_id_binaries_to_features[abs(n)]) + (True if n < 0 else False,))
        #s.append(bt_model.map_features_to_id_binaries[n])
        is_inside=False
        for e in dt_model.map_features_to_id_binaries:
            #print(dt_model.map_features_to_id_binaries)
            if e[0]==n: 
                s.append((dt_model.map_features_to_id_binaries[e])[0])
                is_inside=True
            elif e[0]==-n:
                s.append(-(dt_model.map_features_to_id_binaries[e])[0])
                is_inside=True
        if is_inside is False:
            #print("n not found",n)
            s.append((abs(n),Builder.GT,0.5, True if n < 0 else False))
    return s
def convert(antecedent):
    converted_antecedent = []
    for item in antecedent:
        if 'X_' in item:
            converted_antecedent.append(int(item.replace('X_', '')))
        elif 'N_' in item:
            converted_antecedent.append(int(item.replace('N_', '-')))
        else:
            converted_antecedent.append(item)
    return frozenset(converted_antecedent)

def generelise(rule1, rule2,valeur1,valeur2):
    """
    Check if two rules are in conflict
    """
    rule1 = set(rule1)
    rule2 = set(rule2)
    # Vérifie si les conditions des règles sont les mêmes
    if valeur1==valeur2:
        if rule1.issubset(rule2):
                return True
    return False


def genereliseclassement(rule1, rule2,valeur1,valeur2):
    """
    Check if two rules are in conflict
    """
    # Vérifie si les conditions des règles sont les mêmes
    # if valeur1.issubset(valeur2):
    if rule1.issubset(rule2):
        return True
    return False

def remove_element_from_key(dictionary, key, element_to_remove):
    """
    Supprime un élément spécifique d'une clé dans un dictionnaire et crée une nouvelle clé avec la valeur associée à la clé d'origine.

    Args:
        dictionary (dict): Le dictionnaire contenant les clés et les valeurs.
        key (frozenset): La clé à modifier.
        element_to_remove: L'élément à supprimer de la clé.

    Returns:
        dict: Le dictionnaire mis à jour.
    """
    # Vérifier si la clé existe dans le dictionnaire
    if key in dictionary:
        # Créer une nouvelle clé sans l'élément à supprimer
        new_key = tuple(x for x in key if x != element_to_remove)
        # Copier la valeur associée à la clé d'origine
        value = dictionary[key]
        # Supprimer l'ancienne clé du dictionnaire
        del dictionary[key]
        # Ajouter la nouvelle clé avec la même valeur
        dictionary[new_key] = value
    # else:
    #     print("La clé spécifiée n'existe pas dans le dictionnaire.")
    return dictionary
# Function to transform the tuples
def rules_to_clauses(rules):
    clauses = []
    for antecedent, consequent in rules:
        # Negation of the first tuple and concatenation with the second
        negated_antecedent = tuple(-x for x in antecedent)
        # Concatenate the transformed first tuple with the second tuple.
        clause = negated_antecedent + (consequent, )
        clauses.append(clause)  
    return clauses

def remove_subsumed(cnf):
    cnf = sorted(cnf, key=lambda clause: len(clause))
    subsumed = [False for _ in range(len(cnf) + 1)]
    flags = [False for _ in range(CNFencoding.compute_max_id_variable(cnf) + 1)]
    for i, clause in enumerate(cnf):
        if subsumed[i]:
            continue
        for lit in clause:
            flags[abs(lit)] = True
        for j in range(i + 1, len(cnf)):
            nLiteralsInside = tuple(flags[abs(lit)] for lit in cnf[j]).count(True)
            if nLiteralsInside == len(clause):
                subsumed[j] = True
        for lit in clause:
            flags[abs(lit)] = False
    return CNF([clause for i, clause in enumerate(cnf) if not subsumed[i]])

#Convert the columns into numbers.
def convert(antecedent):
    converted_antecedent = []
    for item in antecedent:
        if 'X_' in item:
            converted_antecedent.append(int(item.replace('X_', '')))
        elif 'N_' in item:
            converted_antecedent.append(-int(item.replace('N_', '')))
        else:
            converted_antecedent.append(int(item))  # Si l'élément n'a ni 'X_' ni 'N_'
    return converted_antecedent

def generate_candidates(itemsets, length):
    # Generate candidates by combining frequent itemsets.
    return {
        itemsets[i].union(itemsets[j])
        for i in range(len(itemsets))
        for j in range(i+1, len(itemsets))
        if len(itemsets[i].union(itemsets[j])) == length
    }
    
    
    return [frozenset(combination) for combination in combinations(itemsets, length)]

def get_frequent_itemsets_old(transactions, candidates, min_support):
    # Count the occurrences of the candidates in the transactions.
    itemset_counts = {}
    for transaction in transactions:
        for candidate in candidates:
            if candidate.issubset(transaction):
                if candidate not in itemset_counts:
                    itemset_counts[candidate] = 0
                itemset_counts[candidate] += 1

    num_transactions = len(transactions)
    frequent_itemsets = {
        itemset for itemset, count in itemset_counts.items()
        if count / num_transactions >= min_support
    }
    itemset_supports = {
        itemset: count / num_transactions
        for itemset, count in itemset_counts.items()
        if count / num_transactions >= min_support
    }
    return frequent_itemsets, itemset_supports

def get_frequent_itemsets(transactions, candidates, min_support):
    # Count the occurrences of the candidates in the transactions.
    #itemset_counts = {itemset: 0 for itemset in candidates}
    itemset_counts = [0]*len(candidates)
    #print("len itemset_counts:", len(itemset_counts))
    #print("Start get_frequent_itemsets")


    for transaction in transactions:
        for i, candidate in enumerate(candidates):
            if candidate.issubset(transaction):
                itemset_counts[i] += 1
    
    num_transactions = len(transactions)
    
    frequent_itemsets = [
        itemset for i, itemset in enumerate(candidates)
        if itemset_counts[i] / num_transactions >= min_support
    ]

    #frequent_itemsets = {
    #    itemset for itemset, count in itemset_counts.items()
    #    if count / num_transactions >= min_support
    #}

    itemset_supports = {
        itemset: itemset_counts[i] / num_transactions
        for i, itemset in enumerate(candidates)
        if itemset_counts[i] / num_transactions >= min_support
    }

    #itemset_supports = {
    #    itemset: count / num_transactions
    #    for itemset, count in itemset_counts.items()
    #    if count / num_transactions >= min_support
    #}
    #print("time get_frequent_itemsets: ", time.time() - st)
    #print("End loop get_frequent_itemsets")
    return frequent_itemsets, itemset_supports


def generate_rules(frequent_itemsets, itemset_supports, min_confidence):
    print("Start generate_rules")
    rules_dict = {}
    for itemset in frequent_itemsets:
        if len(itemset) > 1:
                for subset in map(frozenset, combinations(itemset, len(itemset) - 1)):
                    antecedent = subset
                    consequent = itemset - antecedent
                    if itemset in itemset_supports and antecedent in itemset_supports:
                        confidence = itemset_supports[itemset] / itemset_supports[antecedent]
                        if confidence >= min_confidence:
                            if antecedent in rules_dict:
                                rules_dict[antecedent] = rules_dict[antecedent].union(consequent)
                            else:
                                rules_dict[antecedent] = consequent

    # Generate the rules only if the combination exists in itemset_supports.
    rules = [(antecedent, consequent, itemset_supports[antecedent | consequent] / itemset_supports[antecedent])
             for antecedent, consequent in rules_dict.items() if antecedent | consequent in itemset_supports]
    return rules
import sys

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def madelaine(database, time_limit=3600, n_max_rules=200000, explainer=None):
    
    #Compute: key -> value
    # dict_values_0: index_feature -> list of indexes of instances where the index_feature value is 0
    # dict_values_1: index_feature -> list of indexes of instances where the index_feature value is 1 
    total_time = time.time()

    database_tuples = tuple(database.itertuples(index=False, name=None))
    n_instances = len(database_tuples)
    n_features = len(database_tuples[0])
    print("n_instances:", n_instances)
    print("n_features:", n_features)
    dict_values_0 = {i:[] for i in range(1, n_features+1)}
    dict_values_1 = {i:[] for i in range(1, n_features+1)}

    for index_instance, instance in enumerate(database_tuples):
        for index_feature, value in enumerate(instance):
            dict_values_0[index_feature+1].append(index_instance) if value == 0 else dict_values_1[index_feature+1].append(index_instance)

    for i in range(1, n_features+1):
        dict_values_0[i] = set(dict_values_0[i])
        dict_values_1[i] = set(dict_values_1[i])
    
    hash_a_y = [False]*(n_features+1)
    hash_not_a_y = [False]*(n_features+1)
    hash_a_not_y = [False]*(n_features+1)
    hash_not_a_not_y = [False]*(n_features+1)

    hash_a_b = [False]*((n_features+1)*n_features+1)
    hash_not_a_b = [False]*((n_features+1)*n_features+1)
    hash_a_not_b = [False]*((n_features+1)*n_features+1)
    hash_not_a_not_b = [False]*((n_features+1)*n_features+1)


    # Compute all A->B to remove the A and B -> Y that are subsumed 
    candidates = tuple(combinations(range(1, n_features-1), 2)) # n_features-1 to remove y

    for candidate in candidates:
        a, b = candidate[0], candidate[1]
        key = a * b
        support_a_b = dict_values_1[a].intersection(dict_values_1[b])
        support_a_not_b = dict_values_1[a].intersection(dict_values_0[b])
        support_not_a_b = dict_values_0[a].intersection(dict_values_1[b])
        support_not_a_not_b = dict_values_0[a].intersection(dict_values_0[b])

        if len(support_a_not_b) == 0:
            # for a -> b: there is no (a -> not b) in the instances
            hash_a_b[key] = True
        elif len(support_a_b) == 0:
            # for a -> not b: no a -> b
            hash_a_not_b[key] = True
        if len(support_not_a_not_b) == 0:
            # for not a -> b: no not a -> not b
            hash_not_a_b[key] = True
        elif len(support_not_a_b) == 0:
            # for not a -> not b: no not a -> b
            hash_not_a_not_b[key] = True
        


    #for k == 2: generate all a->b rules
    candidates = tuple(combinations(range(1, n_features), 1))

    print("len candidates (k=2):", len(candidates))

    #Test all candidates: test a -> b, not(a) -> b, a -> not(b) and not(a) -> not(b)  
    rules = []
    y = n_features
    n_tests = 0

    for candidate in candidates:
        a = candidate[0]
        
        support_a_y = dict_values_1[a].intersection(dict_values_1[y])
        support_a_not_y = dict_values_1[a].intersection(dict_values_0[y])
        support_not_a_y = dict_values_0[a].intersection(dict_values_1[y])
        support_not_a_not_y = dict_values_0[a].intersection(dict_values_0[y])

        if len(support_a_not_y) == 0:
            # for a -> b: there is no (a -> not b) in the instances
            rules.append((((a,), y),support_a_y))
            hash_a_y[a] = True
        elif len(support_a_y) == 0:
            # for a -> not b: no a -> b
            rules.append((((a,), -y), support_a_not_y))
            hash_a_not_y[a] = True
        if len(support_not_a_not_y) == 0:
            # for not a -> b: no not a -> not b
            rules.append((((-a,), y), support_not_a_y))
            hash_not_a_y[a] = True
        elif len(support_not_a_y) == 0:
            # for not a -> not b: no not a -> b
            rules.append((((-a,), -y), support_not_a_not_y))
            hash_not_a_not_y[a] = True
    
        n_tests += 1

    
    #print("2k rules: ", rules)
    #for k == 3: generate all a and b -> c rules
    candidates = tuple(combinations(range(1, n_features), 2))
    print("len candidates (k=3):", len(candidates))
    #Test all candidates:
    # a and b => c 
    # a and b => not c 
    # not a and b => c 
    # not a and b => not c 
    for i, candidate in enumerate(candidates):
        
        if len(rules) % 100 == 0 and ((time.time() - total_time) > time_limit):
            break
        a, b = candidate[0], candidate[1]
        key_a_b = a * b
        intersection_a_b = dict_values_1[a].intersection(dict_values_1[b])
        intersection_not_a_b = dict_values_0[a].intersection(dict_values_1[b])
        intersection_a_not_b = dict_values_1[a].intersection(dict_values_0[b])
        intersection_not_a_not_b=dict_values_0[a].intersection(dict_values_0[b])

        support_a_b_y = intersection_a_b.intersection(dict_values_1[y])
        support_a_b_not_y = intersection_a_b.intersection(dict_values_0[y])
        support_not_a_b_y = intersection_not_a_b.intersection(dict_values_1[y])
        support_not_a_b_not_y = intersection_not_a_b.intersection(dict_values_0[y])
        support_a_not_b_y = intersection_a_not_b.intersection(dict_values_1[y])
        support_a_not_b_not_y = intersection_a_not_b.intersection(dict_values_0[y])
        support_not_a_not_b_y = intersection_not_a_not_b.intersection(dict_values_1[y])
        support_not_a_not_b_not_y = intersection_not_a_not_b.intersection(dict_values_0[y])

        # a and b => c: no a and b => not c 
        if len(support_a_b_not_y) == 0:
            if not (hash_a_y[a] or hash_a_y[b] or hash_a_not_b[key_a_b]): 
                rules.append((((a, b), y), support_a_b_y))
        # a and b => not c: no a and b => c 
        elif len(support_a_b_y) == 0:
            if not (hash_a_not_y[a] or hash_a_not_y[b] or hash_a_not_b[key_a_b]): 
                rules.append((((a, b), -y), support_a_b_not_y))
        
        # not a and b => c: no not a and b => not c 
        if len(support_not_a_b_not_y) == 0:
            if not (hash_not_a_y[a] or hash_a_y[b] or hash_not_a_not_b[key_a_b]): 
                rules.append((((-a, b), y), support_not_a_b_y))
        # not a and b => not c: no not a and b => c 
        elif len(support_not_a_b_y) == 0:
            if not (hash_not_a_not_y[a] or hash_a_not_y[b] or hash_not_a_not_b[key_a_b]): 
                rules.append((((-a, b), -y), support_not_a_b_not_y))
        
        # a and not b => c: no a and not b => not c 
        if len(support_a_not_b_not_y) == 0:
            if not (hash_a_y[a] or hash_not_a_y[b] or hash_a_b[key_a_b]): 
                rules.append((((a, -b), y), support_a_not_b_y))
        # a and not b => not c: no a and not b => c 
        elif len(support_a_not_b_y) == 0:
            if not (hash_a_not_y[a] or hash_not_a_not_y[b] or hash_a_b[key_a_b]): 
                rules.append((((a, -b), -y), support_a_not_b_not_y))

        # not a and not b => c: no not a and not b => not c
        if len(support_not_a_not_b_not_y) == 0:
            if not (hash_not_a_y[a] or hash_not_a_y[b] or hash_not_a_b[key_a_b]): 
                rules.append((((-a, -b), y), support_not_a_not_b_y))
        # not a and not b => -c: no not a and not b => c
        elif len(support_not_a_not_b_y) == 0:
            if not (hash_not_a_not_y[a] or hash_not_a_not_y[b] or hash_not_a_b[key_a_b]):
                rules.append((((-a, -b), -y), support_not_a_not_b_not_y))
        n_tests += 1

    if len(rules) > n_max_rules:
        rules = sorted(rules, key=lambda x: x[1], reverse=True)[:n_max_rules]
    
    rules = [r[0] for r in rules]
    len_rules_2 = len(tuple(r for r in rules if len(r[0]) == 1))
    len_rules_3 = len(tuple(r for r in rules if len(r[0]) == 2))

    print("n rules (total):", len(rules))
    
    return len_rules_2, len_rules_3, len(rules), (time.time() - total_time), rules

    

        





# def aprioris(df, min_support, min_confidence,max_length,rules_to_exclude=None):
#     if rules_to_exclude is None:
#         rules_to_exclude = []

#     # Convert the rules to exclude into a frozenset to facilitate comparison.
#     rules_to_exclude = [(frozenset(antecedent), frozenset(consequent)) for antecedent, consequent in rules_to_exclude]
#     size_bitset = len(df.columns) + 1
#     transactions = df.apply(lambda row: frozenset(row[row == 1].index), axis=1).tolist()
#     transactions_bitset = []
#     for transaction in transactions:
#         transactions_bitset.append(BitSet(size_bitset, [df.columns.get_loc(element) for element in transaction]))
#     transactions = transactions_bitset
#     #transactions = df.apply(lambda row: BitSet(size_bitset, row[row == 1].index), axis=1).tolist()


#     #candidates = {frozenset([item]) for item in df.columns}
#     candidates = [BitSet(size_bitset, [i]) for i in range(len(df.columns))]
#     print("candidates:", len(candidates))
    
#     print("transactions:", len(transactions_bitset))

#     #print("candidates:", candidates)
    
#     frequent_itemsets, itemset_supports = get_frequent_itemsets(transactions, candidates, min_support)
    
#     print("frequent_itemsets:", len(frequent_itemsets))
    
#     all_frequent_itemsets = frequent_itemsets.copy()

#     k = 2
#     while k<=max_length:
#         print("aprioris loop: ", k)
#         st = time.time()
#         candidates = generate_candidates(frequent_itemsets, k)
#         print("candidates:", len(candidates))
#         print("time candidates: ", time.time() - st)
        
#         if not candidates:
#             break
#         frequent_itemsets, supports = get_frequent_itemsets(transactions, candidates, min_support)
#         print("frequent_itemsets:", len(frequent_itemsets))
#         itemset_supports.update(supports)
#         all_frequent_itemsets.update(frequent_itemsets)
#         k += 1
#         print("time loop: ", time.time() - st)
        
#     rules = generate_rules(all_frequent_itemsets, itemset_supports, min_confidence)
#     return all_frequent_itemsets, rules
