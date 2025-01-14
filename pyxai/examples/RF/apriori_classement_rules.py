from pyxai import Builder
from itertools import combinations
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
        new_key = key - frozenset([element_to_remove])
        # Copier la valeur associée à la clé d'origine
        value = dictionary[key]
        # Supprimer l'ancienne clé du dictionnaire
        del dictionary[key]
        # Ajouter la nouvelle clé avec la même valeur
        dictionary[new_key] = value
    # else:
    #     print("La clé spécifiée n'existe pas dans le dictionnaire.")
    return dictionary

#Apriori
##########################################################################################################
def generate_candidates(itemsets, length,d,target_features,e):
    candidates = set()
    for itemset1 in itemsets:
        for itemset2 in itemsets:
            candidate = itemset1.union(itemset2)
            if len(candidate) == length:
                if length==d:
                    if 'y' in candidate or 'yy' in candidate:
                        candidates.add(candidate)
                else:
                     candidates.add(candidate)
    return candidates

def get_frequent_itemsets(transactions, candidates, min_support):
    itemset_counts = {itemset: 0 for itemset in candidates}
    for transaction in transactions:
        for candidate in candidates:
            if candidate.issubset(transaction):
                itemset_counts[candidate] += 1

    num_transactions = len(transactions)
    frequent_itemsets = {itemset for itemset, count in itemset_counts.items() if count / num_transactions >= min_support}
    return frequent_itemsets, {itemset: count / num_transactions for itemset, count in itemset_counts.items() if count / num_transactions >= min_support}

def generate_rules(frequent_itemsets, itemset_supports, min_confidence, rules_to_exclude):
    rules_dict = {}
    for itemset in frequent_itemsets:
        if len(itemset) > 1:
            if 'y' in itemset or 'yy' in itemset:
                for subset in map(frozenset, combinations(itemset, len(itemset) - 1)):
                    antecedent = subset
                    consequent = itemset - antecedent

                    # Exclure les règles spécifiques
                    if (antecedent, consequent) in rules_to_exclude:
                        continue
                    if 'y' in consequent or 'yy' in consequent:
                        if itemset in itemset_supports and antecedent in itemset_supports:
                            confidence = itemset_supports[itemset] / itemset_supports[antecedent]
                            if confidence >= min_confidence:
                                if antecedent in rules_dict:
                                    rules_dict[antecedent] = rules_dict[antecedent].union(consequent)
                                else:
                                    rules_dict[antecedent] = consequent

    # Générer les règles seulement si la combinaison existe dans itemset_supports
    rules = [(antecedent, consequent, itemset_supports[antecedent | consequent] / itemset_supports[antecedent])
             for antecedent, consequent in rules_dict.items() if antecedent | consequent in itemset_supports]
    return rules

def apriori(df, min_support, min_confidence,e,d,rules_to_exclude=None):
    if rules_to_exclude is None:
        rules_to_exclude = []

    # Convertir les règles à exclure en frozenset pour faciliter la comparaison
    rules_to_exclude = [(frozenset(antecedent), frozenset(consequent)) for antecedent, consequent in rules_to_exclude]
    transactions = df.apply(lambda row: frozenset(row[row == 1].index), axis=1).tolist()

    candidates = {frozenset([item]) for item in df.columns}
    frequent_itemsets, itemset_supports = get_frequent_itemsets(transactions, candidates, min_support)
    all_frequent_itemsets = frequent_itemsets.copy()

    k = 2
    while k<=e:
        # candidates = generate_candidates(frequent_itemsets, k)
        candidates = generate_candidates(frequent_itemsets, k,d, ['y', 'yy'],e)
        print("okijf")
        print(len(candidates))
        if not candidates:
            break
        frequent_itemsets, supports = get_frequent_itemsets(transactions, candidates, min_support)
        itemset_supports.update(supports)
        all_frequent_itemsets.update(frequent_itemsets)
        k += 1

    rules = generate_rules(all_frequent_itemsets, itemset_supports, min_confidence, rules_to_exclude)
    # print(rules)
    return all_frequent_itemsets, rules