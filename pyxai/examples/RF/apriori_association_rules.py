import pandas as pd
from itertools import combinations
# Function to transform the tuples
def transform_tuples(tuples):
    transformed_list = []
    
    for antecedent, consequent in tuples:
        # Negation of the first tuple and concatenation with the second
        negated_antecedent = tuple(-x for x in antecedent)
        # Concatenate the transformed first tuple with the second tuple.
        combined_tuple = negated_antecedent + tuple(consequent)
        transformed_list.append(combined_tuple)
    
    return transformed_list



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
        frozenset(itemset1.union(itemset2))
        for itemset1 in itemsets
        for itemset2 in itemsets
        if len(itemset1.union(itemset2)) == length
    }


def get_frequent_itemsets(transactions, candidates, min_support):
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


def generate_rules(frequent_itemsets, itemset_supports, min_confidence):
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

def aprioris(df, min_support, min_confidence,e,rules_to_exclude=None):
    if rules_to_exclude is None:
        rules_to_exclude = []

    # Convert the rules to exclude into a frozenset to facilitate comparison.
    rules_to_exclude = [(frozenset(antecedent), frozenset(consequent)) for antecedent, consequent in rules_to_exclude]
    transactions = df.apply(lambda row: frozenset(row[row == 1].index), axis=1).tolist()

    candidates = {frozenset([item]) for item in df.columns}
    frequent_itemsets, itemset_supports = get_frequent_itemsets(transactions, candidates, min_support)
    all_frequent_itemsets = frequent_itemsets.copy()

    k = 2
    while k<=e:
        candidates = generate_candidates(frequent_itemsets, k)
        print(len(candidates))
        if not candidates:
            break
        frequent_itemsets, supports = get_frequent_itemsets(transactions, candidates, min_support)
        itemset_supports.update(supports)
        all_frequent_itemsets.update(frequent_itemsets)
        k += 1

    rules = generate_rules(all_frequent_itemsets, itemset_supports, min_confidence)
    return all_frequent_itemsets, rules
