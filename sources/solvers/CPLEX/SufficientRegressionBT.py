from docplex.mp.model import Model



class SufficientRegression:
    def __init__(self):
        pass

    def create_model(self):
        random.shuffle(featureBlock)
        model = Model()
        model.context.cplex_parameters.threads = 1

        varTrees = [model.binary_var("b" + str(i))
                    for i in range(1 + forest.getMaxIndex())]
        varActiveBranches = [model.binary_var("a" + str(i))
                             for i in range(forest.getNbAllBranches())]
        varValueTree = [model.continuous_var(
            t.getMinValue(), t.getMaxValue(), "t" + str(t.getId())) for t in forest]

        varValueForest = model.continuous_var(lb=forest.getMinPossiblePrediction(
        ), ub=forest.getMaxPossiblePrediction(), name="forest")

        for tree in forest:
            model.add_constraint(
                model.sum([varActiveBranches[branch.getId()] for branch in tree]) == 1)
            model.add_constraint(model.sum([varActiveBranches[branch.getId(
            )] * branch.getValue() for branch in tree]) == varValueTree[tree.getId()])

            for branch in tree:
                model.add_constraint(model.sum([varTrees[l] for l in branch.getLiterals() if (
                        l > 0)] + [1 - varTrees[-l] for l in branch.getLiterals() if (l < 0)] + [-varActiveBranches[branch.getId()]]) <= len(
                    branch.getLiterals()) - 1)

        model.add_constraint(
            model.sum(v for v in varValueTree) == varValueForest)

        varLb = model.binary_var("lb")
        varUb = model.binary_var("ub")

        model.add_indicator(varLb, varValueForest <= lb, active_value=1)
        model.add_indicator(varUb, varValueForest >= ub, active_value=1)
        model.add_constraint(varLb + varUb == 1)

        # add the category constraints
        for cat in listCatBin:
            for i in range(len(cat) - 1):
                for j in range(i + 1, len(cat)):
                    assert cat[i] < len(varTrees) and cat[j] < len(varTrees)
                    model.add_constraint(varTrees[cat[i]] + varTrees[cat[j]] <= 1)

        # add the domain constraints.
        for feature in featureBlock:
            if len(feature) == 1:
                continue
            for i in range(len(feature) - 1):
                model.add_constraint(
                    varTrees[feature[i]] + 1 - varTrees[feature[i + 1]] >= 1)

        valuation = {}
        for i in instance:
            valuation[abs(i)] = i > 0

        notKnown = []
        for feature in featureBlock:
            tmp = []
            for l in feature:
                if abs(l) not in valuation:
                    continue
                if valuation[abs(l)]:
                    tmp.append((l, varTrees[l] == 1))
                else:
                    tmp.append((-l, varTrees[l] == 0))

                model.add_constraint(tmp[-1][1])
            notKnown.append(tmp)

        # to visualize the model, it is in /tmp
        # model.export_as_lp()

        nbRemoveByRotation = 0
        reason = []
        missclass = []

        starting_time = -time.process_time()

        timeCheck = False
        # by feature.
        if verbose:
            print("c First run, elimination by feature")

        while (len(notKnown) > 0 and (starting_time + time.process_time()) < time_out):
            model.set_time_limit(np.max([1, time_out - (starting_time + time.process_time())]))
            oldnotKnown = notKnown + [] + reason
            if verbose:
                print(len(notKnown), len(reason))
            feature = notKnown[-1]
            notKnown.pop()

            # remove all the boolean related to the feature.
            for c in feature:
                model.remove(c[1])

            # check if it is a necessary feature.
            solution = model.solve()
            if solution is not None or (starting_time + time.process_time()) > time_out:
                # for the moment this feature is required.
                tmp = []
                for c in feature:
                    model.add_constraint(c[1])
                    tmp.append(c)
                reason.append(tmp)

        # by boolean.
        notKnown = reason
        # oldnotKnown = notKnown + []
        reason = []
        longReason = []

        if verbose:
            print("c Second run, elimination by boolean")

        while (len(notKnown) > 0 and (starting_time + time.process_time()) < time_out):
            model.set_time_limit(np.max([1, time_out - (starting_time + time.process_time())]))
            oldnotKnown = notKnown + []
            if verbose:
                print(len(notKnown), len(reason))
            feature = notKnown[-1]
            notKnown.pop()

            # we already now that this feature has to be in the solution.
            if len(feature) == 1:
                reason.append(feature[0][0])
                longReason.append(feature[0][0])
                continue

            # search for the transition.
            posPoint = 0
            while (posPoint < len(feature) and feature[posPoint][0] > 0):
                posPoint += 1
            posPoint -= 1
            negPoint = posPoint + 1

            state = 0
            while (posPoint >= 0 or negPoint < len(feature) and (starting_time + time.process_time()) < time_out):
                model.set_time_limit(np.max([1, time_out - (starting_time + time.process_time())]))
                ind = -1
                if posPoint < 0:
                    ind = negPoint
                    negPoint += 1
                elif negPoint >= len(feature):
                    ind = posPoint
                    posPoint -= 1
                elif state % 2 == 0:
                    ind = posPoint
                    posPoint -= 1
                else:
                    ind = negPoint
                    negPoint += 1
                state += 1

                current = feature[ind]
                model.remove(current[1])

                # check if it is a necessary boolean feature.
                solution = model.solve()
                if solution is not None:
                    # this boolean feature is required.
                    reason.append(current[0])
                    longReason.append(current[0])
                    model.add_constraint(current[1])

                    # we know that one the bound is fixed
                    if current[0] > 0:
                        while posPoint >= 0:
                            longReason.append(feature[posPoint][0])
                            posPoint -= 1
                    else:
                        while negPoint < len(feature):
                            longReason.append(feature[negPoint][0])
                            negPoint += 1

                    interpretation = ModelSufficient.computeInterpretation(
                        solution, varTrees)
                    score = forest.computePrediction(interpretation)
                    if score > lb and score < ub:
                        missclass.append(current[0])

        if (starting_time + time.process_time()) > time_out:
            timeCheck = True
            for f in oldnotKnown:
                posPoint = 0
                while (posPoint < len(f) and f[posPoint][0] > 0):
                    posPoint += 1
                posPoint -= 1
                negPoint = posPoint + 1
                if posPoint >= 0:
                    reason.append(f[posPoint][0])
                if negPoint < len(f):
                    reason.append(f[negPoint][0])
            reason = simplifyImplicant(reason, featureBlock, list_cat_bin=listCatBin)


def solve(self, time_limit=None):
        pass