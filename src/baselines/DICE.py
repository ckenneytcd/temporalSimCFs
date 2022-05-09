import dice_ml


class DICE:
    '''
    Wrapper for the DICE algorithm
    https://github.com/interpretml/DiCE
    '''

    def __init__(self, bb_model, dataset, target_action, cont_features, outcome, mutable_features, range_dict):
        self.bb_model = bb_model
        self.dataset = dataset
        self.target_action = target_action
        self.cont_features = cont_features
        self.outcome = outcome
        self.mutable_features = mutable_features
        self.total_CFs = 4
        self.range_dict = range_dict

        self.data = dice_ml.Data(dataframe=self.dataset,
                                 continuous_features=self.cont_features,
                                 outcome_name=self.outcome)

        # only random method works for now
        self.exp = dice_ml.Dice(self.data, self.bb_model, method="random")

    def get_counterfactuals(self, facts):
        cfs = []
        nrows = facts.shape[0]
        for i in range(nrows):
            fact = facts.iloc[[i]]
            dice_exp = self.exp.generate_counterfactuals(fact,
                                                         total_CFs=self.total_CFs,
                                                         desired_class=self.target_action,
                                                         features_to_vary=self.mutable_features,
                                                         permitted_range=self.range_dict)
            cfs.append(dice_exp)

        return cfs