import numpy as np
import tensorflow as tf


class Data:
    """
    Abstract data container. Children should have functionality to
        split train/test data,
        shuffle data
        add (push) new data instances
        delete data
        ...?
    """
    def __init__(self):
        pass

    def get_train_test_split(self):
        raise NotImplementedError


class ChoiceSetData(Data):
    """
    Data structure to store choice set data to use with discrete choice models
    (https://en.wikipedia.org/wiki/Discrete_choice).

    Every row corresponds to one choice alternative. One CHOICE SET consists of
    multiple choice alternative.

    If p is the number of features, the idea is to have p+2 columns, where the
    first column is a choice set index and the second column is the choice
    variable (one-hot encoding). The remaining columns are feature values.

    Data
    """
    def __init__(self,
                 num_features,
                 max_choice_set_size=np.inf):
        self.num_features = num_features
        self.data = np.zeros((0, self.num_features + 2))
        self.choice_set_counter = 0
        self.current_number_of_choice_sets = 0
        self.num_alternatives_per_set = 0
        self.num_alternatives_per_set_is_constant = False
        self.max_choice_set_size = max_choice_set_size
        self.true_weights = np.array([])

    def simulate_data(self,
                      num_choice_sets=100,
                      num_alternatives_per_set=5,
                      temperature=1,
                      weight_means=0,
                      weight_sds=5,
                      feature_means=0,
                      feature_sds=1,
                      probabilistic=True,
                      directed_weights=False,
                      ):
        """
        Simulate data by simulating a decision maker who makes choices according to
        choice probabilities (for action/alternative `a` in state `s`)

                                       e^{U(a', s, \theta)
                        p(a', s) =   ----------------------                (1)
                                      Σ_a e^U(a, s, \theta)

        where U(a, s, \theta) is a utility function of action a in state s. U is
        parametrized by \theta. Currently, U(s, a, \theta) is a linear function
        of state-action-features \phi(a, s), but it could in principle be any
        function (e.g., neural net).

        This function first samples a vector of (Gaussian)feature weights. It then
        samples (Gaussian) state-action features for every alternative in every
        choice set. The actual choice is then sampled according to choice probs (1).

        Currently, every choice set has the same number of alternatives. This will
        change in future versions (TODO)
        """
        self.true_weights = np.random.normal(loc=weight_means, scale=weight_sds, size=self.num_features)
        if directed_weights:
            true_weights = np.abs(self.true_weights)
        features = np.random.normal(loc=feature_means,
                                    scale=feature_sds,
                                    size=(num_choice_sets * num_alternatives_per_set, self.num_features))

        if isinstance(num_alternatives_per_set, int):
            # Pre-allocate data set container (size is known in advance here!)
            states = np.repeat(np.arange(num_choice_sets), num_alternatives_per_set)
            choices = np.zeros(num_choice_sets * num_alternatives_per_set)
            self.data = np.hstack((np.zeros((num_choice_sets * num_alternatives_per_set, 2)), features))
            self.data[:, 0] = states

            # Calculate linear scores (some would say "logits")
            utilities = features.dot(self.true_weights)

            # Sample "true" choices according on softmax of utilities (choice probabilities)
            # on each choice set separately.
            for st in range(num_choice_sets):
                ixs = np.where(states == st)[0]
                utils_st = utilities[ixs]

                if probabilistic:
                    utils_st = utils_st - np.max(utils_st)
                    exp_utils = np.exp(utils_st / temperature)
                    probs = exp_utils / np.sum(exp_utils)
                    choice = np.random.choice(np.arange(num_alternatives_per_set), size=1, p=probs)
                    choices[ixs[choice]] = 1.0
                else:  # Deterministic
                    choices[ixs[utils_st == np.max(utils_st)]] = 1.0
            self.data[:, 1] = choices
        else:
            raise NotImplementedError

        self.choice_set_counter = num_choice_sets
        self.current_number_of_choice_sets = num_choice_sets
        self.num_alternatives_per_set = num_alternatives_per_set
        self.num_alternatives_per_set_is_constant = True

    def get_data(self, transform_to_tf_tensor=False):
        data = self.data
        if transform_to_tf_tensor:
            data = tf.convert_to_tensor(data)
        return data

    def get_train_test_split(self, train_percentage=0.5, transform_to_tf_tensor=False):
        data = self.data
        if transform_to_tf_tensor:
            data = tf.convert_to_tensor(data)
        if self.num_alternatives_per_set_is_constant:
            split_row = int(int(self.current_number_of_choice_sets * train_percentage) * self.num_alternatives_per_set)
            x_train = data[:split_row, 2:]
            y_train = data[:split_row, :2]
            x_test = data[split_row:, 2:]
            y_test = data[split_row:, :2]
        else:
            raise NotImplementedError
        return x_train, y_train, x_test, y_test

    def push(self, features, choice_index, delete_oldest=False):
        """
        Adds a choice set to the existing data set.
        Useful, for example, in RL applications.
        """
        choice_set_len = len(features)
        one_hot_choice = np.zeros((choice_set_len, 1))
        one_hot_choice[choice_index] = 1.
        choice_set_index = np.full(shape=(choice_set_len, 1),
                                   fill_value=self.choice_set_counter)
        self.data = np.vstack((self.data, np.hstack((choice_set_index,
                                                     one_hot_choice, features))))
        self.choice_set_counter += 1.
        self.current_number_of_choice_sets += 1.
        if delete_oldest:
            first_choice_set_index = self.data[0, 0]
            for ix in range(self.max_choice_set_size+1):
                if self.data[ix, 0] != first_choice_set_index:
                    break
            if ix > self.max_choice_set_size:
                raise ValueError("Choice set should not be higher than " + str(self.max_choice_set_size))
            self.data = self.data[ix:]
            if self.current_number_of_choice_sets > 0:
                self.current_number_of_choice_sets -= 1.
        self.data = np.ascontiguousarray(self.data)

    def sample(self):
        # Currently just returns entire data set.
        return self.data

    def delete_data(self, set_counter_to_zero=False):
        self.data = np.zeros((0, self.num_features + 2))
        self.current_number_of_choice_sets = 0
        if set_counter_to_zero:
            self.choice_set_counter = 0

