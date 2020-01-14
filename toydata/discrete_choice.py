import numpy as np
import torch
from torch import Tensor


def create_discrete_choice_data(num_choice_sets=100,
                                num_choices_per_set=5,
                                num_features=3,
                                temperature=1,
                                probabilistic=True,
                                directed_weights=False,
                                return_true_weights=False):

    """
    Choice probabilities are softmaxed linear utility functions.

    TODO: lots of asserts to validate args
    """

    beta = np.random.normal(loc=0, scale=9, size=num_features)
    if directed_weights:
        beta = np.abs(beta)
    print("True beta", beta)
    features = np.random.normal(loc=0,
                                scale=1,
                                size=(num_choice_sets * num_choices_per_set, num_features))

    # Initialize data set container
    states = np.repeat(np.arange(num_choice_sets), num_choices_per_set)
    choices = np.zeros(num_choice_sets * num_choices_per_set)
    data = np.hstack((np.zeros((num_choice_sets * num_choices_per_set, 2)), features))
    data[:, 0] = states

    # Calculate linear scores
    utilities = features.dot(beta)

    # Sample "true" choices according on softmax of utilities (choice probabilities) for each choice set
    for st in range(num_choice_sets):
        ixs = np.where(states == st)[0]
        utils_st = utilities[ixs]

        if probabilistic:
            utils_st = utils_st - np.max(utils_st)
            exp_utils = np.exp(utils_st / temperature)
            probs = exp_utils / np.sum(exp_utils)
            # print(probs)
            choice = np.random.choice(np.arange(num_choices_per_set), size=1, p=probs)
            choices[ixs[choice]] = 1.0
        else:
            # Deterministic
            choices[ixs[utils_st == np.max(utils_st)]] = 1.0
    data[:, 1] = choices

    if return_true_weights:
        return data, beta
    else:
        return data


def deep_discrete_choice_example_data(model=None,
                                      num_choice_sets=100,
                                      num_choices_per_set=5,
                                      num_features=3,
                                      probabilistic=True,
                                      temperature=1):
    """
    Choice probabilities are based on a supplied Pytorch `model`.
    """
    features = torch.randn(size=(num_choice_sets * num_choices_per_set, num_features)) * (2 ** torch.arange(num_features) + 1).float()
    # for param in model.named_parameters():
    #     print(param)

    with torch.no_grad():
        utilities = model(features)
        states = Tensor(np.repeat(np.arange(num_choice_sets), num_choices_per_set))
        choices = Tensor(np.zeros(num_choice_sets * num_choices_per_set))
        data = Tensor(np.hstack((np.zeros((num_choice_sets * num_choices_per_set, 2)), features)))
        data[:, 0] = states
        for st in range(num_choice_sets):  # st = 0
            ixs = np.where(states == st)[0]
            utils_st = utilities[ixs]

            if probabilistic:
                # utils_st = utils_st - utils_st.max()
                exp_utils = torch.exp(utils_st/temperature)
                probs = exp_utils / exp_utils.sum()
                ## TODO: change to pure torch
                choice = np.random.choice(np.arange(num_choices_per_set), size=1, p=probs.numpy().flatten())
                choices[ixs[choice]] = 1.0
            else:
                # Deterministic
                choices[ixs[torch.t(utils_st == utils_st.max()).squeeze()]] = 1.0
        data[:, 1] = choices
    # if return_as_torch_tensor:
    #     data = torch.Tensor(data)  # .double()
    return data


