import numpy as np
import torch
from torch import Tensor
from toydata.data import ChoiceSetData
import warnings


def create_discrete_choice_data(num_choice_sets=100,
                                num_alternatives_per_set=5,
                                num_features=3,
                                temperature=1,
                                weight_means=0,
                                weight_sds=3,
                                feature_means=0,
                                feature_sds=1,
                                probabilistic=True,
                                directed_weights=False):
    cs_data = ChoiceSetData(num_features=num_features)
    cs_data.simulate_data(num_choice_sets=num_choice_sets,
                          num_alternatives_per_set=num_alternatives_per_set,
                          temperature=temperature,
                          weight_means=weight_means,
                          weight_sds=weight_sds,
                          feature_means=feature_means,
                          feature_sds=feature_sds,
                          probabilistic=probabilistic,
                          directed_weights=directed_weights
                          )
    return cs_data


def deep_discrete_choice_example_data(model=None,
                                      num_choice_sets=100,
                                      num_choices_per_set=5,
                                      num_features=3,
                                      probabilistic=True,
                                      temperature=1):
    """
    CURRENTLY NOT SUPPORTED. #TODO

    Choice probabilities are based on a supplied Pytorch `model`.
    """
    warnings.warning("deep_discrete_choice_example_data() currently not supported.")
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


