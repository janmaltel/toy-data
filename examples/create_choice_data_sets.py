from toydata.discrete_choice import create_discrete_choice_data

num_choice_sets = 5
num_choices_per_set = 4
num_features = 3
temperature = 1
probabilistic = True
directed_weights = False
return_true_weights = False

data = create_discrete_choice_data(num_choice_sets=num_choice_sets,
                                   num_choices_per_set=num_choices_per_set,
                                   num_features=num_features,
                                   temperature=temperature,
                                   probabilistic=probabilistic,
                                   directed_weights=directed_weights,
                                   return_true_weights=return_true_weights)

