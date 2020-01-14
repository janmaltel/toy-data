from toydata.discrete_choice import create_discrete_choice_data

num_choice_sets = 5
num_alternatives_per_set = 4
num_features = 3
temperature = 1
probabilistic = True
directed_weights = False
return_true_weights = False

data = create_discrete_choice_data(num_choice_sets=num_choice_sets,
                                   num_alternatives_per_set=num_alternatives_per_set,
                                   num_features=num_features,
                                   temperature=temperature,
                                   probabilistic=probabilistic,
                                   directed_weights=directed_weights)

x_train, y_train, x_test, y_test = data.get_train_test_split(train_percentage=0.5)

