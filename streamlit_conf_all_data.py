import pickle


def get_data(dataset_type, test_category):
    with open(f'preprocessed_data/test_authors_{test_category}.pickle', 'rb') as handle:
        val_and_test_authors = pickle.load(handle)
    train_data = []
    if dataset_type == 'TRAIN':
        return train_data
    elif dataset_type == 'TEST':
        return val_and_test_authors


def read_scores(dataset_type, model_type, category):
    if dataset_type == 'TEST':
        last_model_category = model_type.split('_')[-1]
        model_type_last = model_type.split('_')[-1]
        with open(f'results/{model_type_last}/all_similarities_scores_test_{last_model_category}_{category}.pickle',
                  'rb') as handle:
            all_scores = pickle.load(handle)
    elif dataset_type == 'TRAIN':
        with open('all_similarities_scores_train.pickle', 'rb') as handle:
            all_scores = pickle.load(handle)
    return all_scores
