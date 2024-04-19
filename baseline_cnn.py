"""
code for training a baseline CNN model on the road dataset.
"""
import argparse
import gc

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision

from cnn_utils import *
from models import baseline_model
from roads_sets_definitions import *

mixed_precision.set_global_policy('mixed_float16')
_seed = 12345
np.random.seed(_seed)
tf.random.set_seed(_seed)
if __name__ == '__main__':
    """
    Since we are running multiple experiments with different parameters, we will use argparse to pass the parameters
    to the script. The parameters are:
    1. model_name: unuque name of the model
    2. datasplit_parcent: The percentage of the dataset to be used for training (10% to 90%)
    3. no_of_classes: The number of classes in the dataset (2 or 5)
    """
    parser = argparse.ArgumentParser(description='File and Model Parameters')
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--datasplit_parcent', required=True)
    parser.add_argument('--no_of_classes', required=True)
    args = parser.parse_args()
    model_name = args.model_name
    datasplit_parcent = args.datasplit_parcent
    no_of_classes = args.no_of_classes

    datasplit_parcent = int(datasplit_parcent)
    no_of_classes = int(no_of_classes)

    print("Percentage of training data: ", datasplit_parcent)
    print("Number of classes : ", no_of_classes)

    dir_name = "/work/amuhebwa_umass_edu/CURRENT_PROJECTS/roads"
    data_source = "PLANET"  # "PLANET" or "GEP"

    image_labels_dict = {
        224: '{}/images_labels_256_df.csv'.format(dir_name),
        128: '{}/images_labels_128_df.csv'.format(dir_name)
    }

    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    class_weights = defined_weights.get(no_of_classes)
    roads_dataset = pd.read_csv(image_labels_dict.get(IMAGE_SIZE))
    set_of_roads = sets_of_roads.get(datasplit_parcent)

    number_of_roads = []
    all_accuracies = []
    all_f1_scores = []
    all_roc_auc_scores = []
    all_precision_scores = []
    majority_class_test = []
    majority_class_pred = []
    perc_majority_class_test = []
    perc_majority_class_pred = []
    # this for road sets
    for _, current_roads in enumerate(set_of_roads):
        roads_count = len(current_roads)
        print("Number of roads used for training: {}".format(roads_count))
        train_dataset, validate_dataset, test_dataset = prepare_tensorflow_datasets(roads_dataset, no_of_classes, current_roads, batch_size)

        metrics = ["sparse_categorical_accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(3, name="top-3-accuracy"),
                   tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy")]
        # model = baseline_model(input_shape, no_of_classes, metrics)
        model = baseline_model(input_shape, no_of_classes, metrics)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=0, patience=5, mode='auto', restore_best_weights=True)
        history = model.fit(train_dataset, validation_data=validate_dataset, epochs=epochs, batch_size=batch_size,
                            class_weight=class_weights,
                            callbacks=[early_stopping],
                            verbose=1)

        predicted_results = model.predict(test_dataset)
        predicted_labels = tf.argmax(predicted_results, axis=1)
        prediction_prob = model.predict(test_dataset)
        prediction_prob = prediction_prob.astype(np.float32)
        actual_test_labels = np.concatenate([_labels for _images, _labels in test_dataset], axis=0)

        num_epochs_completed = len(history.history['loss'])
        accuracy, f1_score, precision_score, auc_roc_value = classification_metrics(actual_test_labels,
                                                                                    predicted_labels, prediction_prob,
                                                                                    no_of_classes)
        maj_class_test, maj_class_test_perc = percentage_majority_class(actual_test_labels)
        maj_class_pred, maj_class_pred_perc = percentage_majority_class(predicted_labels)

        number_of_roads.append(roads_count)
        all_accuracies.append(accuracy)
        all_f1_scores.append(f1_score)
        all_roc_auc_scores.append(auc_roc_value)
        all_precision_scores.append(precision_score)
        majority_class_test.append(maj_class_test)
        majority_class_pred.append(maj_class_pred)
        perc_majority_class_test.append(maj_class_test_perc)
        perc_majority_class_pred.append(maj_class_pred_perc)
        print("LEARNING RATE: {}".format(learning_rate))
        print("MODEL NAME: ", model_name)
        print('---> Accuracy: {}'.format(accuracy))
        print("---> AUC_ROC: {}".format(auc_roc_value))
        print("+=" * 30)
        del model
        _ = gc.collect()
        K.clear_session()
        del history

    results_df = pd.DataFrame()
    results_df["Num_test_roads"] = number_of_roads
    results_df['Accuracy'] = all_accuracies
    results_df['AUC_ROC'] = all_roc_auc_scores
    results_df['F1'] = all_f1_scores
    results_df['Precision'] = all_precision_scores
    results_df["test_majority_class"] = majority_class_test
    results_df["predict_majority_class"] = majority_class_pred
    results_df["test_majority_class_perc"] = perc_majority_class_test
    results_df["pred_majority_class_perc"] = perc_majority_class_pred
    name2save = "{}/prediction_results/{}_percent_{}_{}_results_{}_{}_classes.csv".format(dir_name, data_source, datasplit_parcent, model_name, IMAGE_SIZE, no_of_classes)
    results_df.to_csv(name2save, index=False)
