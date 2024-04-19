import argparse
import gc

import evaluate
import numpy as np
import pandas as pd
import tensorflow as tf
from roads_sets_definitions import *
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from typing import Dict

from models import pretrained_model
from roads_sets_definitions import defined_weights

_seed = 12345
np.random.seed(_seed)
tf.random.set_seed(_seed)
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = 224  # not the right place to define this but works for now. Fix it later.
auroc_metric = evaluate.load("roc_auc")

"""
some of these functions are copied from the roads_utils.py file.
Note: Fix it once you get all the code working. 
"""


def class_func(features, label):
    return label


def balanced_class_labels(unbalanced_ds, target_dist, _batch_size):
    resample_ds = (unbalanced_ds.unbatch().rejection_resample(class_func, target_dist=target_dist).batch(_batch_size))
    balanced_ds = resample_ds.map(lambda extra_label, features_and_label: features_and_label)
    return balanced_ds


def percentage_majority_class(class_labels):
    uniques, counts = np.unique(class_labels, return_counts=True)
    percentages = dict(zip(uniques, counts * 100 / len(class_labels)))
    max_class_perc = max(percentages.values())
    max_class = list([k for k, v in percentages.items() if v == max_class_perc]).pop()
    return max_class, max_class_perc


def calculate_accuracy_scores(predicted_labels, actual_labels):
    # predicted_labels = tf.argmax(predicted_results, axis=1)
    # predicted_labels = outputs.predictions.argmax(1)
    _accuracy = accuracy_score(actual_labels, predicted_labels)
    return _accuracy


def classification_metrics(actual_labels, predicted_labels):
    scores_F1 = f1_score(actual_labels, predicted_labels, average='weighted')
    scores_precision = precision_score(actual_labels, predicted_labels, average='weighted', zero_division=0)
    return scores_F1, scores_precision,


def roc_auc_score_multiclass(actual_class_labels, predicted_class_labels, average="macro"):
    # creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class_labels)
    roc_auc_dict = {}
    for per_class in unique_class:
        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class_labels]
        new_pred_class = [0 if x in other_class else 1 for x in predicted_class_labels]

        # using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc
    mean_roc = np.mean([*roc_auc_dict.values()])
    # return roc_auc_dict
    return mean_roc


def custom_auroc_calculator(_predicted_labels, prediction_proba, _actual_labels, _no_of_classes):
    roc_value = -999
    if _no_of_classes == 2:
        pred_proba = prediction_proba[:, 1]
        roc_value = auroc_metric.compute(references=_actual_labels, prediction_scores=pred_proba)
        roc_value = roc_value['roc_auc']
    else:
        roc_value = roc_auc_score_multiclass(_actual_labels, _predicted_labels)
    return roc_value


def get_activation_fn(no_of_classses: int):
    if no_of_classses == 2:
        return "sigmoid"
    else:
        return "softmax"


def roads5ClassLabel(iriValue: np.float64, allowNeg=False):
    if 0 <= int(iriValue) <= 7:
        labelName = 0  # 'great'
    elif 7 < int(iriValue) <= 12:
        labelName = 1  # 'good'
    elif 12 < int(iriValue) <= 15:
        labelName = 2  # 'fair'
    elif 15 < int(iriValue) <= 20:
        labelName = 3  # 'poor'
    elif int(iriValue) > 20:
        labelName = 4  # 'bad'
    else:
        if allowNeg:
            labelName = 0
        else:
            labelName = 'invalid'
    return labelName


def roadsBinaryClassLabel(iriValue: np.float64, allowNeg=False):
    threshold = 7
    if 0 <= int(iriValue) <= threshold:
        labelName = 0  # 'good'
    elif int(iriValue) > threshold:
        labelName = 1  # 'bad'
    else:
        if allowNeg:
            labelName = 0
        else:
            labelName = 'invalid'
    return labelName


def convert_labels_to_classes(num_classes: int, df: pd.DataFrame) -> pd.DataFrame:
    df = df.sample(frac=1.0, replace=False)
    if num_classes == 2:
        df['classLabel'] = df.apply(lambda row: roadsBinaryClassLabel(row['IRI']), axis=1)
    elif num_classes == 5:
        df['classLabel'] = df.apply(lambda row: roads5ClassLabel(row['IRI']), axis=1)
    else:  # regression
        df['classLabel'] = df['IRI']
    return df


def decode_img(img):
    img = tf.io.decode_png(img, channels=3)
    return tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])


def load_and_preprocess_image(path):
    _img = tf.io.read_file(path)
    _img = decode_img(_img)
    return _img


def load_and_preprocess_from_path_label(path, _label):
    return load_and_preprocess_image(path), _label


def prepare_tf_dataset(current_ds, batchsize, IMG_SIZE, shuffle=False, augment=False):
    # 1. resize and scale the image
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    current_ds = current_ds.map(lambda x, y: (normalization_layer(x), y))

    # 2. Flip the images
    # data_augmentation = tf.keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.2), ])
    # data_augmentation = tf.keras.Sequential([layers.RandomRotation(0.2), ])
    # data_augmentation = tf.keras.Sequential([layers.RandomFlip("horizontal_and_vertical")])
    data_augmentation = tf.keras.Sequential([layers.RandomFlip("horizontal"),
                                             layers.RandomRotation(factor=0.02),
                                             layers.RandomZoom(height_factor=0.2, width_factor=0.2)])

    current_ds = current_ds.batch(batchsize)
    if shuffle:
        current_ds = current_ds.shuffle(batchsize * 10, reshuffle_each_iteration=True)

    if augment:
        current_ds = current_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    # return current_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return current_ds


def calculate_class_weights(df: pd.DataFrame) -> Dict:
    y = df['classLabel'].values
    class_wghts = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    class_weights = dict(zip(np.unique(y), class_wghts))
    for key in class_weights:
        if class_weights[key] < 1.0:
            class_weights[key] = 1.0
    return class_weights


def sample_equal_labels(number_of_samples, temp_df):
    _, unique_counts = np.unique(temp_df.classLabel.values, return_counts=True)
    number_of_samples = min(number_of_samples, np.max(unique_counts) // 2)
    balanced_df = temp_df.groupby("classLabel",
                                  as_index=False,
                                  group_keys=False).apply(lambda s: s.sample(number_of_samples, replace=True))
    # balanced_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    balanced_df = balanced_df.reset_index(drop=True)
    return balanced_df


def smart_split_data(df, heldout_train_roads, _split_percentage):
    no_of_samples = 50000
    all_road_names = df.RoadName.unique()
    heldout_train_roads = list(set(heldout_train_roads).intersection(set(all_road_names)))

    if len(heldout_train_roads) != 0 and set(heldout_train_roads).issubset(set(all_road_names)):  # full_held_roads
        train_df = df[df['RoadName'].isin(heldout_train_roads)].reset_index(drop=True)
        test_df = df[~df['RoadName'].isin(heldout_train_roads)]
        test_df = test_df.sample(frac=1.0, replace=False, random_state=1234)
        test_df = test_df.reset_index(drop=True)
        train, validate_df = train_test_split(train_df, train_size=0.80, random_state=1234)
    else:
        df = df.sample(n=df.shape[0], replace=False, random_state=1234).reset_index(drop=True)
        train_df, test_df = train_test_split(df, train_size=_split_percentage, random_state=1234)
        train_df, validate_df = train_test_split(train_df, train_size=0.80, random_state=1234)

    train_df = sample_equal_labels(no_of_samples, train_df)
    validate_df = sample_equal_labels(no_of_samples, validate_df)
    test_df = sample_equal_labels(no_of_samples, test_df)
    train_df = train_df.sample(frac=1.0, replace=False).reset_index(drop=True)
    validate_df = validate_df.sample(frac=1.0, replace=False).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, replace=False).reset_index(drop=True)

    print('-' * 50)
    print('Train Splits')
    print(train_df.groupby('classLabel').count())
    print('-' * 50)
    return train_df, validate_df, test_df


def prepare_tensorflow_datasets(roads2015_dataset, no_of_classes, test_road_names, _split_percentage, batch_size):
    roads2015_dataset = convert_labels_to_classes(no_of_classes, roads2015_dataset)
    train_df, validate_df, test_df = smart_split_data(roads2015_dataset, test_road_names, _split_percentage)
    train_ds = tf.data.Dataset.from_tensor_slices((train_df.ImagePath.values, train_df.classLabel.values))
    train_ds = train_ds.map(load_and_preprocess_from_path_label, num_parallel_calls=AUTOTUNE)
    train_ds = prepare_tf_dataset(train_ds, batch_size, IMAGE_SIZE, shuffle=True, augment=True)

    validate_ds = tf.data.Dataset.from_tensor_slices((validate_df.ImagePath.values, validate_df.classLabel.values))
    validate_ds = validate_ds.map(load_and_preprocess_from_path_label, num_parallel_calls=AUTOTUNE)
    validate_ds = prepare_tf_dataset(validate_ds, batch_size, IMAGE_SIZE, shuffle=True, augment=True)

    test_ds = tf.data.Dataset.from_tensor_slices((test_df.ImagePath.values, test_df.classLabel.values))
    test_ds = test_ds.map(load_and_preprocess_from_path_label, num_parallel_calls=AUTOTUNE)
    test_ds = prepare_tf_dataset(test_ds, batch_size, IMAGE_SIZE, shuffle=True, augment=False)
    return train_ds, validate_ds, test_ds


if __name__ == '__main__':
    data_save_dir = "/gypsum/eguide/projects/amuhebwa/RiversPrediction/ROADSV2_DATASETS"
    dir_name = "/work/amuhebwa_umass_edu/CURRENT_PROJECTS/roads"
    parser = argparse.ArgumentParser(description='File and Model Parameters')
    # parser.add_argument('--model_name', required=True)
    parser.add_argument('--datasplit_parcent', required=True)
    parser.add_argument('--no_of_classes', required=True)
    args = parser.parse_args()
    # model_name = args.model_name
    model_name = "ResNet50"
    datasplit_parcent = args.datasplit_parcent
    no_of_classes = args.no_of_classes
    datasplit_parcent = int(datasplit_parcent)
    no_of_classes = int(no_of_classes)
    print("Percentage of training data: ", datasplit_parcent)
    print("Number of classes : ", no_of_classes)
    roads_df_dict = {
        "GEP": "{}/GEP_images_labels_256_df.csv".format(dir_name),
        "PLANET": "{}/images_labels_256_df.csv".format(dir_name)
    }
    # python final_pretrained_models.py --datasplit_parcent=2 --no_of_classes=2

    data_source = "PLANET"
    batch_size = 8
    epochs = 50
    is_iid = False
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    layers_to_freeze = 3
    _split_percentage = 0.7

    class_weights = defined_weights.get(no_of_classes)

    # roads_dataset = pd.read_csv(image_labels_dict.get(IMAGE_SIZE))
    roads_dataset = pd.read_csv(roads_df_dict.get(data_source))

    '''
    If not i.i.d, the first datasplit_parcent should be 1, not 0
    This means datasplit_parcent = datasplit_parcent + 1, since the first parameter from sbatch --array == 0, not 1
    If we are doing IID training, we need a better split. 
    '''
    if is_iid:
        set_of_roads = sets_of_roads.get(0)
        _split_percentage = float(datasplit_parcent / 10)
    else:
        datasplit_parcent = datasplit_parcent + 1
        set_of_roads = sets_of_roads.get(datasplit_parcent)

    perc_of_roads = []
    all_accuracies = []
    all_f1_scores = []
    all_roc_auc_scores = []
    all_precision_scores = []
    majority_class_test = []
    majority_class_pred = []
    perc_majority_class_test = []
    perc_majority_class_pred = []
    for _, current_roads in enumerate(set_of_roads):
        print("*" * 50)
        print(current_roads)
        print("*" * 50)
        roads_count = len(current_roads)
        print("Number of roads used for training: {}".format(roads_count))
        train_dataset, validate_dataset, test_dataset = prepare_tensorflow_datasets(roads_dataset, no_of_classes, current_roads, _split_percentage, batch_size)
        metrics = ["sparse_categorical_accuracy"]
        model = pretrained_model(model_name, input_shape, no_of_classes, metrics, layers_to_freeze)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=0, patience=10, mode='auto',
                                                          restore_best_weights=True)
        history = model.fit(train_dataset, validation_data=validate_dataset, epochs=epochs, batch_size=batch_size,
                            class_weight=class_weights,
                            callbacks=[early_stopping],
                            verbose=1)

        # predicted_outputs = model.predict(test_dataset)
        predicted_labels = tf.argmax(model.predict(test_dataset), axis=1)
        prediction_prob = model.predict(test_dataset)
        prediction_prob = prediction_prob.astype(np.float32)
        actual_test_labels = np.concatenate([_labels for _images, _labels in test_dataset], axis=0)
        print(confusion_matrix(actual_test_labels, predicted_labels))

        num_epochs_completed = len(history.history['loss'])
        # accuracy, f1_score, precision_score, auc_roc_value = classification_metrics(actual_test_labels, predicted_labels, prediction_prob, no_of_classes)
        accuracy = calculate_accuracy_scores(predicted_labels, actual_test_labels)
        auc_roc_value = custom_auroc_calculator(predicted_labels, prediction_prob, actual_test_labels, no_of_classes)
        _F1, _precision = classification_metrics(actual_test_labels, predicted_labels)
        maj_class_test, maj_class_test_perc = percentage_majority_class(actual_test_labels)
        maj_class_pred, maj_class_pred_perc = percentage_majority_class(predicted_labels)

        perc_of_roads.append(datasplit_parcent)
        all_accuracies.append(accuracy)
        all_f1_scores.append(_F1)
        all_roc_auc_scores.append(auc_roc_value)
        all_precision_scores.append(_precision)
        majority_class_test.append(maj_class_test)
        majority_class_pred.append(maj_class_pred)
        perc_majority_class_test.append(maj_class_test_perc)
        perc_majority_class_pred.append(maj_class_pred_perc)

        print("MODEL NAME: ", model_name)
        print('---> Accuracy: {}'.format(accuracy))
        print("---> AUC_ROC: {}".format(auc_roc_value))
        print("+=" * 30)
        del model
        del history
        _ = gc.collect()
        K.clear_session()

    results_df = pd.DataFrame()
    results_df["Num_test_roads"] = perc_of_roads
    results_df['Accuracy'] = all_accuracies
    results_df['AUC_ROC'] = all_roc_auc_scores
    results_df['F1'] = all_f1_scores
    results_df['Precision'] = all_precision_scores
    results_df["test_majority_class"] = majority_class_test
    results_df["predict_majority_class"] = majority_class_pred
    results_df["test_majority_class_perc"] = perc_majority_class_test
    results_df["pred_majority_class_perc"] = perc_majority_class_pred
    if is_iid:
        name2save = f"{data_save_dir}/prediction_results/iid_RESNET_{data_source}_percent_{datasplit_parcent}_results_{no_of_classes}_classes.csv"
    else:
        name2save = f"{data_save_dir}/prediction_results/RESNET_{data_source}_percent_{datasplit_parcent}_results_{no_of_classes}_classes.csv"
    results_df.to_csv(name2save, index=False)
    print("Saved: ", name2save)
