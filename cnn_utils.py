import evaluate
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.special import softmax
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from tensorflow.keras import layers
from typing import Dict
from typing import List

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = 224
learning_rate = 0.00001

epochs = 10
batch_size = 8
auroc_metric = evaluate.load("roc_auc")

"""
fp16 = True
if fp16:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
"""


# resampling to balance the classes.
# https://www.tensorflow.org/guide/data#rejection_resampling


def class_func(features, label):
    return label


def balanced_class_labels(unbalanced_ds, target_dist, _batch_size):
    resample_ds = (unbalanced_ds.unbatch().rejection_resample(class_func, target_dist=target_dist).batch(_batch_size))
    balanced_ds = resample_ds.map(lambda extra_label, features_and_label: features_and_label)
    return balanced_ds


def percentage_majority_class(predicted_labels):
    """
    Since we have IRI values clustering towards thresholds when converting to labels, we need to calculate the percentage of the majority class to figure out
    which class gets favored the most by the model
    """
    uniques, counts = np.unique(predicted_labels, return_counts=True)
    percentages = dict(zip(uniques, counts * 100 / len(predicted_labels)))
    max_class_perc = max(percentages.values())
    max_class = list([k for k, v in percentages.items() if v == max_class_perc]).pop()
    return max_class, max_class_perc


def roc_auc_score_multiclass(actual_class_labels, predicted_class_labels, average="macro"):
    """
    Helper function to calculate roc_auc_score for multi-class classification.
    There's no direct methods for doing this so we have to use a one vs. all workaround.
    credit: https://stackoverflow.com/questions/39685740/calculate-sklearn-roc-auc-score-for-multi-class
    """
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
    return mean_roc


def classification_metrics(actual_labels, predicted_labels, prediction_scores, no_of_classes):
    """
    Helper function to calculate different classification metrics;
    accuracy, f1_score, precision_score, auc_roc_value
    """
    print(confusion_matrix(actual_labels, predicted_labels))
    _f1_score = f1_score(actual_labels, predicted_labels, average='weighted')
    _precision_score = precision_score(actual_labels, predicted_labels, average='weighted', zero_division=0)
    accuracy = accuracy_score(actual_labels, predicted_labels)
    if no_of_classes == 2:
        try:
            pred_probability = softmax(prediction_scores, axis=1)
            pred_proba = pred_probability[:, 1]
            auc_roc_value = auroc_metric.compute(references=actual_labels, prediction_scores=pred_proba)
            auc_roc_value = auc_roc_value.get('roc_auc')
            # auc_roc_value = metrics.roc_auc_score(actual_labels, pred_probability)
        except ValueError:
            auc_roc_value = -999
    elif no_of_classes > 2:
        try:
            auc_roc_value = roc_auc_score_multiclass(actual_labels, predicted_labels)
        except ValueError:
            auc_roc_value = -999
    else:
        auc_roc_value = -999
    return accuracy, _f1_score, _precision_score, auc_roc_value


def get_activation_fn(no_of_classses: int):
    """
    Select activation function based on the number of classes
    """
    if no_of_classses == 2:
        return "sigmoid"
    else:
        return "softmax"


def roads5ClassLabel(iriValue: np.float64, allowNeg=False):
    """
    Convert IRI values to 5 classes
    """
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
    """
    Convert IRI values to binary classes
    """
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
    """
    parent class for converting IRI values (continous) to discrete labels (classes)
    """
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
    """
    Since we have imbalanced classes, we need to calculate the class weights to balance the classes.
    We will pass these to the CNN model to help it learn the minority classes better.
    """
    y = df['classLabel'].values
    class_wghts = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    class_weights = dict(zip(np.unique(y), class_wghts))
    for key in class_weights:
        if class_weights[key] < 1.0:
            class_weights[key] = 1.0
    return class_weights


def sample_equal_labels(number_of_samples: int, temp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sample equal number of samples from each class to balance the classes
    This is for testing to ensure that the model is not biased towards the majority class
    :return:
    """
    _, unique_counts = np.unique(temp_df.classLabel.values, return_counts=True)
    number_of_samples = min(number_of_samples, np.min(unique_counts))
    balanced_df = temp_df.groupby("classLabel",
                                  as_index=False,
                                  group_keys=False).apply(lambda s: s.sample(number_of_samples, replace=True))
    # balanced_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    balanced_df = balanced_df.reset_index(drop=True)
    return balanced_df


def smart_split_data(df: pd.DataFrame, heldout_train_roads: List, traindata_parcent: np.float64, no_of_classes: int):
    """
    helper function to Split the data into training, validation and testing datasets
    """
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
        train_df, test_df = train_test_split(df, train_size=traindata_parcent, random_state=1234)
        train_df, validate_df = train_test_split(train_df, train_size=0.80, random_state=1234)

    train_df = sample_equal_labels(no_of_samples, train_df)
    validate_df = sample_equal_labels(no_of_samples, validate_df)
    test_df = sample_equal_labels(no_of_samples, test_df)
    train_df = train_df.sample(frac=1.0, replace=False).reset_index(drop=True)
    validate_df = validate_df.sample(frac=1.0, replace=False).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, replace=False).reset_index(drop=True)
    return train_df, validate_df, test_df


def prepare_tensorflow_datasets(roads2015_dataset: pd.DataFrame, no_of_classes: int, test_road_names: np.array, traindata_parcent):
    """
    convert image-> labels pairs to tensorflow datasets.
    Pretty sure there's a better way to do this but I'm not sure how to do it yet.
    This should work for now.
    Note: Look up tensorflow documentation to see if there's a better way to do this.
    """
    roads2015_dataset = convert_labels_to_classes(no_of_classes, roads2015_dataset)
    train_df, validate_df, test_df = smart_split_data(roads2015_dataset, test_road_names, traindata_parcent, no_of_classes)
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
