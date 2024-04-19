"""
Some of these functions were directly copied from cnn_utils.py
Clen this up later.
"""
import os
import shutil
from typing import List
from copy import deepcopy
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomResizedCrop, Resize, ToTensor
import torchvision.transforms as transforms
import torch
import evaluate
from scipy.special import softmax
from sklearn import metrics
import Augmentor as AG
import datasets
import time
from transformers import ViTFeatureExtractor
from tensorflow import keras
from tensorflow.keras import layers
from transformers import ViTForImageClassification, DeiTForImageClassification, ConvNextForImageClassification
from transformers import DefaultDataCollator
import glob
from transformers import TrainingArguments, Trainer, TrainerCallback
from datasets import disable_caching

disable_caching()

metric = evaluate.load("accuracy", load_from_cache_file=False)
auroc_metric = evaluate.load("roc_auc", load_from_cache_file=False)

class CustomCallback(TrainerCallback):
    """
    A custom callback that logs all evaluation metrics at each epoch.
    credit: https://stackoverflow.com/questions/67457480/how-to-get-the-accuracy-per-epoch-or-step-for-the-huggingface-transformers-train
    """

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy


def percentage_majority_class(outputs):
    predicted_labels = outputs.predictions.argmax(1)
    uniques, counts = np.unique(predicted_labels, return_counts=True)
    percentages = dict(zip(uniques, counts * 100 / len(predicted_labels)))
    max_class_perc = max(percentages.values())
    max_class = list([k for k, v in percentages.items() if v == max_class_perc]).pop()
    return max_class, max_class_perc

def classification_metrics(outputs):
    actual_labels = outputs.label_ids
    predicted_labels = outputs.predictions.argmax(1)
    scores_F1 = metrics.f1_score(actual_labels, predicted_labels, average='weighted')
    scores_precision = metrics.precision_score(actual_labels, predicted_labels, average='weighted', zero_division=0)
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
        roc_auc = metrics.roc_auc_score(new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc
    mean_roc = np.mean([*roc_auc_dict.values()])
    # return roc_auc_dict
    return mean_roc


def custom_auroc_calculator(outputs, no_of_classes):
    labels_actual = outputs.label_ids
    labels_predicted = outputs.predictions.argmax(1)
    prediction_proba = softmax(outputs.predictions, axis=1)
    auc_roc_value = -999
    if no_of_classes == 2:
        pred_proba = prediction_proba[:, 1]
        auc_roc_value = auroc_metric.compute(references=labels_actual, prediction_scores=pred_proba)
        auc_roc_value = auc_roc_value['roc_auc']
    else:
        auc_roc_value = roc_auc_score_multiclass(labels_actual, labels_predicted)
    return auc_roc_value


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

def sample_equal_labels(number_of_samples, temp_df):
    _, unique_counts = np.unique(temp_df.classLabel.values, return_counts=True)
    number_of_samples = min(number_of_samples, np.max(unique_counts)//2)
    balanced_df = temp_df.groupby("classLabel",
                                  as_index=False,
                                  group_keys=False).apply(lambda s: s.sample(number_of_samples, replace=True))
    # balanced_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    balanced_df = balanced_df.reset_index(drop=True)
    return balanced_df

def smart_split_data(df: pd.DataFrame, heldout_train_roads: List, traindata_parcent: np.float64):
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

    print('-'*50)
    print('Train Splits')
    print(train_df.groupby('classLabel').count())
    print('-'*50)
    return train_df, validate_df, test_df


def prepare_datasets_splits(roads2015_dataset: pd.DataFrame, no_of_classes: int, test_road_names: np.array, traindata_parcent):
    roads2015_dataset = convert_labels_to_classes(no_of_classes, roads2015_dataset)
    train_df, validate_df, test_df = smart_split_data(roads2015_dataset, test_road_names, traindata_parcent)
    return train_df, validate_df, test_df


def create_folder(base_dir: str, folder_name: str) -> str:
    folder_path = "{}/{}".format(base_dir, folder_name)
    isExist = os.path.exists(folder_path)
    if not isExist:
        os.makedirs(folder_path)
        print("Created new temporary directory: {}".format(folder_path))
    return folder_path


def delete_folder(current_data_path: str) -> None:
    isExist = os.path.exists(current_data_path)
    isADir = os.path.isdir(current_data_path)
    if isExist and isADir:
        try:
            shutil.rmtree(current_data_path)
            print("SUCCESSFULLY DELETED FOLDER: {}".format(current_data_path))
        except FileExistsError:
            print("could not remove the folder")
            pass


def copy_files(parent_folder: str, data_split_name: str, current_df: pd.DataFrame) -> str:
    full_path = create_folder(parent_folder, data_split_name)
    image_files = current_df["ImagePath"].values
    class_labels = current_df["classLabel"].values
    for im_path, im_label in zip(image_files, class_labels):
        road_class_folder = "roadclass0{}".format(im_label)
        destination_folder = create_folder(full_path, road_class_folder)
        shutil.copy(im_path, destination_folder)
    print("Finished copy : {} dataset".format(data_split_name))
    return full_path

def create_label_dicts(no_of_classes):
    label2id, id2label = None, None
    if no_of_classes == 2:
        label2id = {'roadclass00': 0, 'roadclass01': 1}
        id2label = {0: 'roadclass00', 1: 'roadclass01'}
    else:
        label2id = {'roadclass00': 0, 'roadclass01': 1, 'roadclass02': 2, 'roadclass03': 3, 'roadclass04': 4}
        id2label = {0: 'roadclass00', 1: 'roadclass01', 2: 'roadclass02', 3: 'roadclass03', 4: 'roadclass04'}

    return (label2id, id2label)


def custom_data_transforms(feature_extractor):
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    train_data_transforms = Compose([
        Resize(feature_extractor.size),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        Resize(size=(224, 224)),
        ToTensor(),
        normalize,
    ])

    test_data_transforms = Compose([
        Resize(feature_extractor.size),
        Resize(size=(224, 224)),

        ToTensor(),
        normalize,
    ])

    def preprocess_train_ds(current_batch):
        current_batch["pixel_values"] = [train_data_transforms(image.convert("RGB")) for image in current_batch["image"]]
        return current_batch

    def preprocess_test_ds(current_batch):
        current_batch["pixel_values"] = [test_data_transforms(image.convert("RGB")) for image in current_batch["image"]]
        return current_batch
    return preprocess_train_ds, preprocess_test_ds


def compute_metrics(prediction_metrics):
    predictions = np.argmax(prediction_metrics.predictions, axis=1)
    return metric.compute(predictions=predictions, references=prediction_metrics.label_ids)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

selected_models_dict = {
    "ViT": "google/vit-large-patch16-224",
    "Swin": "microsoft/swin-base-patch4-window7-224-in22k",
    "ConvNext": "facebook/convnext-large-224-22k",
    "DeiT": "facebook/deit-base-patch16-224",
}
