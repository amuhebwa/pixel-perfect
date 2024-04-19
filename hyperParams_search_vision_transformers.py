"""
code for searching the best combination of hyperparameters for the Vision Transformer model.
you need to have the wandb library installed in order to visualize the logs in the browser.
"""

import os
os.environ['TRANSFORMERS_CACHE'] = '/gypsum/eguide/projects/amuhebwa/RiversPrediction/ROADSV2_DATASETS/cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/gypsum/eguide/projects/amuhebwa/RiversPrediction/ROADSV2_DATASETS/cache'
import gc
import os
import shutil

import evaluate
import numpy as np
import pandas as pd
import shortuuid
import tensorflow as tf

import torch
from datasets import load_dataset
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import AutoFeatureExtractor
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForImageClassification
import wandb

experiment_name = "Roads_HyperPramSearch_2classes_20Runs"
os.environ['WANDB_PROJECT']= experiment_name
os.environ['WANDB_LOG_MODEL']="true"
AUTOTUNE = tf.data.AUTOTUNE
metric = evaluate.load("accuracy")


torch.cuda.empty_cache()
gc.collect()


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
        roc_auc = metrics.roc_auc_score(new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc
    mean_roc = np.mean([*roc_auc_dict.values()])
    # return roc_auc_dict
    return mean_roc


def decode_img(img):
    img = tf.io.decode_png(img, channels=3)
    return tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])


def load_and_preprocess_image(path):
    _img = tf.io.read_file(path)
    _img = decode_img(_img)
    # _img = tf.image.transpose(_img, [2, 0, 1])
    _img = tf.image.transpose(_img)
    return _img


def load_and_preprocess_from_path_label(path, _label):
    return load_and_preprocess_image(path), _label


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


def sample_equal_labels(number_of_samples, temp_df):
    _, unique_counts = np.unique(temp_df.classLabel.values, return_counts=True)
    number_of_samples = min(number_of_samples, np.min(unique_counts))
    balanced_df = temp_df.groupby("classLabel",
                                  as_index=False,
                                  group_keys=False).apply(lambda s: s.sample(number_of_samples, replace=True))
    balanced_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    balanced_df = balanced_df.reset_index(drop=True)
    return balanced_df


def smart_split_data(df, test_roads):
    no_of_samples = 2000
    all_road_names = df.RoadName.unique()
    if set(test_roads).issubset(set(all_road_names)):  # full_held_roads
        test_df = df[df['RoadName'].isin(test_roads)].reset_index(drop=True)
        train_df = df[~df.index.isin(test_df.index.values)]
        train_df, validate_df = train_test_split(train_df, train_size=0.80, random_state=1234)
    else:
        df = df.sample(n=df.shape[0], replace=False, random_state=1).reset_index(drop=True)
        train_df, test_df = train_test_split(df, train_size=0.8, random_state=1234)
        train_df, validate_df = train_test_split(train_df, train_size=0.80, random_state=1234)

    train_df = sample_equal_labels(no_of_samples, train_df)
    validate_df = sample_equal_labels(no_of_samples, validate_df)
    test_df = sample_equal_labels(no_of_samples, test_df)
    train_df = train_df.sample(frac=1.0, replace=False).reset_index(drop=True)
    validate_df = validate_df.sample(frac=1.0, replace=False).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, replace=False).reset_index(drop=True)
    return train_df, validate_df, test_df


def prepare_tensorflow_datasets(roads2015_dataset: pd.DataFrame, no_of_classes: int, test_road_names: np.array,
                                batch_size: int):
    roads2015_dataset = convert_labels_to_classes(no_of_classes, roads2015_dataset)
    train_df, validate_df, test_df = smart_split_data(roads2015_dataset, test_road_names)
    return train_df, validate_df, test_df


def create_folder(folder_name: str) -> str:
    """
    Create a temporary folder for a specific class.
    Admittedly, this is not the best way to handle this, but it works for now.
    Note: Talk to Gabe about a better way to handle this.
    """
    folder_path = "{}/TEMP_DATASETS_IMAGES/{}".format(dir_name, folder_name)
    isExist = os.path.exists(folder_path)
    if not isExist:
        os.makedirs(folder_path)
        print("Created new temporary directory: {}".format(folder_path))
    return folder_path


def delete_folder(folder_name: str) -> None:
    """
    Once you finish training, delete the temporary folder to save space.
    """
    folder_path = "{}/TEMP_DATASETS_IMAGES/{}".format(dir_name, folder_name)
    isExist = os.path.exists(folder_path)
    isADir = os.path.isdir(folder_path)
    if isExist and isADir:
        try:
            shutil.rmtree(folder_path)
            print("SUCCESSFULLY DELETED FOLDER: {}".format(folder_path))
        except FileExistsError:
            print("couldnot remove the folder")
            pass


def copy_files(parent_folder: str, data_split_name: str, current_df: pd.DataFrame) -> str:
    """
    Copy the images to the temporary folder for training.
    Each folder corresponds to a specific class.
    """
    folder_name = "{}/{}".format(parent_folder, data_split_name)
    full_path = create_folder(folder_name)
    image_files = current_df["ImagePath"].values
    class_labels = current_df["classLabel"].values
    for im_path, im_label in zip(image_files, class_labels):
        road_class_folder = "{}/roadclass0{}".format(folder_name, im_label)
        destination_folder = create_folder(road_class_folder)
        shutil.copy(im_path, destination_folder)
    print("Finished copy : {} dataset".format(data_split_name))
    return full_path

if __name__ == "__main__":
    dir_name = "/work/amuhebwa_umass_edu/CURRENT_PROJECTS/roads"
    roads_dataset = pd.read_csv("{}/images_labels_256_df.csv".format(dir_name))
    IMAGE_SIZE = 224
    model_name = "VISION_TRANSFORMER"
    EPOCHS = 10
    batch_size = 8
    LEARNING_RATE = 5e-5
    no_of_classes = 2
    # num_train_epochs = 2

    current_roads = ["None_A", "None_B", "None_C"]
    train_dataset, validate_dataset, test_dataset = prepare_tensorflow_datasets(roads_dataset, no_of_classes,
                                                                                current_roads, batch_size)
    """
    1. Create a folder
    2. create  subfolders corresponding to classes
    3. load images where the folder names correspond to class labels
    4. train model as required.
    5. test the model
    6. Delete the folder when you are done.
    """

    session_id = shortuuid.uuid()  # created for this specific experiment
    create_folder(session_id)

    train_data_path = copy_files(session_id, "train", train_dataset)
    validate_data_path = copy_files(session_id, "validate", validate_dataset)
    test_data_path = copy_files(session_id, "test", test_dataset)

    train_ds = load_dataset("imagefolder", data_dir=train_data_path)
    validate_ds = load_dataset("imagefolder", data_dir=validate_data_path)
    test_ds = load_dataset("imagefolder", data_dir=test_data_path)

    labels = train_ds["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    model_checkpoint = "google/vit-base-patch16-224-in21k"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    train_transforms = Compose(
        [
            RandomResizedCrop(feature_extractor.size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    val_transforms = Compose(
        [
            Resize(feature_extractor.size),
            CenterCrop(feature_extractor.size),
            ToTensor(),
            normalize,
        ]
    )


    def preprocess_train(example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [
            train_transforms(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch


    def preprocess_val(example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch


    train_ds = train_ds['train']
    validate_ds = validate_ds['train']
    test_ds = test_ds['train']

    train_ds.set_transform(preprocess_train)
    validate_ds.set_transform(preprocess_val)
    test_ds.set_transform(preprocess_val)

    def model_init():
        model = AutoModelForImageClassification.from_pretrained(
            model_checkpoint,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )
        return model


    model_name = model_checkpoint.split("/")[-1]
    # improving performance: https://huggingface.co/docs/transformers/v4.18.0/en/performance


    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)


    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

        # method
    sweep_config = {
        'method': 'random'
    }

    parameters_dict = {
        'epochs': {
            'value': 20
            },
        'batch_size': {
            'values': [8, 16, 32, 64,]
            },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-1,
        },
        'weight_decay': {
            'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        },
    }
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project=experiment_name)
    def train(config=None):
        with wandb.init(config=config):
            # set sweep configuration
            config = wandb.config
            training_args = TrainingArguments(
                f"{model_name}-{session_id}",
                remove_unused_columns=False,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=config.learning_rate,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
                per_device_eval_batch_size=16,
                num_train_epochs=config.epochs,
                weight_decay=config.weight_decay,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
            )

            # define training loop
            trainer = Trainer(
                # model,
                model_init=model_init,
                args=training_args,
                data_collator=collate_fn,
                train_dataset=train_ds,
                eval_dataset=validate_ds,
                compute_metrics=compute_metrics
            )
        # start training loop
        trainer.train()

    wandb.agent(sweep_id, train, count=20)

    """
    WHEN DONE WITH EXPERIMENTS, WE DELETE THE FOLDER TO SAVE SPACE
    """
    delete_folder(session_id)
    torch.cuda.empty_cache()
    gc.collect()
