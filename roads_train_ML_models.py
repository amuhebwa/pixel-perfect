"""
Actual code for fine-tuning the models fro huggingface transformers
"""
import os
os.environ['TRANSFORMERS_CACHE'] = '/gypsum/eguide/projects/amuhebwa/RiversPrediction/ROADSV2_DATASETS/cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/gypsum/eguide/projects/amuhebwa/RiversPrediction/ROADSV2_DATASETS/cache'
import gc
import shortuuid
import tensorflow as tf
import wandb
from datasets import load_dataset
from sklearn.metrics import confusion_matrix
from transformers import AutoFeatureExtractor
from transformers import AutoModelForImageClassification
import argparse
from datasets import disable_caching
from config import sets_of_roads
from roads_utils import *

AUTOTUNE = tf.data.AUTOTUNE
_seed = 42
np.random.seed(_seed)
tf.random.set_seed(_seed)
torch.manual_seed(_seed)

os.environ["WANDB_DISABLED"] = "true"
wandb.init(mode="disabled")

disable_caching()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='File and Model Parameters')
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--datasplit_parcent', required=True)
    parser.add_argument('--no_of_classes', required=True)
    parser.add_argument('--set_index', required=True)
    args = parser.parse_args()
    dir_name = "/work/amuhebwa_umass_edu/CURRENT_PROJECTS/roads"
    data_save_dir = "/gypsum/eguide/projects/amuhebwa/RiversPrediction/ROADSV2_DATASETS"
    roads_df_dict = {
        "GEP": "{}/GEP_images_labels_256_df.csv".format(dir_name),
        "PLANET": "{}/images_labels_256_df.csv".format(dir_name)
    }

    IMAGE_SIZE = 224
    model_name = args.model_name
    datasplit_parcent = args.datasplit_parcent
    no_of_classes = args.no_of_classes
    datasplit_parcent = int(datasplit_parcent)
    no_of_classes = int(no_of_classes)
    set_index = args.set_index
    set_index = int(set_index)
    # This is too small. We need to figure out if this trade-ff is worth it
    batch_size = 8
    LEARNING_RATE = 2e-05
    EPOCHS = 15
    train_test_split = 0.7
    is_iid_split = True
    data_source = "PLANET"

    roads_dataset = pd.read_csv(roads_df_dict.get(data_source))

    """
    size index = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    increment each value by 1 since the array index passed from gypsum starts at 0
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    each of these values will be passed as a percentage to split i.i.d 
    data i.e., 10%, 20%, 30%, ...
    """
    if is_iid_split:
        _parcent = (set_index + 1)
        train_test_split = float(_parcent/10) # convert it to percentage
        heldout_training_roads = sets_of_roads.get(0)  # empty roads that activate the iid option
    else:
        heldout_training_roads = sets_of_roads.get(datasplit_parcent)

    assert model_name in [*selected_models_dict.keys()]
    model_checkpoint = selected_models_dict.get(model_name)
    assert model_checkpoint in [*selected_models_dict.values()]
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

    """
    for each data split k%, pick 1 to 10th set, run a single job and save it to a text file
    """
    current_train_roads = heldout_training_roads[set_index]

    print('<->' * 50)
    print("Model Name: ", model_name)
    print('CURRENT ROADS USED FOR TRAINING')
    print(current_train_roads)
    print("DATE SPLIT PARCENT: {}".format(train_test_split))
    print('<->' * 50)
    train_dataset, validate_dataset, test_dataset = prepare_datasets_splits(
        roads_dataset, no_of_classes, current_train_roads, train_test_split)

    session_id = shortuuid.uuid()  # created for this specific experiment
    current_session_folder = "{}/TEMP_DATASETS_IMAGES".format(data_save_dir)
    current_data_path = create_folder(current_session_folder, session_id)
    print('CURRENT SESSION FOLDER: {}'.format(current_data_path))

    train_data_path = copy_files(current_data_path, "train", train_dataset)
    validate_data_path = copy_files(current_data_path, "validate", validate_dataset)
    test_data_path = copy_files(current_data_path, "test", test_dataset)

    train_ds = load_dataset("imagefolder", data_dir=train_data_path)
    validate_ds = load_dataset("imagefolder", data_dir=validate_data_path)
    test_ds = load_dataset("imagefolder", data_dir=test_data_path)

    train_ds = train_ds['train']
    validate_ds = validate_ds['train']
    test_ds = test_ds['train']

    labels = train_ds.features["label"].names
    label2id, id2label = create_label_dicts(no_of_classes)

    preprocess_train_dataset, preprocess_test_dataset = custom_data_transforms(feature_extractor)

    train_ds.set_transform(preprocess_train_dataset)
    validate_ds.set_transform(preprocess_train_dataset)
    test_ds.set_transform(preprocess_test_dataset)

    model = AutoModelForImageClassification.from_pretrained(model_checkpoint, label2id=label2id, id2label=id2label,
                                                            ignore_mismatched_sizes=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    model_name = model_checkpoint.split("/")[-1]
    # improving performance: https://huggingface.co/docs/transformers/v4.18.0/en/performance
    training_args = TrainingArguments(
        f"{data_save_dir}/dir_vision_models/{model_name}-{session_id}",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        eval_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,  # 1e-6 (seems to be working best), #0.01,
        fp16=True,
        warmup_ratio=0.1,
        logging_steps=10,  # initially 10,
        eval_steps=1,  # initially not there
        logging_strategy="epoch",  # initially not there
        # do_train=True,  # initially not there
        # do_eval=True,  # initially not there
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=_seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=validate_ds,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    # Extra call back to log training accuracy
    trainer.add_callback(CustomCallback(trainer))
    train_results = trainer.train()

    outputs = trainer.predict(test_ds)
    _accuracy = outputs.metrics['test_accuracy']
    _auroc = custom_auroc_calculator(outputs, no_of_classes)
    _F1, _precision = classification_metrics(outputs)
    _major_class, _perc_major_class = percentage_majority_class(outputs)
    labels_actual = outputs.label_ids
    labels_predicted = outputs.predictions.argmax(1)

    print("Data Source: {}".format(data_source))
    print("Model Name: {}".format(model_name))
    if is_iid_split:
        print("Percent of Data Split: {}".format(train_test_split))
    else:
        print("Percent of heldout roads used for training: {}".format(datasplit_parcent))
    print('confusion matrix')
    print(confusion_matrix(labels_actual, labels_predicted))
    print("Accuracy: {}".format(_accuracy))
    print("AUROC: {}".format(_auroc))

    delete_folder(current_data_path)
    train_ds.cleanup_cache_files()
    validate_ds.cleanup_cache_files()
    test_ds.cleanup_cache_files()
    del model, outputs
    gc.collect()
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()

    if is_iid_split:
        data_source = 'iid_{}'.format(data_source)
        save_name = f"{data_save_dir}/prediction_results/{data_source}_{str(datasplit_parcent*100)}trainParcent0{_parcent}_{model_name}_classes_0{str(no_of_classes)}_set0{str(set_index)}.txt"
        datasplit_parcent = (_parcent*10)
    else:
        save_name = "{}/prediction_results/{}_trainParcent0{}_{}_classes_0{}_results_set0{}.txt".format(data_save_dir, data_source, str(datasplit_parcent), model_name, str(no_of_classes), str(set_index))

    results_dict = {"model_name": model_name,
                    "no_of_classes": no_of_classes,
                    "is_data_iid": is_iid_split,
                    "Accuracy": _accuracy,
                    "AUROC": _auroc,
                    "F1": _F1,
                    "Precision": _precision,
                    "HeldoutRoads4Train": datasplit_parcent,
                    "MajorClass": _major_class,
                    "PercMajorclass": _perc_major_class}

    text_file = open(save_name, "w")
    n = text_file.write(str(results_dict))
    text_file.close()
    print('Saved: {}'.format(save_name))
    print('----------> Done <----------------------')
