import argparse
import ast
import logging
import os
import traceback
from typing import Dict

import torch
from torch.utils.data import TensorDataset

import model
import utils
from model import MODEL_NAME, logger
from transformer_utils import DataProcessor, convert_examples_to_features

DEPENDENCIES = ["nltk==3.4.4"]

CONTENT_TYPE_JSON = 'application/json'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO, )

logger = logging.getLogger(__name__)


def run_training(train:str, test:str, model_dir:str, output_data_dir:str, **hyperparams):
    logger.info("Running training with training data: '%s', test data: '%s', model dir: '%s',  output data: %s",
                train, test, model_dir, output_data_dir)

    # install_dependencies()

    params = model.DEFAULT_HYPERPARAMETERS.copy()
    params.update(hyperparams)
    logger.info("Hyperparameters: %s", params)

    model_pipeline, tokenizer = train_model(train,
                                            test,
                                            output_data_dir ,
                                            model_dir,
                                            **params)

    eval_dataset = load_and_cache_examples(tokenizer,
                                           test,
                                           params["max_seq_length"],
                                           evaluate=True)

    results = model.evaluate_checkpoints(eval_dataset, output_data_dir, eval_all_checkpoints=True)


def train_model(train_data_dir:str, test_data_dir: str, output_dir:str, model_dir:str, **hyperparams:Dict):

    do_lower_case = hyperparams.pop("do_lower_case")
    max_seq_length = hyperparams.pop("max_seq_length")

    model_pipeline, tokenizer = model.build_model(do_lower_case=do_lower_case,
                                                  num_labels=(len(DataProcessor().get_labels())))

    train_dataset = load_and_cache_examples(tokenizer,
                                            train_data_dir,
                                            max_seq_length,
                                            evaluate=False)

    logger.info("Start training")

    eval_dataset = load_and_cache_examples(tokenizer,
                                           test_data_dir,
                                           max_seq_length,
                                           evaluate=True)


    model.train(model_pipeline, tokenizer, train_dataset, eval_dataset, output_dir, **hyperparams)

    utils.write_model(model_pipeline, tokenizer, model_dir)

    logger.info("Training done")

    return model_pipeline, tokenizer


def load_and_cache_examples(tokenizer, data_dir, max_seq_length, evaluate=False):
    # Load data features from cache or dataset file

    cached_features_file = get_cache_file_path(data_dir, max_seq_length, evaluate)

    if os.path.exists(cached_features_file):
        features = load_cached_features(cached_features_file)

    else:
        features = load_features(tokenizer, data_dir, max_seq_length, evaluate)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids, all_input_mask, all_segment_ids, all_label_ids = convert_to_tensors(features)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def convert_to_tensors(features):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    return all_input_ids, all_input_mask, all_segment_ids, all_label_ids


def load_cached_features(cached_features_file):
    logger.info("Loading features from cached file %s", cached_features_file)
    features = torch.load(cached_features_file)
    return features


def get_cache_file_path(data_dir, max_seq_length, evaluate):
    cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, MODEL_NAME.split('/'))).pop(),
        str(max_seq_length)))
    return cached_features_file


def load_features(tokenizer, data_dir, max_seq_length, evaluate):
    processor = DataProcessor()
    logger.info("Creating features from dataset file at %s", data_dir)
    label_list = processor.get_labels()
    examples = processor.get_dev_examples(data_dir) if evaluate else processor.get_train_examples(data_dir)
    features = build_features(examples, label_list, max_seq_length, tokenizer)
    return features


def build_features(examples, label_list, max_seq_length, tokenizer):
    features = convert_examples_to_features(examples,
                                            label_list,
                                            max_seq_length,
                                            tokenizer,
                                            "classification",
                                            cls_token_at_end=False,
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=0,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=False,
                                            pad_on_left=False,
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=0,
                                            )
    return features


def _infer_dtype(value: str):
    try:
        return int(value)
    except:
        pass

    try:
        return float(value)
    except:
        pass

    try:
        return ast.literal_eval(value) #tuple
    except:
        pass

    return value


def _parse_cli_arguments():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    _, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith("--"):
            parser.add_argument(arg)

    args = vars(parser.parse_args())

    for k, v in args.items():
        args[k] = _infer_dtype(v)

    return args


if __name__ =='__main__':

    try:
        run_training(**_parse_cli_arguments())
    except Exception as e:
        logger.exception("Training failed")
        traceback.print_exc()
        raise e


