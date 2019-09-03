import argparse
import ast
import logging
import os
import traceback
from typing import Dict

import model
import utils

DEPENDENCIES = ["nltk==3.4.4"]

CONTENT_TYPE_JSON = 'application/json'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO, )

logger = logging.getLogger(__name__)


def run_training(train:str, test:str, model_dir:str, output_data_dir:str, **hyperparams):
    logger.info("Running training with training data: '%s', test data: '%s', model dir: '%s',  output data: %s",
                train, test, model_dir, output_data_dir)

    # install_dependencies()

    model_pipeline = train_model(train, output_data_dir , model_dir, **hyperparams)

    # model.evaluate(model_pipeline, train_data, output_data_dir)

    # test_data = load_data(Path(test) / TEST_FILE_NAME)
    # model.evaluate(model_pipeline, test_data, output_data_dir)



def train_model(data_dir:str, output_dir:str, model_dir:str, **hyperparams:Dict):
    params = model.DEFAULT_HYPERPARAMETERS.copy()
    params.update(hyperparams)
    logger.info("Hyperparameters: %s", params)

    do_lower_case = True
    max_seq_length = 128

    model_pipeline, tokenizer = model.build_model(do_lower_case=do_lower_case)

    train_dataset = model.load_and_cache_examples(tokenizer,
                                                  data_dir,
                                                  max_seq_length,
                                                  evaluate=False)

    eval_dataset = model.load_and_cache_examples(tokenizer,
                                                 data_dir,
                                                 max_seq_length,
                                                 evaluate=True)


    logger.info("Start training")

    global_step, tr_loss = model.train(train_dataset, eval_dataset, model_pipeline, output_dir, **params)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    utils.write_model(model_pipeline, tokenizer, model_dir)


    results = model.evaluate_checkpoints(do_lower_case,
                                         output_dir,
                                         data_dir,
                                         max_seq_length)


    logger.info("Training done")

    return model_pipeline


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
