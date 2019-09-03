import csv
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

import model

DATA_FOLDER = "data"

DATE_UNIT = 'ns'

JSON_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%Z"

MODEL_FILE_NAME = "model.joblib"

logger = logging.getLogger(__name__)


def write_dataframe(df: pd.DataFrame, path: Path) -> Path:
    logger.info("Writing DataFrame to: '%s'", path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # prevents incorrect newlines under windows
    with open(str(path), mode='w', newline='\n', encoding="UTF-8") as fp:
        df.to_csv(fp,
                  index=False,
                  header=True,
                  mode="w",
                  quoting=csv.QUOTE_ALL)

    return path


def load_dataframe(path: str) -> pd.DataFrame:
    logger.info("Loading data from: '%s'", path)

    dataset = pd.read_csv(str(path),
                          header=0,
                          parse_dates=False,
                          quoting=csv.QUOTE_ALL)

    convert_dates(dataset)
    logger.info("Loading done. Dataframe shape %s", dataset.shape)
    return dataset


def convert_dates(dataset: pd.DataFrame):
    for column_name in dataset.columns:
        if column_name.endswith("_date"):
            dataset[column_name] = pd.to_datetime(dataset[column_name], utc=True, unit=DATE_UNIT)


def write_model(model, tokenizer, model_dir: str):
    logger.info("Writing model to: '%s'", model_dir)

    logger.info("Saving model checkpoint to %s", model_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(model_dir, 'training_args.bin'))


def load_model(model_dir: str, do_lower_case:bool):
    logger.info("Loading model from %s", model_dir)
    # Load a trained model and vocabulary that you have fine-tuned
    model_pipeline = model.MODEL_CLASS.from_pretrained(model_dir)
    tokenizer = model.TOKENIZER_CLASS.from_pretrained(model_dir, do_lower_case=do_lower_case)
    model_pipeline.to(model.get_device())

    return model_pipeline, tokenizer


def to_json(df: pd.DataFrame) -> str:
    return df.to_json(orient="records", date_unit=DATE_UNIT)


def from_json(input_data: str) -> pd.DataFrame:
    df = pd.read_json(input_data, orient="records", date_unit=DATE_UNIT, precise_float=True)
    convert_dates(df)
    return df


def get_target_file_path(file_name, use_temp) -> Path:
    if use_temp:
        return Path(tempfile.TemporaryDirectory().name) / file_name

    root_path = Path(__file__).parent.parent
    return root_path / DATA_FOLDER / datetime.now().strftime("%Y-%m-%dT%H-%M_%S") / file_name


