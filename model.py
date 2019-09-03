import glob
import logging
import os
from datetime import datetime
from enum import Enum

import numpy as np
import torch
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from sklearn import metrics
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from transformer_utils import convert_examples_to_features, DataProcessor

MODEL_CLASS = BertForSequenceClassification

TOKENIZER_CLASS = BertTokenizer

MODEL_NAME = "bert-base-german-cased"
MODEL_NAME = "bert-base-uncased"


logger = logging.getLogger(__name__)

WEIGHTS_NAME = "pytorch_model.bin"


class Column(Enum):
    HAS_ATTACHMENTS = "has_attachments"
    CONTENT = "content"
    SUBJECT = "subject"
    SENDER = "sender"
    CHANNEL = "channel_id"
    CHANNEL_NAME = "channel_name"
    LANGUAGE = "language"
    RECEIVED_DATE = "received_date"
    PROCESSING_DATE = "processing_date"
    LABEL = "code_a"
    CODE_B = "code_b"


class Dataset(Enum):
    Train = "train"
    Validation = "validation"


DEFAULT_HYPERPARAMETERS = {
}

def evaluate_checkpoints(do_lower_case, output_dir, data_dir, max_seq_length):
    results = {}
    eval_all_checkpoints = True

    tokenizer = TOKENIZER_CLASS.from_pretrained(output_dir, do_lower_case=do_lower_case)
    eval_dataset = load_and_cache_examples(tokenizer, data_dir, max_seq_length, evaluate=True)

    checkpoints = [output_dir]
    if eval_all_checkpoints:
        checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
        logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging

    logger.info("Evaluate the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:
        global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
        model = MODEL_CLASS.from_pretrained(checkpoint)
        model.to(get_device())

        result = evaluate(eval_dataset, model, output_dir, prefix=global_step)
        result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
        results.update(result)

    return results


def evaluate(eval_dataset, model, eval_output_dir, prefix=""):
    per_gpu_eval_batch_size = 8

    results = {}

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    n_gpu = torch.cuda.device_count()
    eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)

    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    device = get_device()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)

    result = compute_metrics(preds, out_label_ids)
    results.update(result)

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def build_model(do_lower_case):
    processor = DataProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    config = BertConfig.from_pretrained(MODEL_NAME, num_labels=num_labels, finetuning_task="MRPC")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=do_lower_case)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    model.to(get_device())

    return model, tokenizer


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_cache_examples(tokenizer, data_dir, max_seq_length, evaluate=False):
    processor = DataProcessor()
    # Load data features from cache or dataset file

    cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, MODEL_NAME.split('/'))).pop(),
        str(max_seq_length)))

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)

    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(data_dir) if evaluate else processor.get_train_examples(data_dir)

        features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, "classification",
                                                cls_token_at_end=False,
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=False,
                                                pad_on_left=False,
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=0,
                                                )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset



def train(train_dataset, eval_dataset, model, output_dir, **hyperparams):
    """ Train the model """

    per_gpu_train_batch_size = 1
    max_steps = -1
    num_train_epochs = 3.0
    gradient_accumulation_steps = 1
    learning_rate = 2e-5
    adam_epsilon = 1e-8
    warmup_steps = 0
    max_grad_norm = 1.0
    weight_decay = 0.0

    logging_steps = 50
    evaluate_during_training = True
    save_steps = 50

    n_gpu = torch.cuda.device_count()
    train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)


    if max_steps > 0:
        t_total = max_steps
        num_train_epochs = max_steps // (len(train_dataloader) // gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                train_batch_size * gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(num_train_epochs), desc="Epoch", disable=False)
    # set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    tb_writer = SummaryWriter(log_dir=get_log_dir(output_dir))


    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(get_device()) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if logging_steps > 0 and global_step % logging_steps == 0:
                    # Log metrics

                    log_metrics(eval_dataset, evaluate_during_training, global_step, logging_loss, logging_steps, model,
                                output_dir, scheduler, tb_writer, tr_loss)
                    logging_loss = tr_loss


                if save_steps > 0 and global_step % save_steps == 0:
                    # Save model checkpoint
                    save_model_checkpoint(global_step, model, output_dir)

            if max_steps > 0 and global_step > max_steps:
                epoch_iterator.close()
                break

        if max_steps > 0 and global_step > max_steps:
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step


def get_log_dir(output_dir):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(output_dir, 'runs', current_time)
    return log_dir


def log_metrics(eval_dataset, evaluate_during_training, global_step, logging_loss, logging_steps, model, output_dir,
                scheduler, tb_writer, tr_loss):
    if evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
        results = evaluate(eval_dataset, model, output_dir)
        for key, value in results.items():
            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / logging_steps, global_step)


def save_model_checkpoint(global_step, model, output_dir):
    output_dir = os.path.join(output_dir, 'checkpoint-{}'.format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    logger.info("Saving model checkpoint to %s", output_dir)
    return output_dir


def compute_metrics(preds, labels):
    acc = (preds == labels).mean()
    f1_micro = metrics.f1_score(y_true=labels, y_pred=preds, average="micro")
    f1_macro = metrics.f1_score(y_true=labels, y_pred=preds, average="macro")
    return {
        "acc": acc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "acc_and_f1": (acc + f1_micro) / 2,
    }
