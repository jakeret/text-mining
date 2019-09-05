import glob
import logging
import os
from datetime import datetime

import numpy as np
import torch
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from sklearn import metrics
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import utils

EVAL_RESULTS_FILE_NAME = "eval_results_{}.txt"

MODEL_CLASS = BertForSequenceClassification

TOKENIZER_CLASS = BertTokenizer

MODEL_NAME = "bert-base-german-cased"
# MODEL_NAME = "bert-base-uncased"


logger = logging.getLogger(__name__)

WEIGHTS_NAME = "pytorch_model.bin"


DEFAULT_HYPERPARAMETERS = dict(
    per_gpu_train_batch_size = 8,
    max_steps = -1,
    num_train_epochs = 3.0,
    gradient_accumulation_steps = 1,
    learning_rate = 2e-5,
    adam_epsilon = 1e-8,
    warmup_steps = 0,
    max_grad_norm = 1.0,
    weight_decay = 0.0,
    do_lower_case=False,
    max_seq_length = 128,
)


def evaluate_checkpoints(eval_dataset, output_dir, eval_all_checkpoints = True):
    checkpoints = [output_dir]
    if eval_all_checkpoints:
        checkpoints = list(os.path.dirname(c) for c in glob.glob(output_dir + '/**/' + WEIGHTS_NAME, recursive=True))
        logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging

    results = {}
    logger.info("Evaluate the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:
        global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
        model = MODEL_CLASS.from_pretrained(checkpoint)
        model.to(get_device())

        try:
            global_step = int(global_step)
        except ValueError:
            global_step = "final"

        result = evaluate(eval_dataset, model, output_dir, prefix=global_step)
        result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
        results.update(result)

    return results


def evaluate(dataset, model, eval_output_dir, prefix="", per_gpu_eval_batch_size = 8):
    n_gpu = torch.cuda.device_count()
    eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)

    eval_dataloader = DataLoader(dataset,
                                 sampler=SequentialSampler(dataset),
                                 batch_size=eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    device = get_device()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        input_ids, attention_mask, token_type_ids, labels = tuple(t.to(device) for t in batch)

        with torch.no_grad():

            tmp_eval_loss, logits = model(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          token_type_ids=token_type_ids,
                                          labels=labels)

            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)

    results = compute_metrics(preds, out_label_ids)

    results["loss"] = eval_loss / nb_eval_steps

    store_eval_results(results, eval_output_dir, prefix)

    return results


def store_eval_results(result, eval_output_dir, prefix):
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    output_eval_file = os.path.join(eval_output_dir, EVAL_RESULTS_FILE_NAME.format(prefix))
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


def build_model(do_lower_case, num_labels):
    config = BertConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
    tokenizer = TOKENIZER_CLASS.from_pretrained(MODEL_NAME, do_lower_case=do_lower_case)
    model = MODEL_CLASS.from_pretrained(MODEL_NAME, config=config)

    model.to(get_device())

    return model, tokenizer


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, tokenizer, train_dataset, eval_dataset, output_dir, per_gpu_train_batch_size, max_steps,
          num_train_epochs, gradient_accumulation_steps, learning_rate, adam_epsilon, warmup_steps, max_grad_norm,
          weight_decay, logging_steps=50, evaluate_during_training=True, save_steps=50):
    """ Train the model """


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
    tr_loss = 0.0
    logging_loss = 0.0

    model.zero_grad()

    tb_writer = SummaryWriter(log_dir=get_log_dir(output_dir))
    device = get_device()

    train_iterator = trange(int(num_train_epochs), desc="Epoch", disable=False)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            input_ids, attention_mask, token_type_ids, labels = tuple(t.to(device) for t in batch)

            loss, _ = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels)

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
                    log_performance_evaluation(train_dataset, "train", global_step, model, output_dir, tb_writer)

                    if evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        log_performance_evaluation(eval_dataset, "eval", global_step, model, output_dir, tb_writer)

                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / logging_steps, global_step)
                    tb_writer.flush()

                    logging_loss = tr_loss

                if save_steps > 0 and global_step % save_steps == 0:
                    # Save model checkpoint
                    save_model_checkpoint(tokenizer, model, global_step, output_dir)

            if 0 < max_steps < global_step:
                epoch_iterator.close()
                break

        if 0 < max_steps < global_step:
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step


def log_performance_evaluation(dataset, prefix, global_step, model, output_dir, tb_writer):
    results = evaluate(dataset, model, output_dir)
    for key, value in results.items():
        tb_writer.add_scalar('{}/{}'.format(prefix, key), value, global_step)


def get_log_dir(output_dir):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(output_dir, 'runs', current_time)
    return log_dir


def save_model_checkpoint(model, tokenizer, global_step, output_dir):
    output_dir = os.path.join(output_dir, 'checkpoint-{}'.format(global_step))
    utils.write_model(model, tokenizer, output_dir)


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
