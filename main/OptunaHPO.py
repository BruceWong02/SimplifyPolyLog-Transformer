import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn

import numpy as np
import math

import logging
import os
import datetime
import traceback

import Transformer as tf


DEVICE = tf.device
BATCHSIZE = 64
EPOCHS = 10
# DATA_num_per_epoch = 50000

SEED = 420
np.random.seed(SEED)
torch.manual_seed(SEED)

VERSION = "v4_0_0_0"
STUDY_DESCRIPTION = "Use two different non-sine position embedding."
TRAIN_filename = "data/train/Train_tokens_finial_a.csv"
VALIDATE_filename = "data/predict/input/Validate_tokens_finial_a.csv"
TEST_filename = "data/predict/input/Test_tokens_finial_a.csv"


def create_model(trial):
    # define parameters
    paras = {
        "src_vocab_size": tf.vocab_size,
        "tgt_vocab_size": tf.vocab_size,
        "src_pad_idx": tf.pad_idx,
        "tgt_pad_idx": tf.pad_idx,
        "d_model": trial.suggest_int("d_model", 128, 512, step=128),
        "num_heads": 4,
        "num_encoder_layers": 3,
        "num_decoder_layers": 3,
        "dropout_p": 0,
        "max_len": 1000,
        "device": DEVICE,
    }

    model = tf.Transformer(**paras).to(DEVICE)
    return model


def objective(trial):
    test_loss = -1
    try:
        # create model
        model = create_model(trial)

        # get optimizer and learning rate
        # lr_max = trial.suggest_float("lr_max", 1e-5, 1e-3, log=True)
        lr_max = 1e-6
        # warmup_steps = trial.suggest_int("warmup_epochs", 2, 10) * math.ceil(
        #     DATA_num_per_epoch / BATCHSIZE
        # )  # steps

        # scheduler_4_step = tf.CreateNoamSche(opt, model.d_model, warmup_steps, lr_max)
        opt = torch.optim.Adam(model.parameters(), lr=lr_max, betas=(0.9, 0.98), eps=1e-9)
        scheduler_4_step = None

        for epoch in range(EPOCHS):
            print("-" * 25, f"Epoch {epoch}", "-" * 25)

            # load data and fit (train and test)
            # train_dataloader = model.Load_data(
            #     train_dataset, DATA_num_per_epoch, BATCHSIZE, if_shuffle=True
            # )
            train_loss = model.Train(
                opt,
                loss_fn,
                train_dataloader,
                scheduler_for_step=scheduler_4_step,
                epoch=epoch,
            )
            Validate_loss = model.Test(loss_fn, Validate_dataloader)

            trial.report(Validate_loss, epoch)

    except Exception as e:
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        error_message = str(e)
        error_type = type(e).__name__
        traceback_message = traceback.format_exc()
        # Set error message as a user attribute.
        trial.set_user_attr("error_type", error_type)
        trial.set_user_attr("error_message", error_message)
        trial.set_user_attr("traceback", traceback_message)

        # log the error message and traceback message.
        logging.basicConfig(
            filename="runs/logs/OptunaHPOTrialError.log",
            level=logging.ERROR,
        )
        logging.error(f"[{timestamp}]  trial number: {trial.number}")
        logging.error(f"Error type: {error_type}")
        logging.error(f"Error message: {error_message}")
        logging.error(f"Traceback message: {traceback_message}")

        raise e

    if trial.should_prune():
        raise optuna.TrialPruned()

    return test_loss


def log_failed_trial(study, trial):
    if trial.state == optuna.trial.TrialState.FAIL:
        error_message = str(trial.exception)
        trial.set_user_attr("error_message", error_message)  # 将错误信息保存到试验的用户属性中


if __name__ == "__main__":
    if not os.path.exists("runs/logs"):
        os.makedirs("runs/logs")

    study = optuna.create_study(
        storage="sqlite:///runs/optuna_study_database/TransformerHyperparas.db",  # Specify the storage URL here.
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=EPOCHS, reduction_factor=3
        ),
        study_name="{}".format(VERSION),
        direction="minimize",
        load_if_exists=True,
    )
    study.set_user_attr("random_seed", SEED)
    study.set_user_attr("version", VERSION)
    study.set_user_attr("experiment_description", STUDY_DESCRIPTION)

    # get datasets
    train_dataset = tf.MyDataset(TRAIN_filename, tf.Vocab)
    Validate_dataset = tf.MyDataset(VALIDATE_filename, tf.Vocab)
    # train and test
    train_dataloader = tf.DataLoader(
            train_dataset,
            batch_size=BATCHSIZE,
            shuffle=True,
            collate_fn=train_dataset.pad_collate,
        )
    Validate_dataloader = tf.Transformer.Load_data(Validate_dataset, 2000, BATCHSIZE * 4, False)
    # get loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=tf.pad_idx)

    study.optimize(objective, n_trials=5, timeout=5 * 24 * 3600)

    # logging
    logging.basicConfig(filename="runs/logs/OptunaHPO.log", level=logging.INFO)

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    best_trial = study.best_trial

    logging.info("[{}]  version: {}".format(timestamp, VERSION))
    logging.info(f"Sampler: {study.sampler.__class__.__name__}")
    logging.info("Study statistics: ")
    logging.info("\tNumber of finished trials: {}".format(len(study.trials)))
    logging.info("\tNumber of pruned trials: {}".format(len(pruned_trials)))
    logging.info("\tNumber of complete trials: {}".format(len(complete_trials)))
    logging.info("Best trial:")
    logging.info("  Value: {}".format(best_trial.value))
    logging.info("  Params: ")
    for key, value in best_trial.params.items():
        logging.info("\t{}: {}".format(key, value))

    print("HPO over.")
