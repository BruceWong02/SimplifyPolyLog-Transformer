import Transformer as tf

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import pytorch_warmup as warmup
import os
import math
from datetime import datetime
from transformers import optimization


current_date = datetime.now().strftime("%Y%m%d_%H")
# use tensorboard to plot
writerDir = f"runs/tensorboard_logs/binary/{current_date}"

if not os.path.exists(writerDir):
    os.makedirs(writerDir)
writer = SummaryWriter(writerDir)


# Load data
train_filename = "data/train/Train_tokens_finial_a.csv"
train_dataset = tf.MyDataset(train_filename, tf.Vocab)
# data_num_per_epoch = 50000
data_num_per_epoch = None
Validate_filename = "data/train/Validate_tokens_finial_a.csv"
Validate_dataset = tf.MyDataset(Validate_filename, tf.Vocab)


# model
model = tf.model

# load checkpoint
# checkpoint = torch.load("checkpoints/20231119/checkpoint100_20231119_16.pth.tar")
# model.Load_checkpoint(checkpoint, tf.opt, learning_rate)

# Training hyperparameters
num_epochs = 5
batch_size = 32
learning_rate_max = 0.0001
# warmup_steps = 5 * math.ceil(data_num_per_epoch/batch_size)  # steps
warmup_steps = 62500

# optimizer and loss
opt = torch.optim.Adam(model.parameters(), lr=learning_rate_max, betas=(0.9, 0.98), eps=1e-9)
# scheduler_4_step = tf.CreateNoamSche(opt, model.d_model, warmup_steps, learning_rate_max)
scheduler_4_step = optimization.get_constant_schedule_with_warmup(opt, num_warmup_steps=warmup_steps)
# scheduler_4_step = None
scheduler_4_epoch = None

# warmup_scheduler = warmup.LinearWarmup(opt, warmup_period=warmup_steps)
warmup_scheduler = None

criterion = nn.CrossEntropyLoss(ignore_index=model.tgt_pad_idx)


# -------------------------------------------------------
# Start Training

train_loss_list, Validate_loss_list = model.Fit(
    opt,
    criterion,
    train_dataset,
    batch_size=batch_size,
    data_num_per_epoch=data_num_per_epoch,
    Validate_dataset=Validate_dataset,
    if_shuffle_train=True,
    scheduler_for_step=scheduler_4_step,
    scheduler_for_epoch=scheduler_4_epoch,
    warmup_scheduler=warmup_scheduler,
    epochs=num_epochs,
    writer=writer,
    train_filename=train_filename,
    Validate_filename=Validate_filename,
    logging_folder="runs/logs/",
    if_save_checkpoint=True,
)

writer.close()
