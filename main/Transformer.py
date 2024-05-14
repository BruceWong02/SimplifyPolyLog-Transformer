# version: 4.1.0.0

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import BeamSearchScorer, top_k_top_p_filtering
import math
import csv
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
import time

# from transformers import get_cosine_with_hard_restarts_schedule_with_warmup as CosReWarmSched


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    # nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


# Transformer Model
class Transformer(nn.Module):
    # Constructor
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        d_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
        max_len,
        device,
    ):
        """
        src
            shape: (batch_size, src sequence length)
        tgt
            shape: (batch_size, tgt sequence length)
        output
            shape: (batch_size, sequence length, tgt_vocab_size)

        --- src and tgt are all indices tensors ---
        """

        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.d_model = d_model
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.epochs_trained = 0
        self.max_len = max_len
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # LAYERS
        self.src_positional_enbedding = Embedding(max_len, d_model)
        self.tgt_positional_enbedding = Embedding(max_len, d_model)
        self.src_embedding = Embedding(
            src_vocab_size, d_model, padding_idx=src_pad_idx
        )
        self.tgt_embedding = Embedding(
            tgt_vocab_size, d_model, padding_idx=tgt_pad_idx
        )
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, src, tgt):
        N, src_seq_length = src.shape
        N, tgt_seq_length = tgt.shape

        # mask - Out size = (batch_size, sequence length)
        src_pad_mask = self.create_pad_mask(src, self.src_pad_idx)
        tgt_pad_mask = self.create_pad_mask(tgt, self.tgt_pad_idx)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_length).to(
            self.device
        )
        # convert to BoolTensor
        tgt_mask = tgt_mask == -torch.inf

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        # src = self.src_embedding(src) * math.sqrt(self.d_model)
        # tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)
        src_positions = (
            torch.arange(0, src_seq_length).unsqueeze(0).expand(N, -1).to(self.device)
        )
        tgt_positions = (
            torch.arange(0, tgt_seq_length).unsqueeze(0).expand(N, -1).to(self.device)
        )
        src = self.dropout(src + self.src_positional_enbedding(src_positions))
        tgt = self.dropout(tgt + self.tgt_positional_enbedding(tgt_positions))

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        out_logits = self.fc_out(transformer_out)

        return out_logits

    def create_pad_mask(self, matrix: torch.tensor, pad_idx: int) -> torch.tensor:
        """
        matrix shape: (batch_size, sequence_length)

        output shape: (batch_size, sequence_length)

        Support for mismatched key_padding_mask and attn_mask is deprecated. Use same type for both instead.

        UserWarning: Converting mask without torch.bool dtype to bool;
            this will negatively affect performance. Prefer to use a boolean mask directly.
        """
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_idx).to(self.device)

    def save_checkpoint(self, optimizer, train_loss_list):
        checkpoint_new = {
            "epoch": self.epochs_trained,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_list": train_loss_list,
        }
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        date_hour_str = now.strftime("%Y%m%d_%H")
        if not os.path.exists(f"checkpoints/{date_str}"):
            os.makedirs(f"checkpoints/{date_str}")
        checkpoint_filename = f"checkpoints/{date_str}/checkpoint{checkpoint_new['epoch']}_{date_hour_str}.pth.tar"
        torch.save(checkpoint_new, checkpoint_filename)

    def Load_checkpoint(self, checkpoint, opt=None, new_lr=None):
        if checkpoint is None:
            raise ValueError("Checkpoint is empty.")
            # print("checkpoint is None, load nothing")
        self.load_state_dict(checkpoint["model_state_dict"])
        if opt is not None:
            opt.load_state_dict(checkpoint["optimizer_state_dict"])
            if new_lr is not None:
                self.Change_learning_rate(opt, new_lr)
        self.epochs_trained = checkpoint["epoch"]

    @classmethod
    def Load_data(cls, dataset, data_num_per_epoch, batch_size, if_shuffle):
        """Load data from dataset"""
        if data_num_per_epoch is None:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=if_shuffle,
                collate_fn=dataset.pad_collate,
            )
            return dataloader            

        total_data_num = len(dataset)
        assert (
            data_num_per_epoch <= total_data_num
        ), f"number of training data per epoch is bigger than total number of training data: {data_num_per_epoch} > {total_data_num}"
        if data_num_per_epoch == total_data_num:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=if_shuffle,
                collate_fn=dataset.pad_collate,
            )
        else:
            indices = np.random.permutation(total_data_num)[:data_num_per_epoch]
            subset = Subset(dataset, indices)
            dataloader = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=if_shuffle,
                collate_fn=dataset.pad_collate,
            )

        return dataloader

    @staticmethod
    def Change_learning_rate(opt, lr):
        # change learning rate
        for param_group in opt.param_groups:
            param_group["lr"] = lr

    # def Train(self, opt, loss_fn, dataloader, scheduler=None, writer=None, epoch=0):
    def Train(
        self,
        opt,
        loss_fn,
        dataloader,
        scheduler_for_step=None,
        warmup_scheduler=None,
        epoch=0,
    ):
        """
        Train the model
        src and tgt should have <sos> and <eos>
        """

        self.train()
        total_loss = 0
        # scalar_name = f"Training loss/epoch {epoch}"
        steps_num = len(dataloader)

        for step, batch in enumerate(dataloader):
            # src and tgt shape: (batch_size, sequence_length)
            src, tgt = batch["src"].to(self.device), batch["tgt"].to(self.device)

            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]  # (batch_size, sequence_length)

            # Training
            pred = self(src, tgt_input)
            pred = pred.permute(
                0, 2, 1
            )  # (batch_size, tgt_vocab_size, sequence_length)

            # Calculate loss
            loss = loss_fn(pred, tgt_expected)
            if torch.isnan(loss):
                error_log_file = "./tf_error.log"
                print(f"Train loss is NaN. Error info saved at {error_log_file}")
                logging.basicConfig(filename=error_log_file, level=logging.ERROR)
                logging.error("--- Train loss is NaN ---")
                logging.info(
                    f"pred shape (batch_size, tgt_vocab_size, sequence_length): {pred.shape}\t tgt_expected shape (batch_size, sequence_length): {tgt_expected.shape}"
                )
                logging.info(
                    f"tgt expected max: {torch.max(tgt_expected)}\t min: {torch.min(tgt_expected)}"
                )
                tensors_to_save = {"pred": pred, "tgt": tgt}
                torch.save(tensors_to_save, "tensors_train.pt")
                logging.info(f"pred and tgt saved at tensors_train.pt")
                # raise ValueError("Train loss is NaN.")

            # Optimize
            opt.zero_grad()
            loss.backward()
            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
            opt.step()

            # upgrate learning rate
            if scheduler_for_step is not None:
                if warmup_scheduler is not None:
                    with warmup_scheduler.dampening():
                        scheduler_for_step.step(epoch + step / steps_num)
                else:
                    # scheduler_for_step.step(epoch + step / steps_num)
                    scheduler_for_step.step()

            # plot to tensorboard
            # writer.add_scalar(scalar_name, loss, global_step=step)

            total_loss += loss.detach().item()

        mean_loss = total_loss / steps_num
        return mean_loss

    # def Test(self, loss_fn, dataloader, writer=None, epoch=0):
    def Test(self, loss_fn, dataloader):
        """
        Test the model
        src and tgt should have <sos> and <eos>
        """

        self.eval()
        total_loss = 0
        # scalar_name = f"Testing loss/epoch {epoch}"
        steps_num = len(dataloader)

        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                # src and tgt shape: (batch_size, sequence_length)
                src, tgt = batch["src"].to(self.device), batch["tgt"].to(self.device)

                # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
                tgt_input = tgt[:, :-1]
                tgt_expected = tgt[:, 1:]  # (batch_size, sequence_length)

                # Testing
                pred = self(src, tgt_input)
                pred = pred.permute(
                    0, 2, 1
                )  # (batch_size, tgt_vocab_size, sequence_length)

                # Calculate loss
                loss = loss_fn(pred, tgt_expected)
                if torch.isnan(loss):
                    error_log_file = "./tf_error.log"
                    print(f"Test loss is NaN. Error info saved at {error_log_file}")
                    logging.basicConfig(filename=error_log_file, level=logging.ERROR)
                    logging.error("--- Test loss is NaN ---")
                    logging.info(
                        f"pred shape (batch_size, tgt_vocab_size, sequence_length): {pred.shape}\t tgt_expected shape (batch_size, sequence_length): {tgt_expected.shape}"
                    )
                    logging.info(
                        f"tgt expected max: {torch.max(tgt_expected)}\t min: {torch.min(tgt_expected)}"
                    )
                    tensors_to_save = {"pred": pred, "tgt": tgt}
                    torch.save(tensors_to_save, "tensors_test.pt")
                    logging.info(f"pred and tgt saved at tensors_test.pt")
                    raise ValueError("Test loss is NaN.")

                # plot to tensorboard
                # writer.add_scalar(scalar_name, loss, global_step=step)

                total_loss += loss.detach().item()

        mean_loss = total_loss / steps_num
        return mean_loss

    def Fit(
        self,
        opt,
        loss_fn,
        train_dataset,
        batch_size,
        data_num_per_epoch=None,
        Validate_dataset=None,
        if_shuffle_train=True,
        scheduler_for_step=None,
        scheduler_for_epoch=None,
        warmup_scheduler=None,
        epochs=1,
        writer=None,
        train_filename=None,
        Validate_filename=None,
        logging_folder="runs/logs/",
        if_save_checkpoint=False,
    ):
        """Train and Validate"""

        # Used for plotting and logging later on
        train_loss_list, Validate_loss_list = [], []
        train_scalar_name = "Loss/Training loss"
        Validate_scalar_name = "Loss/Validation loss"

        start_time = time.time()

        # training model
        if Validate_dataset is not None:
            # load data
            Validate_dataloader = self.Load_data(
                Validate_dataset, None, batch_size * 4, False
            )
        train_dataloader = self.Load_data(
            train_dataset, data_num_per_epoch, batch_size, if_shuffle_train
        )
        for epoch in range(epochs):
            print("-" * 25, f"Epoch {epoch}", "-" * 25)

            train_loss = self.Train(
                opt,
                loss_fn,
                train_dataloader,
                scheduler_for_step,
                warmup_scheduler,
                epoch,
            )
            train_loss_list.append(train_loss)
            if scheduler_for_epoch is not None:
                scheduler_for_epoch.step()
            # plot to tensorboard
            if writer is not None:
                writer.add_scalar(train_scalar_name, train_loss, global_step=epoch)
            # print loss
            print(f"Training loss: {train_loss:.4f}")

            if Validate_dataset is not None:
                # load data and test
                Validate_loss = self.Test(loss_fn, Validate_dataloader)
                Validate_loss_list.append(Validate_loss)
                if writer is not None:
                    writer.add_scalar(Validate_scalar_name, Validate_loss, global_step=epoch)
                print(f"Validation loss: {Validate_loss:.4f}\n")

        self.epochs_trained = epochs + self.epochs_trained

        end_time = time.time()
        runtime = end_time - start_time

        print("Total time taken: {:.4f} seconds".format(runtime))

        # record information
        if logging_folder is not None:
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            optimizer_name = type(opt).__name__

            if not os.path.exists(logging_folder):
                os.makedirs(logging_folder)

            if Validate_dataset is not None:
                logging_path = os.path.join(logging_folder, "TrainingAndValidating.log")
                print("Training and Validating model")
            else:
                logging_path = os.path.join(logging_folder, "TrainingAndValidating.log")
                print("Training model")
            logging.basicConfig(filename=logging_path, level=logging.INFO)

            if scheduler_for_step is None and scheduler_for_epoch is None:
                learning_rate = opt.param_groups[0]["lr"]
            else:
                learning_rate = "may not fixed due to the using of scheduler."

            logging.info(
                "[{}] Total time taken: {:.4f} seconds. Using {} device".format(
                    timestamp, runtime, self.device
                )
            )
            if train_filename is not None:
                logging.info("\ttraining dataset filename: {}".format(train_filename))
            if Validate_filename is not None:
                logging.info("\ttesting dataset filename: {}".format(Validate_filename))
            logging.info(
                "\tepoch range: ({}, {}), epochs: {}".format(
                    self.epochs_trained - epochs + 1, self.epochs_trained, epochs
                )
            )
            logging.info(
                "\ttraining dataset length: {}, training datas per epoch: {}".format(
                    len(train_dataset), data_num_per_epoch
                )
            )
            logging.info("\tlearning rate: {}".format(learning_rate))
            logging.info(
                "\ttraining batch size: {}, optimizer: {}, training time per epoch: {:.4f} seconds".format(
                    train_dataloader.batch_size, optimizer_name, runtime / epochs
                )
            )
            logging.info("\tLast training loss: {:.4f}".format(train_loss_list[-1]))
            if Validate_dataset is not None:
                logging.info("\tLast testing loss: {:.4f}".format(Validate_loss_list[-1]))

        # save model
        if if_save_checkpoint:
            self.save_checkpoint(opt, train_loss_list)

        return train_loss_list, Validate_loss_list

    def Generate_greedy(self, input_sequence, SOS_idx, EOS_idx, max_length=15):
        """
        use trained model to predict

        We supposed input_sequence is tokenized and converted into indices tensor here,
            which has <SOS> and <EOS> at the head and rear.
        This function process one sequence per time.

        input_sequence : input indices tensor
            shape: (batch_size, sequence_length)
        max_length : max output sequence length

        output : predicted indices list with sos and eos
            shape: (sequence_length)
        """

        batch_size = input_sequence.size(0)
        # (batch_size, sequence_length)
        tgt_input = torch.full(
            (batch_size, 1), SOS_idx, dtype=torch.long, device=self.device
        )
        finished_seq = [0]*batch_size


        model.eval()
        # num_tokens = len(input_sequence[0])
        with torch.no_grad():
            for _ in range(max_length):
                pred = self(input_sequence, tgt_input)
                # (batch_size, output_ids length, tgt_vocab_size)
                next_token_logits = pred[:, -1, :]
                # (batch_size, tgt_vocab_size)
                next_item = next_token_logits.topk(1)[1] # num with highest probability
                # (batch_size, 1)

                # let next_item to be tgt_pad_idx for finished sequence
                next_item = next_item * (1 - torch.tensor(finished_seq, device=self.device).unsqueeze(1)) + torch.tensor(finished_seq, device=self.device).unsqueeze(1) * self.tgt_pad_idx

                # Concatenate previous input with predicted best word
                tgt_input = torch.cat((tgt_input, next_item), dim=1)

                # Check eos token
                eos_status = next_item.squeeze(1) == EOS_idx
                finished_seq = [1 if eos_status[i] else finished_seq[i] for i in range(batch_size)]  

                # stop if all sequences are finished
                if sum(finished_seq) == batch_size:
                    break

        return tgt_input

    def Generate_topp(
            self, 
            input_sequence, 
            SOS_idx, 
            EOS_idx, 
            top_p=1, 
            max_length=15):
        
        batch_size = input_sequence.size(0)
        # (batch_size, sequence_length)
        tgt_input = torch.full(
            (batch_size, 1), SOS_idx, dtype=torch.long, device=self.device
        )
        finished_seq = [0]*batch_size


        model.eval()
        # num_tokens = len(input_sequence[0])
        with torch.no_grad():
            for _ in range(max_length):
                pred = self(input_sequence, tgt_input)
                # (batch_size, output_ids length, tgt_vocab_size)
                next_token_logits = pred[:, -1, :]
                # (batch_size, tgt_vocab_size)

                if 0 < top_p < 1:
                    # use top-p filter
                    filtered_next_token_logits = top_k_top_p_filtering(
                        next_token_logits,
                        top_k=0,
                        top_p=top_p,
                    )
                else:
                    # do nothing
                    filtered_next_token_logits = next_token_logits
                # (batch_size * num_beams, tgt_vocab_size)

                next_item = torch.multinomial(
                    F.softmax(filtered_next_token_logits, dim=-1), num_samples=1
                )
                # (batch_size, 1)

                # let next_item to be tgt_pad_idx for finished sequence
                next_item = next_item * (1 - torch.tensor(finished_seq, device=self.device).unsqueeze(1)) + torch.tensor(finished_seq, device=self.device).unsqueeze(1) * self.tgt_pad_idx

                # Concatenate previous input with predicted best word
                tgt_input = torch.cat((tgt_input, next_item), dim=1)

                # Check eos token
                eos_status = next_item.squeeze(1) == EOS_idx
                finished_seq = [1 if eos_status[i] else finished_seq[i] for i in range(batch_size)]  

                # stop if all sequences are finished
                if sum(finished_seq) == batch_size:
                    break

        return tgt_input

    def Generate_beam(
        self,
        input_sequence,
        SOS_idx,
        EOS_idx,
        num_beams=2,  # beam size
        length_penalty=0.75,
        num_beam_hyps_to_keep=1,
        max_length=1000,
    ):
        """
        Generate beam search sequences using a transformer model.

        Args:
            model (torch.nn.Module): The transformer model to use for generation.
            input_sequence (torch.Tensor): The input sequence to generate from.
            SOS_idx (int): The index of the start-of-sequence token.
            PAD_idx (int): The index of the padding token.
            EOS_idx (int): The index of the end-of-sequence token.
            batch_size (int, optional): The batch size to use for generation. Defaults to 1.
            max_length (int, optional): The maximum length of the generated sequence. Defaults to 15.
            num_beams (int, optional): The number of beams to use for beam search. Defaults to 2.
            num_beam_hyps_to_keep (int, optional): The number of beam hypotheses that shall be returned
                upon calling. Defaults to 1.

        Returns:
            torch.Tensor: The generated sequences.
        """

        # check inputs
        assert num_beams >= 1

        PAD_idx = self.tgt_pad_idx
        batch_size = input_sequence.size(0)

        # initiate BeamSearchScorer
        # do_early_stopping = False
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            device=self.device,
            length_penalty=length_penalty,
            num_beam_hyps_to_keep=num_beam_hyps_to_keep,
        )

        output_ids = torch.full(
            (batch_size * num_beams, 1), SOS_idx, dtype=torch.long, device=self.device
        )
        # (batch_size * num_beams, 1)

        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=output_ids.device
        )
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)
        # (batch_size * num_beams, 1)

        # repeat for beams
        input_sequence = input_sequence.repeat_interleave(repeats=num_beams, dim=0)
        # (batch_size * num_beams, src length)

        # Start generating
        for _ in range(max_length-1):
            outputs = self(input_sequence, output_ids)
            # (batch_size * num_beams, output_ids length, tgt_vocab_size)
            next_token_logits = outputs[:, -1, :]
            # (batch_size * num_beams, tgt_vocab_size)

            next_scores = F.log_softmax(next_token_logits, dim=-1)
            next_scores = next_scores + beam_scores[:, None].expand_as(next_scores)
            # (batch_size * num_beams, tgt_vocab_size)
            next_scores = next_scores.view(
                batch_size, num_beams * self.tgt_vocab_size
            )  # (batch_size, num_beams * tgt_vocab_size)

            # choose top k
            next_scores, next_tokens = torch.topk(
                next_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            # (batch_size, 2*num_beams), (batch_size, 2*num_beams)

            next_indices = next_tokens // self.tgt_vocab_size
            next_tokens = next_tokens % self.tgt_vocab_size

            scorer_results = beam_scorer.process(
                output_ids,
                next_scores,
                next_tokens,
                next_indices,
                PAD_idx,
                EOS_idx,
            )

            # check if is all done
            if beam_scorer.is_done:
                break

            beam_scores = scorer_results["next_beam_scores"]
            beam_next_tokens = scorer_results["next_beam_tokens"]
            beam_idx = scorer_results["next_beam_indices"]

            output_ids = torch.cat(
                [output_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
            )

        # get finial sequence
        sequences_results = beam_scorer.finalize(
            output_ids,
            beam_scores,
            beam_next_tokens,  # no use
            beam_idx,  # no use
            max_length,
            PAD_idx,
            EOS_idx,
        )

        sequences = sequences_results["sequences"]

        # (batch_size*num_beam_hyps_to_keep, sent_max_len)
        return sequences


# Build Dataset
class MyDataset(Dataset):
    def __init__(self, filename, vocab, if_add_tgt_mma=False):
        self.if_add_tgt_mma = if_add_tgt_mma
        if if_add_tgt_mma:
            self.dataframe = pd.read_csv(filename)[["src", "tgt", "tgt_sp"]]
            self.vocab = vocab
        else:
            self.dataframe = pd.read_csv(filename)[["src", "tgt"]]
            self.vocab = vocab

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        src_pn = self.dataframe["src"][idx]  # prefix notation string of src
        tgt_pn = self.dataframe["tgt"][idx]  # prefix notation string of tgt
        # split to tokens list
        src_tokens = src_pn.split()
        tgt_tokens = tgt_pn.split()
        # Then each batch in Dataloader has size: (batch_size, sequence_length)
        src = torch.tensor([self.vocab[token] for token in src_tokens])
        tgt = torch.tensor([self.vocab[token] for token in tgt_tokens])
        # Add <sos> and <eos>
        sos_id = self.vocab["<sos>"]
        eos_id = self.vocab["<eos>"]
        src = torch.cat((torch.tensor([sos_id]), src, torch.tensor([eos_id])))
        tgt = torch.cat((torch.tensor([sos_id]), tgt, torch.tensor([eos_id])))

        if self.if_add_tgt_mma:
            tgt_sp = self.dataframe["tgt_sp"][
                idx
            ]  # mathematica expression of tgt for validation
            return {"src": src, "tgt": tgt, "tgt_sp": tgt_sp}
        else:
            return {"src": src, "tgt": tgt}

    def pad_collate(self, batch):
        """Padding batch"""
        keys = batch[0].keys()
        padded_batch = {key: [] for key in keys}

        for key in keys:
            # collect data for each key
            tensors = [item[key] for item in batch]
            if key in {"src", "tgt"}:
                # padding for each key
                padded_data = pad_sequence(
                    tensors, batch_first=True, padding_value=self.vocab["<pad>"]
                )
                padded_batch[key] = padded_data
            else:
                padded_batch[key] = tensors

        return padded_batch


def build_vocab():
    Vocab = {
        "<pad>": 0,
        "<sos>": 1,
        "<eos>": 2,
        "x": 3,
        "polylog": 4,
        "Pow": 5,
        "Rational": 6,
        "Add": 7,
        "Mul": 8,
        "-": 9,
        "+": 10,
    }
    numbers_dict = {str(i): i + 11 for i in range(10)}
    Vocab.update(numbers_dict)
    return Vocab


def CreateNoamSche(optimizer, d_model, warmup_steps, lr_max):
    factor = lr_max / ((d_model * warmup_steps) ** (-0.5))
    noam_lambda = (
        lambda step: factor
        * (d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5)))
        if step > 0
        else 0
    )
    noam_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)
    return noam_scheduler


# -------------------------------------------------------------------
# Define things

device = "cuda" if torch.cuda.is_available() else "cpu"


# vocab
Vocab = build_vocab()
pad_idx = Vocab["<pad>"]
sos_idx = Vocab["<sos>"]
eos_idx = Vocab["<eos>"]
vocab_size = len(Vocab)

paras = {
    "src_vocab_size": vocab_size,
    "tgt_vocab_size": vocab_size,
    "src_pad_idx": pad_idx,
    "tgt_pad_idx": pad_idx,
    "d_model": 512,
    "num_heads": 4,
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "dropout_p": 0.1,
    "max_len": 1000,
    "device": device,
}

# model
model = Transformer(**paras).to(device)


# -------------------------------------------------------------------
