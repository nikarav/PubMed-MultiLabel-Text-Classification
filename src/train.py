import argparse
import os
import sys
import torch
import torch.utils
import torch.utils.data
import util.utils as utils
import logging
from data.dataset import TextDataset
from functools import partial

from tqdm import tqdm
from torchmetrics.classification import (
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelF1Score,
)

# from utils.const import _PROJECT_NAME, _NEPTUNE_PROJECT

# TODO:
# 1. typing and costrings


def main(config) -> None:
    utils.print_config(config=config, verbose=config.verbose)

    # Set matrix multiplication precision
    torch.set_float32_matmul_precision(config.matmul_precision)

    logging.info("Starting training procedure.")

    device = torch.device("cuda:0" if config.cuda else "cpu")
    n_labels = config.data.n_labels
    weights_save_path = config.model.save_path
    os.makedirs(weights_save_path, exist_ok=True)  # create weights folder

    # Get tokenize, based on the model we want to use.
    # In case of Bert, use Bert tokenizer
    tokenizer = utils.getTokenizer(config.model.type)

    tokenizer = partial(
        tokenizer,
        max_length=config.data.max_length,
        padding=config.data.padding,
        truncation=config.data.truncation,
    )

    train_dataset = TextDataset(
        data_path=config.data.train_path,
        tokenizer=tokenizer,
        features_cols=config.data.features_cols,
        label_cols=config.data.label_cols,
    )

    val_dataset = TextDataset(
        data_path=config.data.val_path,
        tokenizer=tokenizer,
        features_cols=config.data.features_cols,
        label_cols=config.data.label_cols,
    )

    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch,
        shuffle=True,
        pin_memory=config.data.pin_memory,
        num_workers=config.data.num_workers,
    )
    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.data.batch,
        shuffle=False,
        pin_memory=config.data.pin_memory,
        num_workers=config.data.num_workers,
    )
    model = utils.get_model(config.model.type, n_labels)
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(config.model.lr))

    criterion = (
        torch.nn.BCEWithLogitsLoss()
    )  # treats each label as an independent binary classification problem

    # These metrics use threshold of 0.5 by default
    val_precision_macro = MultilabelPrecision(num_labels=n_labels)
    val_recall_macro = MultilabelRecall(num_labels=n_labels)
    val_f1_macro = MultilabelF1Score(num_labels=n_labels)

    epochs = config.model.epochs
    avg_train_loss = []
    avg_val_loss = []
    avg_val_prec, avg_val_rec, avg_f1_val = (
        [],
        [],
        [],
    )  # actually not necessary to have arrays, same for the losses. However, could be used in the future to save results in csv format
    for epoch in range(epochs):
        model.train()  # train mode

        steps = 0
        epoch_loss = 0
        train_progbar = tqdm(train_dl, desc=f"Epoch {epoch+1} - Training")
        for batch in train_progbar:
            ids, masks, labels = batch

            ids = ids.to(device)  # to gpu or cpu
            masks = masks.to(device)  # to gpu or cpu
            labels = labels.to(device)  # to gpu or cpu
            labels = labels.to(torch.float32)  # cast to float32, same as outputs

            optim.zero_grad()
            outputs = model(ids, attention_mask=masks)
            outputs = outputs[0]  # logits

            loss = criterion(outputs, labels)

            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            steps += 1
        avg_train_epoch_loss = epoch_loss / steps
        avg_train_loss.append(avg_train_epoch_loss)
        # train_progbar.set_postfix(avg_train_epoch_loss)

        model.eval()  # val mode
        steps = 0
        epoch_loss = 0
        val_progbar = tqdm(val_dl, desc=f"Epoch {epoch+1} - Validation")
        for batch in val_progbar:
            ids, masks, labels = batch

            ids = ids.to(device)  # to gpu or cpu
            masks = masks.to(device)  # to gpu or cpu
            labels = labels.to(device)  # to gpu or cpu
            with torch.no_grad():
                outputs = model(ids, attention_mask=masks)
                outputs = outputs[0]  # logits

                labels = labels.to(torch.float32)
                loss = criterion(outputs, labels)

                epoch_loss += loss.item()
                steps += 1

                outputs = outputs.detach().cpu()
                labels = labels.cpu()
                val_precision_macro.update(outputs, labels)
                val_recall_macro.update(outputs, labels)
                val_f1_macro.update(outputs, labels)

        # compute across epoch
        avg_val_epoch_loss = epoch_loss / steps
        avg_val_loss.append(avg_val_epoch_loss)

        avg_val_prec.append(val_precision_macro.compute().item())
        avg_val_rec.append(val_recall_macro.compute().item())
        avg_f1_val.append(val_f1_macro.compute().item())

        # reset for next epoch
        val_precision_macro.reset()
        val_recall_macro.reset()
        val_f1_macro.reset()

        logging.debug(
            f"Epoch: {epoch+1}, Train Loss: {avg_train_epoch_loss}, Val Loss: {avg_val_epoch_loss}"
        )
        logging.debug(
            f"Epoch: {epoch+1}, Val Precision (macro): {avg_val_prec[-1]}, Val Recall (macro): {avg_val_rec[-1]}, Val F1 (macro): {avg_f1_val[-1]}"
        )

        torch.save(model.state_dict(), f"{weights_save_path}/model_epoch{epoch+1}.pth")

    logging.info("Finished Training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train analog meter recognition")

    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        type=str,
        required=True,
        help="Specify the configuration file",
    )

    parser.add_argument(
        "--cuda",
        dest="cuda",
        required=False,
        default=False,
        action="store_true",
        help="Usa cuda",
    )

    parser.add_argument(
        "--gpu",
        dest="gpu",
        type=int,
        required=False,
        default=0,
        help="Specify gpu to run",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        required=False,
        default=False,
        action="store_true",
        help="increase output verbosity",
    )

    args = parser.parse_args()

    level = "DEBUG" if args.verbose else "INFO"
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.cuda:
        if torch.cuda.is_available():
            logging.info("Cuda is available.")
        else:
            logging.error("Cuda is not available. Exiting...")
            sys.exit()
    else:
        logging.info("No cuda specified. Using cpu.")

    config = utils.read_config_file(args.config)
    if config is None:
        raise Exception(f"Error reading file: {config}")

    utils.set_seed()  # for reproducibility
    main(utils.merge_configs(args, config))
