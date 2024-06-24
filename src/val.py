import argparse
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


# TODO:
# 1. typing and costrings


def main(config) -> None:
    utils.print_config(config=config, verbose=config.verbose)
    torch.set_float32_matmul_precision(config.matmul_precision)

    logging.info("Starting validation")

    device = torch.device("cuda:0" if config.cuda else "cpu")
    n_labels = config.data.n_labels

    tokenizer = utils.getTokenizer(config.model.type)

    tokenizer = partial(
        tokenizer,
        max_length=config.data.max_length,
        padding=config.data.padding,
        truncation=config.data.truncation,
    )

    val_dataset = TextDataset(
        data_path=config.data.val_path,
        tokenizer=tokenizer,
        features_cols=config.data.features_cols,
        label_cols=config.data.label_cols,
    )

    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.data.batch,
        shuffle=False,
        pin_memory=config.data.pin_memory,
        num_workers=config.data.num_workers,
    )

    logging.debug("Loaded dataloader.")

    # load model checkpoint
    model = utils.get_model(
        type=config.model.type,
        n_labels=n_labels,
        model_path=config.model.weight_path,
        device=device,
    )

    model.eval()
    model.to(device)
    logging.debug("Loaded model checkpoint.")

    # TODO: Provide the option for more eval metrics

    val_precision_macro = MultilabelPrecision(num_labels=n_labels)  # 0.5 thresh
    val_recall_macro = MultilabelRecall(num_labels=n_labels)  # 0.5 thresh
    val_f1_macro = MultilabelF1Score(num_labels=n_labels)  # 0.5 thresh

    val_progbar = tqdm(val_dl, desc="Validation")
    for batch in val_progbar:
        ids, masks, labels = batch

        ids = ids.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(ids, attention_mask=masks)
            outputs = outputs[0]  # logits

            outputs = outputs.detach().cpu()
            labels = labels.cpu()
            val_precision_macro.update(outputs, labels)
            val_recall_macro.update(outputs, labels)
            val_f1_macro.update(outputs, labels)

    # compute across epoch

    avg_val_prec = val_precision_macro.compute().item()
    avg_val_rec = val_recall_macro.compute().item()
    avg_f1_val = val_f1_macro.compute().item()

    # instead of logging, we could also create a file to store the results.
    logging.info(
        f"Precision (macro): {avg_val_prec}, Recall (macro): {avg_val_rec}, F1 (macro): {avg_f1_val}"
    )

    logging.info("Finished Validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train analog meter recognition")

    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        type=str,
        required=True,
        help="Specify the configuration file for validation.",
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
