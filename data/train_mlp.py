import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from glob import glob
import pandas as pd
from fineweb2_hq.mlp import BinaryClassifier
import os
import argparse


def set_seed(seed):
    import torch
    import numpy as np
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_one_epoch(model, optimizer, loss_fn, train_loader):
    model.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.flatten(), labels)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Batch {i} loss: {loss}")


def classifier_metrics(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    for i, data in enumerate(test_loader):
        inputs, labels = data

        outputs = torch.nn.functional.sigmoid(model(inputs))
        outputs = outputs.flatten()

        for output in outputs:
            if output > 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
        for label in labels:
            if label > 0.5:
                y_true.append(1)
            else:
                y_true.append(0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
    }


def load_data(folder_dir):
    df = pd.concat(
        [pd.read_parquet(file_path) for file_path in glob(f"{folder_dir}/*.parquet")]
    )
    embds_np = np.array(
        [embd for embd in df["metadata"].map(lambda x: x["embeddings"][0])],
        dtype=np.float32,
    )
    labels_np = (
        df["metadata"]
        .map(lambda x: 1 if x["label"] == "positive" else 0)
        .to_numpy(dtype=np.float32)
    )
    return embds_np, labels_np


def train_classifier(
    data_dir,
    save_file_path,
    num_epochs=6,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=42,
):
    set_seed(seed)
    embds_train, labels_train = load_data(f"{data_dir}/train_80/")
    embds_test, labels_test = load_data(f"{data_dir}/valid_10/")

    train_data = torch.tensor(embds_train).to(device)
    train_labels = torch.tensor(labels_train).to(device)

    test_data = torch.tensor(embds_test).to(device)
    test_labels = torch.tensor(labels_test).to(device)

    print(f"{len(train_data)=}, {len(test_data)=}")

    train_set = list(zip(train_data, train_labels))
    test_set = list(zip(test_data, test_labels))

    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=256)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256)

    classifier = BinaryClassifier().to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=3e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for i in range(num_epochs):
        print(f"Epoch {i}")
        train_one_epoch(classifier, optimizer, loss_fn, train_loader)
        print(f"Train metrics:{classifier_metrics(classifier, train_loader)}")
        print(f"Val.  metrics:{classifier_metrics(classifier, test_loader)}")

    print(f"Saving model to {save_file_path}")
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
    classifier.to_pt(save_file_path)

    classifier2 = BinaryClassifier.from_pt(save_file_path).to(device)

    print(
        f"Train metrics  (trained model):\n{classifier_metrics(classifier, train_loader)}"
    )
    print(
        f"Val.  metrics  (trained model):\n{classifier_metrics(classifier, test_loader)}"
    )
    print(
        f"Val.  metrics (load from file):\n{classifier_metrics(classifier2, test_loader)}"
    )


def main(args):
    train_classifier(
        args.dataset_dir,
        args.output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-path", help="path of the trained classifier", required=True
    )
    parser.add_argument(
        "--dataset-dir",
        help="directory of the MKC/MKC+ dataset with embeddings",
        required=True,
    )

    args = parser.parse_args()

    main(args)
