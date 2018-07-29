import os
import arff
import argparse
import numpy as np

def get_features(path, features_dim):
    file_content = arff.load(open(path, "r"))
    data=np.array(file_content['data'], dtype="float32")
    return data[:, : features_dim]

def get_labels(path, features_dim, labels_dim):
    file_content = arff.load(open(path, "r"))
    data=np.array(file_content['data'], dtype="float32")
    return data[: , features_dim : ]

def set_dims(dataset_path):
    with open(os.path.join(dataset_path, "count.txt"), "r") as f:
        return [int (i) for i in f.read().split("\n") if i != ""]

if __name__ == '__main__':
    # Convert and save arff files to numpy-pickles for faster data I/O.
    print("Starting Training")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Name of the Datset")
    args = parser().get_parser().parse_args()
    dataset = args.dataset
    features_dim, labels_dim = set_dims("./" + dataset + "/")
    train_features, train_labels = get_features("./" + dataset + "/" + dataset + "-train.arff", features_dim), get_labels("./" + dataset + "/" + dataset + "-train.arff", features_dim, labels_dim)
    train_features.dump("./" + dataset + "/" + dataset + "-train-features.pkl")
    train_labels.dump("./" + dataset + "/" + dataset + "-train-labels.pkl")
    print("End Train")
    print("Start Test")
    test_features, test_labels = get_features("./" + dataset + "/" + dataset + "-test.arff", features_dim), get_labels("./" + dataset + "/" + dataset + "-test.arff", features_dim, labels_dim)
    test_features.dump("./" + dataset + "/" + dataset + "-test-features.pkl")
    test_labels.dump("./" + dataset + "/" + dataset + "-test-labels.pkl")
