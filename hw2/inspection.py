# task1: calculate the label entropy
# task2: error rate
import numpy as np
import sys
from collections import Counter


class DataProcess:
    def __init__(self, tsv_file_path=None):
        self.data = None
        self.attr = None
        self.labels = None
        self.features = None
        if tsv_file_path is not None:
            self.load_data(tsv_file_path)

    def load_data(self, tsv_file_path):
        self.data = np.genfromtxt(tsv_file_path, delimiter='\t', skip_header=True)
        self.labels = self.data[:, -1]
        self.features = self.data[:, :-1]

    def get_data(self, tsv_file_path):
        self.data = np.genfromtxt(tsv_file_path, delimiter='\t', skip_header=True)
        return self.data

    def get_attr(self, tsv_file_path):
        all_col = np.genfromtxt(tsv_file_path, delimiter='\t', dtype=str, max_rows=1)
        self.attr = all_col[:-1]
        return self.attr

    def get_labels(self):
        self.labels = self.data[:, -1]
        return self.labels

    def get_features_and_labels(self):
        return self.features, self.labels


def cal_entropy(labels):
    label_count = Counter(labels)
    total_samples = len(labels)

    entropy = 0
    for count in label_count.values():
        p_i = count / total_samples
        entropy -= p_i * np.log2(p_i)
    return entropy


def cal_error_rate(predictions, labels):
    errors = np.sum(predictions != labels)
    error_rate = errors / len(labels)
    return error_rate


# write results into file
def main(in_file, out_file):
    processor = DataProcess(in_file)
    labels = processor.get_labels()

    entropy = cal_entropy(labels)
    majority_label = Counter(labels).most_common(1)[0][0]
    majority_predictions = np.full_like(labels, majority_label)
    error_rate = cal_error_rate(majority_predictions, labels)

    # Write the results to the output file
    with open(out_file, 'w') as f:
        f.write(f'entropy: {entropy:.6f}\n')
        f.write(f'error: {error_rate:.6f}\n')


# Command line interface
if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    main(input_file, output_file)
