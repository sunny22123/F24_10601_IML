import argparse
import numpy as np
from inspection import DataProcess, cal_error_rate


class Node:
    def __init__(self, attr=None, vote=None):
        self.left = None
        self.right = None
        self.attr = attr
        self.vote = vote


class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = None
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.root = self._fit(X, y)
        return self

    def _fit(self, X, y, depth=0):
        if len(set(y)) == 1:
            return Node(vote=y[0])

        if depth >= self.max_depth:
            unique, counts = np.unique(y, return_counts=True)
            probabilities = counts / len(y)
            vote_dict = dict(zip(unique, probabilities))
            return Node(vote=vote_dict)

        best_attr, best_mi = self.split(X, y)
        if best_mi == 0:
            unique, counts = np.unique(y, return_counts=True)
            probabilities = counts / len(y)
            vote_dict = dict(zip(unique, probabilities))
            return Node(vote=vote_dict)

        node = Node(attr=best_attr)

        left_indices = X[:, best_attr] == 0
        right_indices = X[:, best_attr] == 1

        node.left = self._fit(X[left_indices], y[left_indices], depth + 1)
        node.right = self._fit(X[right_indices], y[right_indices], depth + 1)

        return node

    def split(self, x, y):
        best_attr = None
        best_mi = -1

        for i in range(x.shape[1]):
            x_column = x[:, i]
            mi = self.cal_mutual_information(x_column, y)

            if mi > best_mi:
                best_mi = mi
                best_attr = i

        return best_attr, best_mi

    def predict(self, X):
        return np.array([self._predict(x, self.root) for x in X])

    def _predict(self, x, node):
        if isinstance(node.vote, dict):
            # If probabilities are tied, choose the higher label (1 over 0)
            max_prob = max(node.vote.values())
            max_labels = [label for label, prob in node.vote.items() if prob == max_prob]
            return max(max_labels)
        if node.vote is not None:
            return node.vote
        return self._predict(x, node.left if x[node.attr] == 0 else node.right)

    def print_tree(self, node, depth=0, file=None, features=None, X=None, y=None, parent_attr=None,
                   is_right_child=False):
        if node is None:
            return

        if X is None:
            X = self.X
        if y is None:
            y = self.y

        class_dist = self.get_class_distribution(y)
        dist_str = f"[{class_dist[0]} 0/{class_dist[1]} 1]"

        if depth == 0:
            file.write(f"{dist_str}\n")
        else:
            attr_name = features[parent_attr] if features is not None and parent_attr is not None else "Unknown"
            attr_value = "1" if is_right_child else "0"
            indent = "| " * (depth - 1)
            file.write(f"{indent}{attr_name} = {attr_value}: {dist_str}\n")

        if node.attr is not None:
            left_X, left_y = self.split_data(X, y, node.attr, 0)
            right_X, right_y = self.split_data(X, y, node.attr, 1)

            self.print_tree(node.left, depth + 1, file, features, left_X, left_y, node.attr, False)
            self.print_tree(node.right, depth + 1, file, features, right_X, right_y, node.attr, True)

    def split_data(self, X, y, attr, value):
        mask = X[:, attr] == value
        return X[mask], y[mask]

    @staticmethod
    def get_class_distribution(y):
        if y is None or len(y) == 0:
            return (0, 0)
        unique, counts = np.unique(y, return_counts=True)
        dist = dict(zip(unique, counts))
        return dist.get(0, 0), dist.get(1, 0)

    @staticmethod
    def cal_entropy(y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    @staticmethod
    def cal_con_entropy(x_column, y):
        data_0 = x_column == 0
        data_1 = x_column == 1

        p_0 = np.sum(data_0) / len(x_column)
        p_1 = np.sum(data_1) / len(x_column)

        h_y_given_x0 = DecisionTree.cal_entropy(y[data_0])
        h_y_given_x1 = DecisionTree.cal_entropy(y[data_1])

        return p_0 * h_y_given_x0 + p_1 * h_y_given_x1

    @staticmethod
    def cal_mutual_information(x_column, y):
        h_y = DecisionTree.cal_entropy(y)
        h_y_given_x = DecisionTree.cal_con_entropy(x_column, y)
        return h_y - h_y_given_x


def validate_tree_depths(x_train, y_train, x_test, y_test, max_depths):
    results = []
    for depth in max_depths:
        try:
            tree = DecisionTree(max_depth=depth)
            tree.fit(x_train, y_train)

            train_predictions = tree.predict(x_train)
            test_predictions = tree.predict(x_test)

            train_error = cal_error_rate(train_predictions, y_train)
            test_error = cal_error_rate(test_predictions, y_test)

            results.append({
                'depth': depth,
                'train_error': train_error,
                'test_error': test_error,
                'status': 'Success'
            })
            print(f"Depth {depth}: Train error = {train_error:.4f}, Test error = {test_error:.4f}")
        except Exception as e:
            results.append({
                'depth': depth,
                'status': f'Failed: {str(e)}'
            })
            print(f"Depth {depth}: Failed - {str(e)}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str)
    parser.add_argument("test_input", type=str)
    parser.add_argument("max_depth", type=int)
    parser.add_argument("train_out", type=str)
    parser.add_argument("test_out", type=str)
    parser.add_argument("metrics_out", type=str)
    parser.add_argument("print_out", type=str)
    args = parser.parse_args()

    train_data = DataProcess(args.train_input)
    test_data = DataProcess(args.test_input)

    x_train, y_train = train_data.get_features_and_labels()
    x_test, y_test = test_data.get_features_and_labels()

    tree = DecisionTree(max_depth=args.max_depth)
    tree.fit(x_train, y_train)

    train_predictions = tree.predict(x_train)
    test_predictions = tree.predict(x_test)

    train_error = cal_error_rate(train_predictions, y_train)
    test_error = cal_error_rate(test_predictions, y_test)

    np.savetxt(args.train_out, train_predictions, fmt='%d')
    np.savetxt(args.test_out, test_predictions, fmt='%d')

    with open(args.metrics_out, 'w') as f:
        f.write(f"error(train): {train_error}\n")
        f.write(f"error(test): {test_error}\n")

    with open(args.print_out, 'w') as f:
        features = train_data.get_attr(args.train_input)
        tree.print_tree(tree.root, file=f, features=features, y=y_train)
