import json
from sacrebleu.metrics import CHRF
from sentence_transformers import SentenceTransformer, util
from typing import Callable
from tqdm import tqdm
from argparse import ArgumentParser


def transform_table_string(table_string):
    """
    Transforms a table from a single string format into a list of cell values.
    The expected format is a header row and a data row, separated by '\n'.
    Columns within each row are separated by '|'.
    Example: "header1|header2\ndata1|data2" -> ["header1", "header2", "data1", "data2"]
    """
    if not table_string or '\n' not in table_string:
        return []

    parts = table_string.strip().split('\n')
    header = parts[0].split('|')
    data = parts[1].split('|')

    return header, data


def exact_match(c, c_prime):
    """
    Checks for exact string identity.
    """
    return 1 if c == c_prime else 0


class chrFScore(CHRF):
    """
    A class to compute the chrF score between two strings.
    """
    def chrf_score(self, c, c_prime):
        return self.corpus_score([c], [[c_prime]]).score / 100.0


class RescaledBertScore:
    """
    A class to compute a rescaled BERT-based similarity score.
    It uses sentence embeddings from a pre-trained model and calculates
    cosine similarity, which is then rescaled to be between 0 and 1.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the model.
        """
        self.model = SentenceTransformer(model_name)

    def __call__(self, c, c_prime):
        """
        Computes the similarity score between two strings.
        """
        # Encode the strings into embeddings
        embedding1 = self.model.encode(c, convert_to_tensor=True)
        embedding2 = self.model.encode(c_prime, convert_to_tensor=True)

        # Compute cosine similarity
        cosine_sim = util.pytorch_cos_sim(embedding1, embedding2).item()

        # Rescale from [-1, 1] to [0, 1]
        return (cosine_sim + 1) / 2


def calculate_precision(t, tgt, similarity_metric: Callable):
    """
    Calculates the precision for an enriched table 't' against a ground truth table 'tgt'.

    Args:
        t (list): The enriched table, represented as a list of strings.
        tgt (list): The ground truth table, a list of strings.
        similarity_metric (function): The function to use for comparing cells.

    Returns:
        float: The precision score.
    """
    if not t:
        return 0.0

    sum_of_max_similarities = 0
    for c in t:
        max_similarity = 0
        for c_prime in tgt:
            similarity = similarity_metric(c, c_prime)
            if similarity > max_similarity:
                max_similarity = similarity
        sum_of_max_similarities += max_similarity

    return sum_of_max_similarities / len(t)


def calculate_recall(t: str, tgt: str, similarity_metric: Callable) -> float:
    """
    Calculates the recall for an enriched table 't' against a ground truth table 'tgt'.

    Args:
        t (list): The enriched table, a list of strings.
        tgt (list): The ground truth table, a list of strings.
        similarity_metric (function): The function to use for comparing cells.

    Returns:
        float: The recall score.
    """
    if not tgt:
        return 0.0

    sum_of_max_similarities = 0
    for c_prime in tgt:
        max_similarity = 0
        for c in t:
            similarity = similarity_metric(c, c_prime)
            if similarity > max_similarity:
                max_similarity = similarity
        sum_of_max_similarities += max_similarity

    return sum_of_max_similarities / len(tgt)


def calculate_f1_score(t, tgt, similarity_metric: Callable):
    """
    Calculates the F1 score from precision and recall given the similarity metric.
    """
    precision = calculate_precision(t, tgt, similarity_metric)
    recall = calculate_recall(t, tgt, similarity_metric)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def main():

    parser = ArgumentParser()


    parser.add_argument(
        "--ground_truth",
        action="store_true",
        help="When True, predictions will be made for the test set's ground truth documents, not DPR Retrieved"
    )
    args = parser.parse_args()

    if args.ground_truth:
        print("Calculating metrics for the test set's ground truth documents.")
        results_file='data/results_ground_truth_retrieved.json'
    else:
        print("Calculating metrics for the DPR index documents.")
        results_file='data/results_dpr_retrieved.json'

    if not results_file.endswith('.json'):
        raise ValueError('results_file must end with .json')

    with open(results_file, 'r') as enriched_tables:
        test_data_ground_truth = json.load(enriched_tables)
    metrics_results = []
    bert_scorer = RescaledBertScore()
    chrf_scorer = chrFScore()
    for item in tqdm(test_data_ground_truth.values(), total=len(test_data_ground_truth.values())):
        ground_truth_header, ground_truth_values = transform_table_string(item['serialized_query_csv'])
        enriched_header, enriched_values = transform_table_string(item['enriched_truncated_serialized_query_csv'])
        metrics_curr_result = {
            'header':
                {
                    'exact_match': calculate_f1_score(t=enriched_header,
                                                      tgt=ground_truth_header,
                                                      similarity_metric=exact_match),
                    'chrf_score': calculate_f1_score(t=enriched_header,
                                                     tgt=ground_truth_header,
                                                     similarity_metric=chrf_scorer.chrf_score),
                    'bert_score': calculate_f1_score(t=enriched_header, tgt=ground_truth_header,
                                                     similarity_metric=bert_scorer),
                },
            'values':
                {
                    'exact_match': calculate_f1_score(t=enriched_values,
                                                      tgt=ground_truth_values,
                                                      similarity_metric=exact_match),
                    'chrf_score': calculate_f1_score(t=enriched_values,
                                                     tgt=ground_truth_values,
                                                     similarity_metric=chrf_scorer.chrf_score),
                    'bert_score': calculate_f1_score(t=enriched_values, tgt=ground_truth_values,
                                                     similarity_metric=bert_scorer),
                }
        }
        metrics_results.append(metrics_curr_result)
    metrics_file = results_file.split('.json')[0] + '_metrics.json'
    with open(metrics_file, 'w') as metrics_results_file:
        json.dump(metrics_results, metrics_results_file)


if __name__ == '__main__':
    main()