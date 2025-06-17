import matplotlib.pyplot as plt
import json
from argparse import ArgumentParser
import seaborn as sns
import pandas as pd


def read_data(ground_truth: bool) -> pd.DataFrame():
    if ground_truth:
        filename = './data/results_ground_truth_retrieved_metrics.json'
    else:
        filename = './data/results_dpr_retrieved_metrics.json'
    print(f'Reading {filename} results...')
    with open(filename, 'r') as f:
        data = json.load(f)

    if len(data) == 0:
        raise Exception("No data found")

    rows = []
    for item in data:
        rows.append({
            'header_exact': item['header']['exact_match'],
            'header_chrf': item['header']['chrf_score'],
            'header_bert': item['header']['bert_score'],
            'values_exact': item['values']['exact_match'],
            'values_chrf': item['values']['chrf_score'],
            'values_bert': item['values']['bert_score']
        })
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    gt_or_dpr = 'Ground Truth' if ground_truth else 'DPR'
    if ground_truth:
        results_filename = './data/mean_results_ground_truth_retrieved_metrics.txt'
    else:
        results_filename = './data/mean_results_dpr_retrieved_metrics.txt'
    with open(results_filename, 'w') as f:
        for key, value  in df.mean(axis=0).items():
            f.write(f'{gt_or_dpr} {key}: {value:.4f}\n')
    return df


def distribution_plots(df: pd.DataFrame, ground_truth: bool) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = ['exact', 'chrf', 'bert']
    for i, metric in enumerate(metrics):
        sns.histplot(df[f'header_{metric}'], bins=50, ax=axes[0, i], kde=True)
        axes[0, i].set_title(f'Header {metric} Score Distribution')

        sns.histplot(df[f'values_{metric}'], bins=50, ax=axes[1, i], kde=True, color='orange')
        axes[1, i].set_title(f'Values {metric} Score Distribution')

    plt.tight_layout()
    if ground_truth:
        plt.savefig('plots/distribution_plots_ground_truth.png')
    else:
        plt.savefig('plots/distribution_plots_dpr.png')


def aggregated_values(df: pd.DataFrame, ground_truth: bool, window_size: int = 1000) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for idx, val in enumerate(['header', 'values']):
        for metric in ['exact', 'chrf', 'bert']:
            axes[idx].plot(list(range(len(df))), df[f'{val}_{metric}'].rolling(window_size).mean(),
                            label=f'{val.capitalize()} {metric}')

        axes[idx].set_xlabel('Data Point Index')
        axes[idx].set_ylabel(f'Score ({window_size}-point rolling avg)')
        axes[idx].set_title(f'{val.capitalize()} Metrics')
        axes[idx].legend()
    if ground_truth:
        plt.savefig('plots/aggregated_values_ground_truth.png')
    else:
        plt.savefig('plots/aggregated_values_dpr.png')


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--ground_truth",
        action="store_true",
        help="When True, plots will be made for the test set's ground truth results metrics, not DPR Retrieved"
    )
    args = parser.parse_args()

    df = read_data(args.ground_truth)

    distribution_plots(df, args.ground_truth)
    aggregated_values(df, args.ground_truth, window_size=2000)


if __name__ == '__main__':
    main()
