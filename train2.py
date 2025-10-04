
# train2.py
"""
KernelRidge training script.
Prints metrics and saves them to results_summary.csv.
"""

from sklearn.kernel_ridge import KernelRidge
from misc import load_data, split_data, preprocess_scale, train_model, evaluate_model, pretty_print_results, save_metrics
import argparse

def main(random_state: int = 42, alpha: float = 1.0, kernel: str = 'linear', test_size: float = 0.2, show_n: int = 10, save_csv: str = "results_summary.csv"):
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df, test_size=test_size, random_state=random_state)
    X_train_scaled, X_test_scaled, _ = preprocess_scale(X_train, X_test)
    model = KernelRidge(alpha=alpha, kernel=kernel)
    model = train_model(model, X_train_scaled, y_train)
    results = evaluate_model(model, X_test_scaled, y_test)
    preds = results.pop("preds")
    pretty_print_results(f"KernelRidge (kernel={kernel}, alpha={alpha})", results, y_test, preds, n_show=show_n)
    save_metrics(save_csv, f"KernelRidge_kernel={kernel}_alpha={alpha}", results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--kernel", type=str, default='linear')
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--show_n", type=int, default=10)
    parser.add_argument("--save_csv", type=str, default="results_summary.csv")
    args = parser.parse_args()
    main(random_state=args.random_state, alpha=args.alpha, kernel=args.kernel, test_size=args.test_size, show_n=args.show_n, save_csv=args.save_csv)

