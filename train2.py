import argparse
from sklearn.kernel_ridge import KernelRidge
from misc import load_data, split_data, preprocess_scale, train_model, evaluate_model, pretty_print_results, save_metrics

def main(random_state: int = 42, alpha: float = 1.0, kernel: str = 'linear', gamma: float = None, test_size: float = 0.2, show_n: int = 10):
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df, test_size=test_size, random_state=random_state)
    X_train_scaled, X_test_scaled, _ = preprocess_scale(X_train, X_test)

    # pass gamma only if not None
    if gamma is not None:
        model = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma)
    else:
        model = KernelRidge(alpha=alpha, kernel=kernel)

    model = train_model(model, X_train_scaled, y_train)
    results = evaluate_model(model, X_test_scaled, y_test)
    preds = results.pop("preds")
    pretty_print_results(f"KernelRidge (kernel={kernel}, alpha={alpha}, gamma={gamma})", results, y_test, preds, show_n)
    save_metrics("results_summary.csv", f"KernelRidge_kernel={kernel}_alpha={alpha}_gamma={gamma}", results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--kernel", type=str, default='linear', choices=['linear','poly','rbf','sigmoid','cosine'])
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--show_n", type=int, default=10)
    args = parser.parse_args()
    main(random_state=args.random_state, alpha=args.alpha, kernel=args.kernel, gamma=args.gamma, test_size=args.test_size, show_n=args.show_n)

