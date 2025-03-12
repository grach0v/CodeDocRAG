import argparse
import optuna
from evaluation import run_experiment
import numpy as np
import optuna.visualization as vis
from functools import partial
from typing import Any, Optional

def objective(trial: optuna.trial.Trial, args: argparse.Namespace) -> float:
    """
    Objective function for hyperparameter optimization using optuna.

    Parameters:
        trial (optuna.trial.Trial): An Optuna trial object.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        float: Mean F1 score from the experiment.
    """
    chunk_size = trial.suggest_int(
        'chunk_size', 
        args.chunk_size_min, 
        args.chunk_size_max, 
        step=args.chunk_size_step
    )
    chunk_overlap_ration = trial.suggest_float(
        'chunk_overlap', 
        args.chunk_overlap_ratio_min, 
        args.chunk_overlap_ratio_max, 
        step=args.chunk_overlap_ratio_step
    )

    chunk_overlap = int(chunk_size * chunk_overlap_ration)

    num_retrieved_chunks = trial.suggest_int(
        'num_retrieved_chunks', 
        args.num_retrieved_chunks_min, 
        args.num_retrieved_chunks_max
    )
    threshold = trial.suggest_float(
        'threshold', 
        args.threshold_min, 
        args.threshold_max, 
        step=args.threshold_step
    )
    model_names = args.model_names.split(',')
    model_name = trial.suggest_categorical('model_name', model_names)
    
    results = run_experiment(
        questions_path=args.questions_path,
        text_path=args.text_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        num_retrieved_chunks=num_retrieved_chunks,
        model_name=model_name,
        threshold=threshold,
        device=args.device,
    )
    
    f1_scores = [metric['F1'] for metric in results]
    mean_f1 = np.mean(f1_scores) if f1_scores else 0
    return mean_f1

def run_study(args: argparse.Namespace) -> optuna.study.Study:
    """
    Run optuna study optimization.
    
    Parameters:
        args (argparse.Namespace): Parsed command-line arguments.
        
    Returns:
        optuna.study.Study: The completed study.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(partial(objective, args=args), n_trials=args.trials)
    return study

def save_study_viz(study: optuna.study.Study, out_path: str) -> None:
    """
    Save optuna study visualizations as a combined HTML file.

    Parameters:
        study (optuna.study.Study): Completed study with trials.
        out_path (str): Path to save the visualization HTML.
    """
    hist_fig = vis.plot_optimization_history(study)
    param_fig = vis.plot_param_importances(study)
    slice_fig = vis.plot_slice(study)
    
    with open(out_path, 'w') as f:
        f.write("<html><head><meta charset='utf-8' /></head><body>")
        f.write("<h1>Optimization History</h1>")
        f.write(hist_fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<h1>Parameter Importances</h1>")
        f.write(param_fig.to_html(full_html=False, include_plotlyjs=False))
        f.write("<h1>Slice Plot</h1>")
        f.write(slice_fig.to_html(full_html=False, include_plotlyjs=False))
        f.write("</body></html>")
    print(f"Combined optimization figures saved to {out_path}")

def save_study(study: optuna.study.Study, out_path: str) -> None:
    """
    Save study trials to a CSV file.

    Parameters:
        study (optuna.study.Study): Completed study with trials.
        out_path (str): CSV file path to save study trials.
    """
    trials_df = study.trials_dataframe()
    trials_df.to_csv(out_path, index=False)

def main() -> None:
    """
    Main function to parse arguments, run hyperparameter optimization, and optionally save results.
    """
    parser = argparse.ArgumentParser(
        description="Optimize hyperparameters for the evaluation experiment."
    )
    parser.add_argument("--trials", type=int, default=100, help="Number of optimization trials")
    parser.add_argument("--questions_path", type=str, default="data/state_of_the_union_questions_df.csv", help="Path to questions CSV")
    parser.add_argument("--text_path", type=str, default="data/state_of_the_union.md", help="Path to text file")
    # renamed parameter, default None skips saving viz
    parser.add_argument("--viz_out", type=str, default=None, help="Output HTML file for optimization plots visualization (if provided)")
    
    # New parameters for optuna hyperparameter ranges
    parser.add_argument("--chunk_size_min", type=int, default=50, help="Minimum chunk size")
    parser.add_argument("--chunk_size_max", type=int, default=300, help="Maximum chunk size")
    parser.add_argument("--chunk_size_step", type=int, default=None, help="Step size for chunk size")
    
    parser.add_argument("--chunk_overlap_ratio_min", type=float, default=0.0, help="Minimum chunk overlap ratio to chunk size")
    parser.add_argument("--chunk_overlap_ratio_max", type=float, default=0.7, help="Maximum chunk overlap ratio to chunk size")
    parser.add_argument("--chunk_overlap_ratio_step", type=float, default=None, help="Step size for chunk overlap ratio")
    
    parser.add_argument("--num_retrieved_chunks_min", type=int, default=1, help="Minimum number of retrieved chunks")
    parser.add_argument("--num_retrieved_chunks_max", type=int, default=10, help="Maximum number of retrieved chunks")
    
    parser.add_argument("--threshold_min", type=float, default=0.0, help="Minimum threshold value")
    parser.add_argument("--threshold_max", type=float, default=0.9, help="Maximum threshold value")
    parser.add_argument("--threshold_step", type=float, default=None, help="Step size for threshold")
    
    # New device parameter
    parser.add_argument("--device", type=str, default="cuda", help="Device to run experiments on")
    parser.add_argument("--model_names", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Comma-separated list of model names")
    # New study CSV parameter
    parser.add_argument("--study_csv", type=str, default=None, help="Output CSV file for study trials (if provided)")
    
    args = parser.parse_args()
    study = run_study(args)
    
    if args.viz_out is not None:
        save_study_viz(study, args.viz_out)
    else:
        print("Visualization output not provided; skipping saving viz HTML.")
    
    if args.study_csv is not None:
        save_study(study, args.study_csv)
    else:
        print("Study CSV output not provided; skipping saving study CSV.")

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
if __name__ == '__main__':
    main()
