import argparse
import optuna
from evaluation import run_experiment
import numpy as np
import optuna.visualization as vis

def objective(trial):
    chunk_size = trial.suggest_int(
        'chunk_size', 
        args.chunk_size_min, 
        args.chunk_size_max, 
        step=args.chunk_size_step
    )
    chunk_overlap = trial.suggest_int(
        'chunk_overlap', 
        args.chunk_overlap_min, 
        args.chunk_overlap_max, 
        step=args.chunk_overlap_step
    )
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
    model_name = trial.suggest_categorical(
        'model_name', 
        [
            "sentence-transformers/all-MiniLM-L6-v2", 
            "sentence-transformers/multi-qa-mpnet-base-dot-v1"
        ]
    )
    
    results = run_experiment(
        questions_path=args.questions_path,
        text_path=args.text_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        num_retrieved_chunks=num_retrieved_chunks,
        model_name=model_name,
        threshold=threshold
    )
    
    f1_scores = [metric['F1'] for metric in results]
    mean_f1 = np.mean(f1_scores) if f1_scores else 0
    return mean_f1

def main():
    parser = argparse.ArgumentParser(
        description="Optimize hyperparameters for the evaluation experiment."
    )
    # Existing parameters
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--questions_path", type=str, default="data/state_of_the_union_questions_df.csv", help="Path to questions CSV")
    parser.add_argument("--text_path", type=str, default="data/state_of_the_union.md", help="Path to text file")
    parser.add_argument("--history_out", type=str, default="study_optimization_history.html", help="Output file for optimization history plot")
    parser.add_argument("--param_out", type=str, default="study_param_importances.html", help="Output file for parameter importance plot")
    
    # New parameters for optuna hyperparameter ranges
    parser.add_argument("--chunk_size_min", type=int, default=50, help="Minimum chunk size")
    parser.add_argument("--chunk_size_max", type=int, default=300, help="Maximum chunk size")
    parser.add_argument("--chunk_size_step", type=int, default=10, help="Step size for chunk size")
    
    parser.add_argument("--chunk_overlap_min", type=int, default=0, help="Minimum chunk overlap")
    parser.add_argument("--chunk_overlap_max", type=int, default=50, help="Maximum chunk overlap")
    parser.add_argument("--chunk_overlap_step", type=int, default=5, help="Step size for chunk overlap")
    
    parser.add_argument("--num_retrieved_chunks_min", type=int, default=1, help="Minimum number of retrieved chunks")
    parser.add_argument("--num_retrieved_chunks_max", type=int, default=3, help="Maximum number of retrieved chunks")
    
    parser.add_argument("--threshold_min", type=float, default=0.0, help="Minimum threshold value")
    parser.add_argument("--threshold_max", type=float, default=0.5, help="Maximum threshold value")
    parser.add_argument("--threshold_step", type=float, default=0.1, help="Step size for threshold")
    
    global args
    args = parser.parse_args()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    hist_fig = vis.plot_optimization_history(study)
    hist_fig.write_html(args.history_out)
    print(f"Optimization history saved to {args.history_out}")
    
    param_fig = vis.plot_param_importances(study)
    param_fig.write_html(args.param_out)
    print(f"Parameter importance plot saved to {args.param_out}")

if __name__ == '__main__':
    main()
