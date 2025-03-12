# CodeDocRAG

CodeDocRAG evaluates the FixedTokenChunker using proposed evaluation metrics. The core idea is to compare the intersection of retrieved and target token spans to their union. In the original paper, the size of token sets was used, but in this implementation, we compute the length of the corresponding text segments.

## Project Overview

- **FixedTokenChunker Evaluation:**  
  The project chunks source texts using a fixed token-based approach and evaluates the segmentation by comparing:
  - **Intersection:** The overlapping parts of retrieved and target token spans.
  - **Union:** The combined span of both retrieved and target intervals.
  This IoU metric is indicative of how well the chunking captures relevant text segments.

- **Metrics:**  
  Metrics computed include Intersection over Union (IoU), Precision, Recall, and F1 score. These metrics provide a quantitative measure of how closely the retrieved chunks match the annotated targets.

- **Experiments:**  
  - **Hyperparameter Optimization:**  
    Using Optuna, experiments optimize parameters such as `chunk_size`, `chunk_overlap`, `num_retrieved_chunks`, and `threshold` to maximize the F1 score.
  - **Visualization:**  
    Results and parameter importances are visualized in the `experiments.ipynb` notebook, which also documents trends observed in the hyperparameter optimization.
    
- **Data and Code Organization:**  
  - All required data files are stored in the `data` folder.
  - Source code and detailed experiments are maintained within the `code` repository.

## Experimental Findings and Conclusions

- Our experiments reveal that **chunk_size** and **threshold** parameters have the most significant impact on performance.
- Smaller chunk sizes with sufficient overlap tend to produce more flexible and detailed segmentations.
- Filtering chunks with low thresholds helps in reducing noise and improving metric scores.
- Overall, while our metrics provide a solid foundation for evaluation, they tend to penalize additional information strongly. Future work may involve weighting factors to balance over- and under-estimation.

## How to Use

1. Ensure that all data files are in the `data` folder.
2. Run the experiments using the main script or through the `experiments.ipynb` notebook.
   - The notebook will load saved trial data from the CSV file to display graphs quickly.
   - Note: Rerunning the experiment takes significant time and RAM.
3. Review the generated visualizations and results to understand the impact of different hyperparameters.
