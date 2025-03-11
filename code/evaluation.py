from typing import List, Dict, Tuple
import pandas as pd 
import json 
from fixed_token_chunker import FixedTokenChunker
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse

def get_ranges(text: str, chunks: List[str], chunk_indices: List[int]) -> List[Tuple[int, int]]:
    """Return list of (start, end) ranges for given chunk indices found in text."""
    ranges: List[Tuple[int, int]] = []
    for index in chunk_indices:
        start = text.find(chunks[index])
        if start == -1:
            continue
        end = start + len(chunks[index])
        ranges.append((start, end))
    return ranges

def sum_of_ranges(ranges: List[Tuple[int, int]]) -> int:
    """Return the sum of the lengths of the given intervals."""
    return sum(end - start for start, end in ranges)

def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping intervals into consolidated intervals."""
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        last_start, last_end = merged[-1]
        curr_start, curr_end = current
        if curr_start <= last_end:
            merged[-1] = (last_start, max(last_end, curr_end))
        else:
            merged.append(current)
    return merged

def intersect_intervals(retrieved: List[Tuple[int, int]], targets: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Return intersections between retrieved and target intervals."""
    retrieved.sort(key=lambda x: x[0])
    targets.sort(key=lambda x: x[0])
    i, j = 0, 0
    intersections: List[Tuple[int, int]] = []
    while i < len(retrieved) and j < len(targets):
        r_start, r_end = retrieved[i]
        t_start, t_end = targets[j]
        start = max(r_start, t_start)
        end = min(r_end, t_end)
        if start <= end:
            intersections.append((start, end))
        if r_end < t_end:
            i += 1
        else:
            j += 1
    return intersections

def run_experiment(questions_path: str, text_path: str, chunk_size: int = 100, 
                   chunk_overlap: int = 20, num_retrieved_chunks: int = 1, 
                   model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                   threshold: float = 0.0) -> List[Dict[str, float]]:
    """
    Run evaluation experiment on the provided corpus and question set.

    Parameters:
        questions_path (str): Path to CSV file containing questions.
        text_path (str): Path to text file containing corpus text.
        chunk_size (int): Chunk size for splitting text.
        chunk_overlap (int): Overlap between successive chunks.
        num_retrieved_chunks (int): Number of closest chunks to retrieve.
        model_name (str): HuggingFace model name used for embeddings.
        threshold (float): Minimal cosine similarity value to consider a chunk.

    Returns:
        List[Dict[str, float]]: List of evaluation metrics (IoU, Precision, Recall, F1) per question.
    """
    questions_df = pd.read_csv(questions_path)
    with open(text_path, encoding="utf-8") as f:
        text = f.read()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    chunker = FixedTokenChunker(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = chunker.split_text(text)

    model = SentenceTransformer(model_name)
    text_embeddings = model.encode(chunks)
    question_embeddings = model.encode(questions_df['question'].tolist())

    similarities = cosine_similarity(question_embeddings, text_embeddings)
    
    # Retrieve indices with similarity >= threshold per query
    retrieved_indices = []
    for sim in similarities:
        valid_idx = np.where(sim >= threshold)[0]
        if valid_idx.size:
            sorted_valid = valid_idx[np.argsort(sim[valid_idx])][::-1]
            retrieved_indices.append(sorted_valid[:num_retrieved_chunks])
        else:
            retrieved_indices.append(np.array([], dtype=int))
    
    closest_texts_indices = retrieved_indices

    retrieved_intervals = [
        get_ranges(text, chunks, list(indices)) 
        for indices in closest_texts_indices
    ]
    target_intervals = [
        [(ref['start_index'], ref['end_index']) for ref in json.loads(references)]
        for references in questions_df['references']
    ]
    retrieved_intervals = [merge_intervals(intervals) for intervals in retrieved_intervals]
    target_intervals = [merge_intervals(intervals) for intervals in target_intervals]

    results: List[Dict[str, float]] = []
    for retrieved, target in zip(retrieved_intervals, target_intervals):
        intersections = intersect_intervals(retrieved.copy(), target.copy())
        total_intersection = sum_of_ranges(intersections)
        total_retrieved = sum_of_ranges(retrieved)
        total_target = sum_of_ranges(target)
        IoU = total_intersection / (total_retrieved + total_target - total_intersection) if (total_retrieved + total_target - total_intersection) else 0
        precision = total_intersection / total_retrieved if total_retrieved else 0
        recall = total_intersection / total_target if total_target else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        results.append({
            'IoU': IoU,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        })
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run evaluation experiment')
    parser.add_argument('--questions', default='../data/questions_df.csv', help='Path to questions CSV')
    parser.add_argument('--text', default='../data/state_of_the_union.md', help='Path to text file')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size')
    parser.add_argument('--chunk_overlap', type=int, default=20, help='Chunk overlap')
    parser.add_argument('--num_retrieved_chunks', type=int, default=1, help='Number of closest chunks to retrieve')
    parser.add_argument('--model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='HuggingFace model name')
    parser.add_argument('--threshold', type=float, default=0.0, help='Minimum cosine similarity threshold')
    args = parser.parse_args()

    metrics = run_experiment(
        questions_path=args.questions,
        text_path=args.text,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        num_retrieved_chunks=args.num_retrieved_chunks,
        model_name=args.model_name,
        threshold=args.threshold
    )
    
    for m in metrics:
        print(f"IoU: {m['IoU']:.2f} Precision: {m['Precision']:.2f}, "
              f"Recall: {m['Recall']:.2f}, F1: {m['F1']:.2f}")