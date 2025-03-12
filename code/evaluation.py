from typing import List, Dict, Tuple, Any
import pandas as pd 
import json 
from fixed_token_chunker import FixedTokenChunker
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse
import gc

def get_ranges(text: str, chunks: List[str], chunk_indices: List[int]) -> List[Tuple[int, int]]:
    """
    Return list of (start, end) ranges for given chunk indices found in text.
    
    Parameters:
        text (str): The source text.
        chunks (List[str]): List of text chunks.
        chunk_indices (List[int]): Indices of chunks in the list.
    
    Returns:
        List[Tuple[int, int]]: List of (start, end) ranges.
    """
    ranges: List[Tuple[int, int]] = []
    for index in chunk_indices:
        start = text.find(chunks[index])
        if start == -1:
            continue
        end = start + len(chunks[index])
        ranges.append((start, end))
    return ranges

def sum_of_ranges(ranges: List[Tuple[int, int]]) -> int:
    """
    Calculate the total length of all intervals.
    
    Parameters:
        ranges (List[Tuple[int, int]]): List of (start, end) intervals.
    
    Returns:
        int: Sum of interval lengths.
    """
    return sum(end - start for start, end in ranges)

def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merge overlapping intervals into consolidated intervals.
    
    Parameters:
        intervals (List[Tuple[int, int]]): List of (start, end) intervals.
    
    Returns:
        List[Tuple[int, int]]: Merged intervals.
    """
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
    """
    Compute the intersections between retrieved and target intervals.
    
    Parameters:
        retrieved (List[Tuple[int, int]]): List of retrieved intervals.
        targets (List[Tuple[int, int]]): List of target intervals.
    
    Returns:
        List[Tuple[int, int]]: List of intersection intervals.
    """
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

def create_chunks(text: str, chunk_size: int, chunk_overlap: int, model_name: str) -> Tuple[List[str], Any]:
    """
    Create text chunks using FixedTokenChunker.
    
    Args:
        text (str): The source text.
        chunk_size (int): The desired size of each chunk.
        chunk_overlap (int): The number of overlapping tokens between chunks.
        model_name (str): The HuggingFace model name used for tokenization.
    
    Returns:
        Tuple[List[str], Any]: A tuple of the list of text chunks and the tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    chunker = FixedTokenChunker(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = chunker.split_text(text)
    return chunks, tokenizer

def create_embeddings(chunks: List[str], questions: List[str], model_name: str, device: str) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Compute embeddings for text chunks and questions using SentenceTransformer.
    
    Args:
    
        chunks (List[str]): List of text chunks.
        questions (List[str]): List of questions.
        model_name (str): The SentenceTransformer model name.
        device (str): Device on which to run the model.
    
    Returns:
        Tuple[Any, np.ndarray, np.ndarray]: The model instance, text embeddings, and question embeddings.
    """
    model = SentenceTransformer(model_name, device=device)
    text_embeddings = model.encode(chunks)
    question_embeddings = model.encode(questions)
    return text_embeddings, question_embeddings

def retrieve_chunks(question_embeddings: np.ndarray, text_embeddings: np.ndarray, threshold: float, num_retrieved_chunks: int) -> List[np.ndarray]:
    """
    Calculate cosine similarities between question and text embeddings and retrieve indices exceeding a threshold.

    Args:
        question_embeddings (np.ndarray): Embeddings for questions.
        text_embeddings (np.ndarray): Embeddings for text chunks.
        threshold (float): Minimum similarity score to consider.
        num_retrieved_chunks (int): Number of top chunks to retrieve for each query.

    Returns:
        List[np.ndarray]: A list containing numpy arrays of retrieved chunk indices for each question.
    """
    similarities = cosine_similarity(question_embeddings, text_embeddings)
    retrieved_indices = []
    for sim in similarities:
        valid_idx = np.where(sim >= threshold)[0]
        if valid_idx.size:
            sorted_valid = valid_idx[np.argsort(sim[valid_idx])][::-1]
            retrieved_indices.append(sorted_valid[:num_retrieved_chunks])
        else:
            retrieved_indices.append(np.array([], dtype=int))
    return retrieved_indices

def measure_metrics(text: str, chunks: List[str], retrieved_indices: List[np.ndarray], questions_df: pd.DataFrame) -> List[Dict[str, float]]:
    # Prepare intervals and compute evaluation metrics.
    retrieved_intervals = [merge_intervals(get_ranges(text, chunks, list(indices))) for indices in retrieved_indices]
    target_intervals = [
        merge_intervals([(ref['start_index'], ref['end_index']) for ref in json.loads(references)])
        for references in questions_df['references']
    ]
    results = []
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

def run_experiment(questions_path: str, text_path: str, chunk_size: int = 100, 
                   chunk_overlap: int = 20, num_retrieved_chunks: int = 1, 
                   model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                   threshold: float = 0.0,
                   device: str = 'cpu') -> List[Dict[str, float]]:
    """
    Execute the evaluation experiment by performing chunking, embedding, retrieval, and metric calculations.
    
    Args:
        questions_path (str): Path to the CSV file containing questions.
        text_path (str): Path to the text file.
        chunk_size (int, optional): Size of each text chunk. Defaults to 100.
        chunk_overlap (int, optional): Number of overlapping tokens between chunks. Defaults to 20.
        num_retrieved_chunks (int, optional): Number of closest chunks to retrieve per query. Defaults to 1.
        model_name (str, optional): HuggingFace model name used for processing. Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
        threshold (float, optional): Minimum cosine similarity threshold. Defaults to 0.0.
        device (str, optional): Device on which the model is run. Defaults to 'cpu'.
        
    Returns:
        List[Dict[str, float]]: A list of metric dictionaries for each evaluation.
    """
    # Load data
    questions_df = pd.read_csv(questions_path)
    with open(text_path, encoding="utf-8") as f:
        text = f.read()

    # Create text chunks
    chunks, tokenizer = create_chunks(text, chunk_size, chunk_overlap, model_name)
    
    # Compute embeddings for chunks and questions
    text_embeddings, question_embeddings = create_embeddings(chunks, questions_df['question'].tolist(), model_name, device)
    # Retrieve best chunk indices per question (cosine similarity computed inside retrieve_chunks)
    retrieved_indices = retrieve_chunks(question_embeddings, text_embeddings, threshold, num_retrieved_chunks)
    
    # Measure evaluation metrics
    results = measure_metrics(text, chunks, retrieved_indices, questions_df)
    
    # Free heavy resources
    del tokenizer, model, text_embeddings, question_embeddings
    gc.collect()
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
    parser.add_argument('--device', type=str, default='cpu', help='Device to run model on')
    args = parser.parse_args()

    metrics = run_experiment(
        questions_path=args.questions,
        text_path=args.text,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        num_retrieved_chunks=args.num_retrieved_chunks,
        model_name=args.model_name,
        threshold=args.threshold,
        device=args.device
    )
    
    for m in metrics:
        print(f"IoU: {m['IoU']:.2f} Precision: {m['Precision']:.2f}, "
              f"Recall: {m['Recall']:.2f}, F1: {m['F1']:.2f}")