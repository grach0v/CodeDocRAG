import pandas as pd 
import json 
from fixed_token_chunker import FixedTokenChunker
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse


def get_ranges(text, chunks, chunk_indices):
    ranges = []
    
    for index in chunk_indices:
        start = text.find(chunks[index])

        if start == -1:
            continue

        end = start + len(chunks[index])
        ranges.append((start, end))

    return ranges

def sum_of_ranges(ranges):
    return sum(end - start for start, end in ranges)

def merge_intervals(intervals):
    if not intervals:
        return []
    # Sort intervals based on the start time.
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        last_start, last_end = merged[-1]
        curr_start, curr_end = current
        # Check for overlap (assuming intervals are inclusive)
        if curr_start <= last_end:
            # Merge by extending the end time if necessary.
            merged[-1] = (last_start, max(last_end, curr_end))
        else:
            merged.append(current)
    return merged

def intersect_intervals(retrieved, targets):
    # Sort both lists by start times
    retrieved.sort(key=lambda x: x[0])
    targets.sort(key=lambda x: x[0])
    
    i, j = 0, 0
    intersections = []
    
    while i < len(retrieved) and j < len(targets):
        r_start, r_end = retrieved[i]
        t_start, t_end = targets[j]
        
        # Find overlap boundaries
        start = max(r_start, t_start)
        end = min(r_end, t_end)
        
        if start <= end:  # They overlap
            intersections.append((start, end))
        
        # Move the pointer that ends first
        if r_end < t_end:
            i += 1
        else:
            j += 1
            
    return intersections


def run_experiment(questions_path, text_path, chunk_size=100, chunk_overlap=20, k=1):
    questions_df = pd.read_csv(questions_path)
    questions_df = questions_df.query('corpus_id == "state_of_the_union"')[['question', 'references']]

    with open(text_path) as f:
        text = ''.join(f.readlines())

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    chunker = FixedTokenChunker(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = chunker.split_text(text)

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    text_embeddings = model.encode(chunks)
    question_embeddings = model.encode(questions_df['question'].tolist())

    similarities = cosine_similarity(question_embeddings, text_embeddings)
    closest_texts_indices = np.argsort(similarities, axis=1)[:, -k:][:, ::-1]  # Top k closest indices

    retrieved_intervals = [
        get_ranges(text, chunks, indicies) 
        for indicies in closest_texts_indices
    ]

    target_intervals = [
        [(ref['start_index'], ref['end_index']) for ref in json.loads(references)]
        for references in questions_df['references']
    ]

    retrieved_intervals = [
        merge_intervals(intervals)
        for intervals in retrieved_intervals
    ]

    target_intervals = [
        merge_intervals(intervals)
        for intervals in target_intervals
    ]

    results = []
    for retrieved, target in zip(retrieved_intervals, target_intervals):
        intersections = intersect_intervals(retrieved.copy(), target.copy())
        total_intersection = sum_of_ranges(intersections)
        total_retrieved = sum_of_ranges(retrieved)
        total_target = sum_of_ranges(target)

        IoU = total_intersection / (total_retrieved + total_target - total_intersection)
        precision = total_intersection / total_retrieved
        recall = total_intersection / total_target
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
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
    parser.add_argument('--k', type=int, default=1, help='Number of closest chunks to retrieve')
    args = parser.parse_args()

    metrics = run_experiment(
        questions_path=args.questions,
        text_path=args.text,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        k=args.k
    )
    
    for m in metrics:
        print(f"IoU: {m['IoU']:.2f} Precision: {m['Precision']:.2f}, "
              f"Recall: {m['Recall']:.2f}, F1: {m['F1']:.2f}")