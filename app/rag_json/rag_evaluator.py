# evaluate_rag.py

import json
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

from graph_rag_system import VietnamesePlantRAG
from config import RAGConfig

# --- Metric Calculation Functions ---

def calculate_hit_rate(retrieved_docs: list, ground_truth_docs: list, k: int) -> int:
    """Returns 1 if any ground truth doc is in the top K retrieved docs, else 0."""
    return 1 if any(doc in ground_truth_docs for doc in retrieved_docs[:k]) else 0

def calculate_precision_at_k(retrieved_docs: list, ground_truth_docs: list, k: int) -> float:
    """Calculates precision@k."""
    retrieved_top_k = set(retrieved_docs[:k])
    ground_truth_set = set(ground_truth_docs)
    return len(retrieved_top_k & ground_truth_set) / k

def calculate_mrr(retrieved_docs: list, ground_truth_docs: list) -> float:
    """Calculates Mean Reciprocal Rank."""
    for i, doc in enumerate(retrieved_docs):
        if doc in ground_truth_docs:
            return 1.0 / (i + 1)
    return 0.0

def calculate_ndcg_at_k(retrieved_docs: list, ground_truth_docs: list, k: int) -> float:
    """Calculates Normalized Discounted Cumulative Gain @K."""
    # Create a relevance list for retrieved docs
    relevance = [1 if doc in ground_truth_docs else 0 for doc in retrieved_docs[:k]]
    
    # DCG (Discounted Cumulative Gain)
    dcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(relevance)])
    
    # IDCG (Ideal Discounted Cumulative Gain)
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance)])
    
    return dcg / idcg if idcg > 0 else 0.0


# --- Main Evaluation Script ---

class RAGEvaluator:
    def __init__(self, test_set_path: str, plant_graph_path: str, top_k: int = 5):
        print("Initializing RAG Evaluator...")
        self.rag_system = VietnamesePlantRAG()
        self.top_k = top_k
        self.test_set = self._load_json(test_set_path)
        self.plant_graph = self._load_json(plant_graph_path).get('plant_graph', {})
        self.plant_name_to_doc_id = self._create_name_to_doc_id_map()
        print(f"Loaded {len(self.test_set)} test questions.")
        print(f"Evaluation will be performed for top_k = {self.top_k}")

    def _load_json(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _create_name_to_doc_id_map(self):
        """Creates a mapping from scientific_name to doc_id for easy lookup."""
        mapping = {}
        for plant_key, data in self.plant_graph.items():
            info = data.get('plant_info', {})
            scientific_name = info.get('scientific_name')
            doc_id = info.get('doc_id')
            if scientific_name and doc_id:
                mapping[scientific_name] = doc_id
        return mapping

    def _get_doc_ids_from_results(self, results: list) -> list:
        """Converts search results into a list of doc_ids."""
        doc_ids = []
        for result_tuple in results:
            plant_info_dict = result_tuple[0]
            plant_name = plant_info_dict.get('plant_name')
            doc_id = self.plant_name_to_doc_id.get(plant_name)
            if doc_id:
                doc_ids.append(doc_id)
        return doc_ids

    def _run_sparse_only(self, query: str, k: int) -> list:
        """Replicates the sparse-only part of the hybrid search."""
        tokenized_query = self.rag_system._tokenize_vietnamese(query)
        bm25_scores = self.rag_system.bm25_model.get_scores(tokenized_query)
        
        # Get top k indices from BM25 scores
        top_indices = np.argsort(bm25_scores)[::-1][:k]
        
        # Convert indices to the format expected by _get_doc_ids_from_results
        sparse_results = []
        for idx in top_indices:
            plant_info_dict = self.rag_system.plant_index[idx]
            score = bm25_scores[idx]
            sparse_results.append((plant_info_dict, score))
        return sparse_results

    def evaluate(self):
        """Runs the full evaluation across the test set."""
        evaluation_results = []

        for item in tqdm(self.test_set, desc="Evaluating Questions"):
            question = item['question']
            ground_truth_docs = item['context']
            
            # --- 1. Dense Search (FAISS) ---
            dense_results_raw = self.rag_system.optimized_dense_search(question, self.top_k)
            # The dense search already returns index and score. We need to map index to plant info.
            dense_results_mapped = [(self.rag_system.plant_index[idx], score) for idx, score in dense_results_raw]
            dense_retrieved_docs = self._get_doc_ids_from_results(dense_results_mapped)
            
            # --- 2. Sparse Search (BM25) ---
            sparse_results_raw = self._run_sparse_only(question, self.top_k)
            sparse_retrieved_docs = self._get_doc_ids_from_results(sparse_results_raw)

            # --- 3. Hybrid Search (FAISS + BM25 + RRF) ---
            hybrid_results_raw = self.rag_system.hybrid_search(question, self.top_k)
            hybrid_retrieved_docs = self._get_doc_ids_from_results(hybrid_results_raw)

            # --- Calculate metrics for each method ---
            result_row = {
                'question': question,
                'ground_truth': ground_truth_docs,
                'dense_retrieved': dense_retrieved_docs,
                'sparse_retrieved': sparse_retrieved_docs,
                'hybrid_retrieved': hybrid_retrieved_docs,
                
                'dense_hit_rate': calculate_hit_rate(dense_retrieved_docs, ground_truth_docs, self.top_k),
                'sparse_hit_rate': calculate_hit_rate(sparse_retrieved_docs, ground_truth_docs, self.top_k),
                'hybrid_hit_rate': calculate_hit_rate(hybrid_retrieved_docs, ground_truth_docs, self.top_k),

                'dense_precision': calculate_precision_at_k(dense_retrieved_docs, ground_truth_docs, self.top_k),
                'sparse_precision': calculate_precision_at_k(sparse_retrieved_docs, ground_truth_docs, self.top_k),
                'hybrid_precision': calculate_precision_at_k(hybrid_retrieved_docs, ground_truth_docs, self.top_k),
                
                'dense_mrr': calculate_mrr(dense_retrieved_docs, ground_truth_docs),
                'sparse_mrr': calculate_mrr(sparse_retrieved_docs, ground_truth_docs),
                'hybrid_mrr': calculate_mrr(hybrid_retrieved_docs, ground_truth_docs),

                'dense_ndcg': calculate_ndcg_at_k(dense_retrieved_docs, ground_truth_docs, self.top_k),
                'sparse_ndcg': calculate_ndcg_at_k(sparse_retrieved_docs, ground_truth_docs, self.top_k),
                'hybrid_ndcg': calculate_ndcg_at_k(hybrid_retrieved_docs, ground_truth_docs, self.top_k),
            }
            evaluation_results.append(result_row)
            
        return pd.DataFrame(evaluation_results)


if __name__ == '__main__':
    # --- Configuration ---
    TEST_SET_FILE = '/home/sora/pretrain_llm/infer/app/rag_json/test.json' # IMPORTANT: Change this path
    PLANT_GRAPH_FILE = RAGConfig.PLANT_GRAPH_PATH
    TOP_K = 10

    # --- Run Evaluation ---
    start_time = time.time()
    evaluator = RAGEvaluator(
        test_set_path=TEST_SET_FILE,
        plant_graph_path=PLANT_GRAPH_FILE,
        top_k=TOP_K
    )
    results_df = evaluator.evaluate()
    end_time = time.time()

    # --- Display Results ---
    print("\n" + "="*50)
    print("           RAG EVALUATION RESULTS")
    print("="*50 + "\n")
    
    # Define metric columns for aggregation
    metric_cols = [
        'dense_hit_rate', 'sparse_hit_rate', 'hybrid_hit_rate',
        'dense_precision', 'sparse_precision', 'hybrid_precision',
        'dense_mrr', 'sparse_mrr', 'hybrid_mrr',
        'dense_ndcg', 'sparse_ndcg', 'hybrid_ndcg'
    ]
    
    summary = results_df[metric_cols].mean().to_frame('Average Score')
    print("--- Overall Performance Summary ---\n")
    print(summary)
    print("\n" + "-"*50)
    
    print("\n--- Detailed Results per Question ---\n")
    display_cols = ['question', 'ground_truth', 'hybrid_retrieved', 'hybrid_ndcg', 'hybrid_mrr']
    print(results_df[display_cols])
    
    # Save results to a file for further analysis
    results_df.to_csv('rag_evaluation_results.csv', index=False, encoding='utf-8-sig')
    print("\n\nFull evaluation results saved to 'rag_evaluation_results.csv'")
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds.")