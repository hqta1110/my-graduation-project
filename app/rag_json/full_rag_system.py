# graph_rag_system.py

import json
import pickle
import numpy as np
import os
import faiss
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize
import time
from functools import lru_cache
from collections import defaultdict # NEW

from rag_json.config import RAGConfig

class VietnamesePlantRAG:
    """Optimized RAG system with FAISS for fast similarity search"""
    
    def __init__(self, cache_dir):
        # ... (no changes in __init__)
        print("🚀 Initializing Optimized Vietnamese Plant RAG System with FAISS...")
        
        # Initialize embedding model
        print(f"📥 Loading BGE-M3 model: {RAGConfig.EMBEDDING_MODEL}...")
        self.embedding_model = SentenceTransformer(RAGConfig.EMBEDDING_MODEL, cache_folder=cache_dir)
        print("✅ BGE-M3 model loaded!")
        
        # Initialize Gemini client
        if not RAGConfig.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set.")
        self.gemini_client = genai.Client(api_key=RAGConfig.GEMINI_API_KEY)
        
        # Storage components
        self.plant_data = {}
        self.plant_index = []
        self.embedding_dimension = None
        
        # FAISS components
        self.faiss_index = None
        self.faiss_index_file = RAGConfig.EMBEDDINGS_FILE.replace('.pkl', '_faiss.index')
        
        # BM25 components
        self.tokenized_corpus = []
        self.bm25_model = None
        
        # Performance tracking
        self.query_cache = {}
        self.embedding_cache_size = 1000
        
        # Load existing data
        self.load_indexed_data()
        print("✅ Optimized RAG System initialized!")

    def _tokenize_vietnamese(self, text: str) -> List[str]:
        """Cached Vietnamese tokenization"""
        return word_tokenize(text.lower(), format="text").split()

    @lru_cache(maxsize=1000)
    def _cached_tokenize(self, text: str) -> tuple:
        """LRU cached tokenization for frequently used texts"""
        return tuple(self._tokenize_vietnamese(text))

    # MODIFIED: Renamed to be more specific
    def _create_medical_context(self, scientific_name: str, plant_graph_data: Dict) -> str:
        """Create medical-focused context text from the graph data."""
        context_parts = []
        plant_info = plant_graph_data.get('plant_info', {})
        vietnamese_name = plant_info.get('vietnamese_name')

        # Use Vietnamese name if available, otherwise scientific name
        display_name = vietnamese_name if vietnamese_name else scientific_name
        context_parts.append(f"Cây thuốc: {display_name} ({scientific_name})")

        treats_data = plant_graph_data.get('treats', {})
        if treats_data:
            context_parts.append("Công dụng chữa bệnh:")
            for condition, details in treats_data.items():
                prep = details.get('preparation', '')
                dose = details.get('dosage', '')
                condition_text = f"- Chữa {condition}"
                if prep: condition_text += f" (Cách dùng: {prep})"
                if dose: condition_text += f" (Liều dùng: {dose})"
                context_parts.append(condition_text)
        
        # Return None if no meaningful medical data was found
        return "\n".join(context_parts) if treats_data else None

    # NEW: Function to create context from the original metadata file
    def _create_botanical_context(self, scientific_name: str, plant_original_data: Dict) -> str:
        """Create general botanical context text from original metadata."""
        context_parts = []
        
        vietnamese_name = plant_original_data.get("Tên tiếng Việt", "")
        display_name = vietnamese_name if vietnamese_name else scientific_name
        context_parts.append(f"Cây: {display_name} ({scientific_name})")
        
        # Select relevant fields for the context
        fields_to_include = {
            "Mô tả": "Mô tả",
            "Sinh học & Sinh thái": "Sinh học và Sinh thái",
            "Phân bố": "Phân bố",
            "Giá trị": "Giá trị sử dụng",
            "Tên họ tiếng Việt": "Họ"
        }

        has_content = False
        for key, display_key in fields_to_include.items():
            value = plant_original_data.get(key)
            if value and value.lower() not in ["không có thông tin", ""]:
                context_parts.append(f"- {display_key}: {value}")
                has_content = True

        # Return None if no meaningful botanical data was found
        return "\n".join(context_parts) if has_content else None


    def has_meaningful_data(self, plant_full_data: Dict) -> bool:
        """Check if plant has meaningful medical data"""
        # This function is less critical now as we check context creation directly, but we keep it for now.
        return bool(plant_full_data.get('treats'))

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        # ... (no changes in this function)
        dimension = embeddings.shape[1]
        n_embeddings = embeddings.shape[0]
        
        print(f"🔧 Building FAISS index for {n_embeddings} embeddings with dimension {dimension}")
        
        # Choose index type based on dataset size
        if n_embeddings < 1000:
            # Small dataset: use exact search
            index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
            print("📊 Using exact search (IndexFlatIP) for small dataset")
            
        elif n_embeddings < 10000:
            # Medium dataset: use IVF with few clusters
            nlist = min(100, n_embeddings // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            print(f"📊 Using IVF index with {nlist} clusters for medium dataset")
            
        else:
            # Large dataset: use IVF with PQ compression
            nlist = min(1000, n_embeddings // 50)
            m = 8  # Number of subquantizers
            bits = 8  # Bits per subquantizer
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, bits)
            print(f"📊 Using IVF-PQ index with {nlist} clusters for large dataset")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Train index if needed
        if hasattr(index, 'train'):
            print("🎯 Training FAISS index...")
            index.train(embeddings)
        
        # Add embeddings to index
        print("📥 Adding embeddings to FAISS index...")
        index.add(embeddings)
        
        print("✅ FAISS index built successfully!")
        return index

    # MODIFIED: The core logic for building embeddings is now much more powerful.
    def build_embeddings(self, graph_json_file: str, original_json_file: str):
        """Build embeddings and FAISS index from multiple data sources."""
        print(f"📖 Loading medical graph data from {graph_json_file}")
        print(f"📖 Loading botanical data from {original_json_file}")
        
        start_time = time.time()
        
        try:
            with open(graph_json_file, 'r', encoding='utf-8') as f:
                graph_data = json.load(f).get('plant_graph', {})
            with open(original_json_file, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
        except Exception as e:
            print(f"❌ Error loading JSON files: {e}")
            return

        # Create a lookup map from scientific name to the key in graph_data
        # This is crucial for linking the two datasets.
        s_name_to_graph_key = {
            entry.get('plant_info', {}).get('scientific_name'): key
            for key, entry in graph_data.items()
            if entry.get('plant_info', {}).get('scientific_name')
        }
        
        print(f"🌿 Processing {len(original_data)} plants from botanical source...")

        context_chunks = []
        plant_index_temp = []
        tokenized_corpus_temp = []

        for scientific_name, plant_entry in original_data.items():
            # 1. Create botanical context chunk
            botanical_context = self._create_botanical_context(scientific_name, plant_entry)
            if botanical_context:
                context_chunks.append(botanical_context)
                plant_index_temp.append({
                    'plant_name': scientific_name,
                    'vietnamese_name': plant_entry.get("Tên tiếng Việt", ""),
                    'chunk_type': 'botanical',
                    'context_text': botanical_context,
                    'original_data': plant_entry # Store the full original data
                })
                tokenized_corpus_temp.append(self._tokenize_vietnamese(botanical_context))

            # 2. Create medical context chunk (if available)
            graph_key = s_name_to_graph_key.get(scientific_name)
            if graph_key and graph_key in graph_data:
                medical_context = self._create_medical_context(scientific_name, graph_data[graph_key])
                if medical_context:
                    context_chunks.append(medical_context)
                    plant_index_temp.append({
                        'plant_name': scientific_name,
                        'vietnamese_name': graph_data[graph_key].get('plant_info', {}).get('vietnamese_name', ""),
                        'chunk_type': 'medical',
                        'context_text': medical_context,
                        'original_data': graph_data[graph_key] # Store the full graph data
                    })
                    tokenized_corpus_temp.append(self._tokenize_vietnamese(medical_context))

        if not context_chunks:
            print("❌ No context chunks could be created from the data!")
            return

        print(f"📊 Created {len(context_chunks)} total context chunks for embedding.")

        # --- The rest of the function is the same, it just operates on the new `context_chunks` ---
        print("🔍 Generating BGE-M3 embeddings in batches...")
        batch_size = 4
        all_embeddings = []
        import torch
        with torch.no_grad():
            for i in range(0, len(context_chunks), batch_size):
                batch_texts = context_chunks[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch_texts, 
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
                all_embeddings.append(batch_embeddings)
                print(f"   ✓ Embedded batch {i//batch_size + 1}/{(len(context_chunks) + batch_size - 1)//batch_size}")
        
        plant_embeddings = np.vstack(all_embeddings)
        self.embedding_dimension = plant_embeddings.shape[1]
        
        self.faiss_index = self.build_faiss_index(plant_embeddings)
        self.plant_index = plant_index_temp

        print("🔍 Building BM25 index...")
        self.tokenized_corpus = tokenized_corpus_temp
        self.bm25_model = BM25Okapi(self.tokenized_corpus)

        self.save_indexed_data()
        
        total_time = time.time() - start_time
        print(f"✅ Built indexes for {len(context_chunks)} chunks in {total_time:.2f} seconds!")

    def save_indexed_data(self):
        # ... (no changes in this function)
        try:
            os.makedirs(os.path.dirname(RAGConfig.EMBEDDINGS_FILE), exist_ok=True)

            # Save FAISS index
            faiss.write_index(self.faiss_index, self.faiss_index_file)
            
            # Save other components
            with open(RAGConfig.PLANT_INDEX_FILE, 'wb') as f:
                pickle.dump(self.plant_index, f)

            with open(RAGConfig.BM25_CORPUS_FILE, 'wb') as f:
                pickle.dump(self.tokenized_corpus, f)
            
            with open(RAGConfig.BM25_MODEL_FILE, 'wb') as f:
                pickle.dump(self.bm25_model, f)
            
            # Save embedding dimension for later loading
            meta_info = {'embedding_dimension': self.embedding_dimension}
            with open(RAGConfig.EMBEDDINGS_FILE.replace('.pkl', '_meta.pkl'), 'wb') as f:
                pickle.dump(meta_info, f)
            
            print(f"💾 Optimized indexes saved successfully!")
            
        except Exception as e:
            print(f"❌ Error saving indexed data: {e}")

    def load_indexed_data(self):
        # ... (no changes in this function)
        try:
            if (os.path.exists(self.faiss_index_file) and 
                os.path.exists(RAGConfig.PLANT_INDEX_FILE) and
                os.path.exists(RAGConfig.BM25_CORPUS_FILE) and
                os.path.exists(RAGConfig.BM25_MODEL_FILE)):
                
                # Load FAISS index
                self.faiss_index = faiss.read_index(self.faiss_index_file)
                
                # Load metadata
                meta_file = RAGConfig.EMBEDDINGS_FILE.replace('.pkl', '_meta.pkl')
                if os.path.exists(meta_file):
                    with open(meta_file, 'rb') as f:
                        meta_info = pickle.load(f)
                        self.embedding_dimension = meta_info['embedding_dimension']
                
                # Load other components
                with open(RAGConfig.PLANT_INDEX_FILE, 'rb') as f:
                    self.plant_index = pickle.load(f)

                with open(RAGConfig.BM25_CORPUS_FILE, 'rb') as f:
                    self.tokenized_corpus = pickle.load(f)
                
                with open(RAGConfig.BM25_MODEL_FILE, 'rb') as f:
                    self.bm25_model = pickle.load(f)
                
                print(f"📥 Loaded optimized indexes for {len(self.plant_index)} chunks.")
                
            else:
                print("⚠️ No complete optimized indexed data found.")
                
        except Exception as e:
            print(f"⚠️ Could not load optimized indexed data: {e}")

    def _reciprocal_rank_fusion(self, ranked_lists: List[List[Tuple[int, float]]]) -> List[Tuple[int, float]]:
        # ... (no changes in this function)
        fused_scores = {}
        k = RAGConfig.RRF_K

        for ranked_list in ranked_lists:
            for rank, (doc_idx, _) in enumerate(ranked_list):
                fused_scores[doc_idx] = fused_scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)

        sorted_fused_scores = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_fused_scores

    def optimized_dense_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        # ... (no changes in this function)
        if self.faiss_index is None:
            print("❌ FAISS index not available.")
            return []
        
        # Check cache first
        cache_key = f"dense_{hash(query)}_{top_k}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        start_time = time.time()
        
        # Generate and normalize query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search with FAISS
        similarities, indices = self.faiss_index.search(query_embedding, top_k * 2)  # Get more for filtering
        
        # Convert to list of tuples and filter by threshold
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if similarity >= RAGConfig.MIN_SIMILARITY_THRESHOLD and idx != -1:  # -1 means not found
                results.append((int(idx), float(similarity)))
        
        # Cache results (limit cache size)
        if len(self.query_cache) < self.embedding_cache_size:
            self.query_cache[cache_key] = results
        
        search_time = time.time() - start_time
        print(f"FAISS search completed in {search_time:.3f}s, found {len(results)} results")
        
        return results
        
    def hybrid_search(self, query: str, top_k: int = None) -> List[Tuple[Dict, float]]:
        # ... (no functional changes, but logic is now more powerful)
        if top_k is None:
            top_k = RAGConfig.TOP_K_RETRIEVAL

        if self.faiss_index is None or self.bm25_model is None:
            print("❌ Indexes not available for hybrid search.")
            return []

        candidate_k = top_k * 3 
        print("🚀 Performing optimized dense retrieval...")
        dense_ranked_list = self.optimized_dense_search(query, candidate_k)

        print("🔍 Performing sparse retrieval...")
        cache_key = f"sparse_{hash(query)}_{candidate_k}"
        
        if cache_key in self.query_cache:
            sparse_ranked_list = self.query_cache[cache_key]
        else:
            tokenized_query = self._tokenize_vietnamese(query)
            bm25_scores = self.bm25_model.get_scores(tokenized_query)
            
            sparse_ranked_list = []
            for idx in np.argsort(bm25_scores)[::-1]:
                if bm25_scores[idx] > 6.0:
                    sparse_ranked_list.append((idx, bm25_scores[idx]))
                if len(sparse_ranked_list) >= candidate_k:
                    break
            if len(self.query_cache) < self.embedding_cache_size:
                self.query_cache[cache_key] = sparse_ranked_list

        print("🔄 Fusing results...")
        fused_results = self._reciprocal_rank_fusion([dense_ranked_list, sparse_ranked_list])
        
        final_results = []
        seen_indices = set()
        for doc_idx, rrf_score in fused_results:
            if len(final_results) >= top_k:
                break
            if doc_idx not in seen_indices and doc_idx < len(self.plant_index):
                seen_indices.add(doc_idx)
                chunk_info_dict = self.plant_index[doc_idx]
                final_results.append((chunk_info_dict, rrf_score))

        print(f"✅ Optimized hybrid search returned {len(final_results)} chunks")
        return final_results

    # MODIFIED: This function now intelligently groups chunks by plant.
    def build_context_for_generation(self, search_results: List[Tuple[Dict, float]]) -> str:
        """Build context by grouping retrieved chunks by plant."""
        if not search_results:
            return "Không tìm thấy thông tin cây thuốc phù hợp."

        # Group contexts by plant name to avoid redundancy
        grouped_contexts = defaultdict(list)
        for chunk_info, score in search_results:
            plant_name = chunk_info['plant_name']
            grouped_contexts[plant_name].append((chunk_info, score))

        context_parts = ["THÔNG TIN LIÊN QUAN TỪ CÁC LOÀI CÂY:\n"]
        
        plant_counter = 1
        for plant_name, chunks in grouped_contexts.items():
            # Get the best vietnamese name available for display
            v_name = next((c[0]['vietnamese_name'] for c in chunks if c[0]['vietnamese_name']), '')
            display_name = f"{v_name} ({plant_name})" if v_name else plant_name
            
            context_parts.append(f"--- Cây {plant_counter}: {display_name} ---")
            
            # Sort chunks by score descending to show most relevant first
            chunks.sort(key=lambda x: x[1], reverse=True)
            
            for chunk_info, score in chunks:
                chunk_type = chunk_info.get('chunk_type', 'general')
                context_text = chunk_info['context_text']
                
                # Remove the first line of the context text as we already have the header
                context_text_body = "\n".join(context_text.split("\n")[1:])
                
                # Add a header for the type of information
                if chunk_type == 'medical':
                    context_parts.append("[Thông tin Y học]")
                elif chunk_type == 'botanical':
                    context_parts.append("[Thông tin Thực vật học]")
                
                context_parts.append(context_text_body)
                context_parts.append(f"(Độ liên quan của đoạn này: {score:.3f})")
                context_parts.append("") # Add a blank line for readability
            
            plant_counter += 1
            
        return "\n".join(context_parts)
    
    def generate_answer(self, question: str, context: str) -> str:
        # ... (no changes in this function)
        # The prompt is general enough to work perfectly with the new, richer context.
        system_prompt = """
Bạn là chuyên gia y học cổ truyền Việt Nam với nhiều năm kinh nghiệm. Nhiệm vụ của bạn là trả lời câu hỏi về cây thuốc dựa trên thông tin được cung cấp.

YÊU CẦU:
1. Trả lời bằng tiếng Việt tự nhiên và dễ hiểu.
2. Dựa trên thông tin được cung cấp, không thêm kiến thức bên ngoài nếu không được yêu cầu rõ ràng.
3. Đưa ra lời khuyên thực tế về cách sử dụng, liều dùng và cách chế biến cụ thể.
4. Nếu có nhiều lựa chọn, sắp xếp theo mức độ phù hợp hoặc ưu tiên.
5. Luôn khuyên tham khảo ý kiến chuyên gia y tế trước khi áp dụng.

LƯU Ý AN TOÀN:
- Nhấn mạnh tầm quan trọng của việc tham khảo bác sĩ/thầy thuốc.
- Cảnh báo về liều dùng và cách sử dụng an toàn.
- Không khuyến khích tự điều trị cho các bệnh nghiêm trọng.
"""

        user_prompt = f"""
Câu hỏi: {question}

{context}

Hãy trả lời câu hỏi dựa trên thông tin cây thuốc ở trên. Cung cấp thông tin chi tiết về:
- Cây thuốc phù hợp nhất.
- Cách chế biến và sử dụng.
- Liều dùng cụ thể.
- Lưu ý an toàn.
"""

        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_prompt)],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                system_instruction=[types.Part.from_text(text=system_prompt)],
                temperature=0.3,
            )
            
            response = self.gemini_client.models.generate_content(
                model=RAGConfig.GEMINI_MODEL,
                contents=contents,
                config=generate_content_config,
            )
            
            return response.text
            
        except Exception as e:
            print(f"❌ Lỗi khi tạo câu trả lời từ Gemini: {e}")
            return f"Xin lỗi, tôi gặp sự cố khi tạo câu trả lời. Vui lòng thử lại sau. (Lỗi: {e})"

    def query(self, question: str) -> Dict[str, Any]:
        # MODIFIED to generate the answer by default
        print(f"🔍 Optimized query: {question}")
        start_time = time.time()
        
        search_results = self.hybrid_search(question)
        
        if not search_results:
            return {
                'question': question,
                'answer': "Xin lỗi, tôi không tìm thấy thông tin cây thuốc phù hợp với câu hỏi của bạn. Vui lòng thử câu hỏi khác.",
                'relevant_plants': [],
                'search_results_count': 0,
                'context_used': "Không có thông tin liên quan được tìm thấy.",
                'processing_time': time.time() - start_time
            }
        
        context = self.build_context_for_generation(search_results)
        answer = self.generate_answer(question, context) # Re-enabled generation
        
        # Get unique plant names from the results
        relevant_plants = sorted(list(set(res[0]['plant_name'] for res in search_results)))
        
        total_time = time.time() - start_time
        print(f"⚡ Total query processed in {total_time:.3f} seconds")
        
        return {
            'question': question,
            'answer': answer, # MODIFIED
            'relevant_plants': relevant_plants,
            'search_results_count': len(search_results), # This is chunk count
            'context_for_llm': context,
            'processing_time': total_time
        }

    def clear_cache(self):
        # ... (no changes)
        self.query_cache.clear()
        print("🧹 Query cache cleared")

    def get_performance_stats(self) -> Dict[str, Any]:
        # MODIFIED to reflect chunks instead of plants
        return {
            'total_chunks': len(self.plant_index) if self.plant_index else 0,
            'embedding_dimension': self.embedding_dimension,
            'faiss_index_type': type(self.faiss_index).__name__ if self.faiss_index else None,
            'cache_size': len(self.query_cache),
            'bm25_corpus_size': len(self.tokenized_corpus) if self.tokenized_corpus else 0
        }