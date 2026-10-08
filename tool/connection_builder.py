"""
Connection Builder Tool - Build connections between memories (Internalize)
Agent tool for establishing semantic relationships in memory graph
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def find_candidate_pairs(
    n_neighbors: int = 15,
    distance_range: Tuple[float, float] = (0.3, 0.5),
    embeddings_dir: str = "embeddings"
) -> Dict:
    """
    Find candidate memory pairs for connection building using FAISS
    
    Args:
        n_neighbors: Number of neighbors to find for each memory
        distance_range: (min_dist, max_dist) - only return pairs in this range
        embeddings_dir: Directory containing embeddings
    
    Returns:
        Dict with candidate pairs
    """
    try:
        mem_file = Path("mem.json")
        if not mem_file.exists():
            return {
                "status": "error",
                "message": "mem.json not found"
            }
        
        # Load memories
        with open(mem_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        memories = data.get('memory', [])
        
        # Load embeddings
        embeddings = []
        valid_indices = []
        
        for i, mem in enumerate(memories):
            emb_path = mem['metadata'].get('embedding')
            if emb_path and emb_path != 'null' and Path(emb_path).exists():
                try:
                    emb = np.load(emb_path)
                    embeddings.append(emb)
                    valid_indices.append(i)
                except Exception:
                    continue
        
        if len(embeddings) < 2:
            return {
                "status": "error",
                "message": "Not enough embeddings found"
            }
        
        # Create FAISS index
        embeddings_matrix = np.array(embeddings, dtype='float32')
        faiss.normalize_L2(embeddings_matrix)
        
        index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
        index.add(embeddings_matrix)
        
        # Find neighbors for each memory
        similarities, indices = index.search(
            embeddings_matrix,
            min(n_neighbors + 1, len(embeddings))  # +1 because it includes itself
        )
        
        # Build candidate pairs
        candidates = []
        seen_pairs = set()
        
        min_dist, max_dist = distance_range
        
        for i in range(len(embeddings)):
            mem_id_i = memories[valid_indices[i]]['id']
            
            for j_idx, sim in enumerate(similarities[i]):
                if j_idx == 0:  # Skip self
                    continue
                
                j = indices[i][j_idx]
                mem_id_j = memories[valid_indices[j]]['id']
                
                # Convert similarity to distance (1 - cosine_sim)
                distance = 1.0 - sim
                
                # Filter by distance range
                if not (min_dist <= distance <= max_dist):
                    continue
                
                # Avoid duplicates (order doesn't matter)
                pair_key = tuple(sorted([mem_id_i, mem_id_j]))
                if pair_key in seen_pairs:
                    continue
                
                seen_pairs.add(pair_key)
                candidates.append({
                    "mem_id_1": mem_id_i,
                    "mem_id_2": mem_id_j,
                    "similarity": float(sim),
                    "distance": float(distance)
                })
        
        return {
            "status": "success",
            "candidates": candidates,
            "count": len(candidates),
            "message": f"Found {len(candidates)} candidate pairs"
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to find candidates: {str(e)}"
        }


def get_memory_pair_info(mem_id_1: str, mem_id_2: str) -> str:
    """
    Get formatted info for a memory pair (for Agent to analyze)
    
    Args:
        mem_id_1: First memory ID
        mem_id_2: Second memory ID
    
    Returns:
        Formatted string with memory pair information
    """
    mem_file = Path("mem.json")
    with open(mem_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    memories = data.get('memory', [])
    id_to_mem = {mem['id']: mem for mem in memories}
    
    mem1 = id_to_mem.get(mem_id_1)
    mem2 = id_to_mem.get(mem_id_2)
    
    if not mem1 or not mem2:
        return f"Error: Memory not found"
    
    return f"""Memory Pair Analysis:

Memory 1: {mem1['id']}
Abstract: {mem1['abstract']}
Keywords: {mem1['keyword']}

Memory 2: {mem2['id']}
Abstract: {mem2['abstract']}
Keywords: {mem2['keyword']}

Analyze their relationship:
- Relation type: builds_upon / related_to / contradicts / enables / applies
- Strength: 0.0-1.0
- Reason: brief explanation
"""


def add_connection(
    mem_id_1: str,
    mem_id_2: str,
    relation: str,
    strength: float,
    reason: str
) -> Dict:
    """
    Add a bidirectional connection between two memories
    
    Args:
        mem_id_1: First memory ID
        mem_id_2: Second memory ID
        relation: Relation type (builds_upon/related_to/contradicts/enables/applies)
        strength: Connection strength 0.0-1.0
        reason: Brief explanation
    
    Returns:
        Status dict
    """
    from tool.memory_update import update_memory_connections
    
    connection_info = {
        "relation": relation,
        "strength": strength,
        "reason": reason
    }
    
    # Update both directions
    update_memory_connections(mem_id_1, {mem_id_2: connection_info})
    update_memory_connections(mem_id_2, {mem_id_1: connection_info})
    
    return {
        "status": "success",
        "message": f"Connected {mem_id_1} <-> {mem_id_2}"
    }


if __name__ == "__main__":
    # Test finding candidates
    result = find_candidate_pairs(n_neighbors=10, distance_range=(0.3, 0.6))
    print(result)

