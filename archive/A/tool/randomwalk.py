"""
Idea Generator Tool - Generate research ideas through memory graph random walk (Associate)
Agent tool for creative idea generation via knowledge recombination
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def random_walk_memory(
    start_memory_id: Optional[str] = None,
    walk_length: int = 5,
    min_connection_strength: float = 0.5
) -> Dict:
    """
    Perform random walk on memory graph
    
    Args:
        start_memory_id: Starting memory ID (None = random start)
        walk_length: Number of steps in random walk
        min_connection_strength: Minimum connection strength to follow
    
    Returns:
        Dict with walk path and collected memories
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
        
        if len(memories) == 0:
            return {
                "status": "error",
                "message": "No memories found"
            }
        
        # Create ID to memory mapping
        id_to_mem = {mem['id']: mem for mem in memories}
        
        # Select start memory
        if start_memory_id:
            if start_memory_id not in id_to_mem:
                return {
                    "status": "error",
                    "message": f"Start memory {start_memory_id} not found"
                }
            current_id = start_memory_id
        else:
            # Random start from memories with connections
            memories_with_connections = [
                mem['id'] for mem in memories
                if mem.get('connections') and len(mem['connections']) > 0
            ]
            
            if not memories_with_connections:
                # If no connections, just pick random memories
                walk_path = random.sample([m['id'] for m in memories], min(walk_length, len(memories)))
                return {
                    "status": "success",
                    "walk_path": walk_path,
                    "memories": [id_to_mem[mid] for mid in walk_path],
                    "note": "No connections found, returning random memories"
                }
            
            current_id = random.choice(memories_with_connections)
        
        # Perform random walk
        walk_path = [current_id]
        visited = {current_id}
        
        for _ in range(walk_length - 1):
            current_mem = id_to_mem[current_id]
            connections = current_mem.get('connections', {})
            
            # Filter by strength
            valid_connections = [
                (next_id, conn_info)
                for next_id, conn_info in connections.items()
                if conn_info.get('strength', 0) >= min_connection_strength
                and next_id in id_to_mem
                and next_id not in visited  # Avoid cycles
            ]
            
            if not valid_connections:
                # Dead end or all visited - try jumping to unvisited memory with connections
                unvisited_with_connections = [
                    mem['id'] for mem in memories
                    if mem['id'] not in visited
                    and mem.get('connections')
                    and len(mem['connections']) > 0
                ]
                
                if unvisited_with_connections:
                    current_id = random.choice(unvisited_with_connections)
                else:
                    # No more unvisited memories with connections
                    break
            else:
                # Weighted random selection based on strength
                next_id = random.choices(
                    [c[0] for c in valid_connections],
                    weights=[c[1].get('strength', 0.5) for c in valid_connections],
                    k=1
                )[0]
                current_id = next_id
            
            walk_path.append(current_id)
            visited.add(current_id)
        
        # Collect memories along path
        memories_on_path = [id_to_mem[mid] for mid in walk_path]
        
        return {
            "status": "success",
            "walk_path": walk_path,
            "walk_length": len(walk_path),
            "memories": memories_on_path
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed random walk: {str(e)}"
        }


def format_walk_for_ideation(walk_result: Dict, focus_area: Optional[str] = None) -> str:
    """
    Format random walk result for Agent to generate ideas
    
    Args:
        walk_result: Result from random_walk_memory
        focus_area: Optional focus area
    
    Returns:
        Formatted prompt string for Agent
    """
    if walk_result["status"] != "success":
        return "Error: No valid walk result"
    
    memories = walk_result["memories"]
    
    memories_text = []
    for i, mem in enumerate(memories):
        memories_text.append(f"""
Memory {i+1} ({mem['id']}):
Type: {mem['type']}
Abstract: {mem['abstract']}
Keywords: {mem['keyword']}
""")
    
    focus_text = f"\nFocus area: {focus_area}" if focus_area else ""
    
    return f"""You encountered these memories through random walk:
{''.join(memories_text)}
{focus_text}

Generate a novel research idea combining these insights.
Use write_memory to save the idea with type="idea".
"""




if __name__ == "__main__":
    # Test random walk
    result = random_walk_memory(walk_length=4)
    if result["status"] == "success":
        print(f"Walk path: {' -> '.join(result['walk_path'])}")

