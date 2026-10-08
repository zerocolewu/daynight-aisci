"""
Memory Read Tool - Read memories by ID with different detail levels
"""

import json
from pathlib import Path
from typing import List, Union, Literal, Dict, Any, Optional
from agno.tools import FunctionCall, tool

"""
def pre_hook(fc: FunctionCall):
    print(f"Pre-hook: {fc.function.name}")
    print(f"Arguments: {fc.arguments}")
    print(f"Result: {fc.result}")
@tool(pre_hook=pre_hook)
"""

def read_memory(
    memory_ids: Union[str, List[str]],
    detail: Union[Literal["abstract", "content", "full"], Dict[str, Any], str] = "content",
    memory_type: Optional[
        Union[
            Literal["knowledge", "idea", "experience", "resource", "integration"],
            List[Literal["knowledge", "idea", "experience", "resource", "integration"]],
        ]
    ] = None,
) -> dict:
    """    
    Args:
        memory_ids: Single ID or list of IDs (e.g., "mem_001" or ["mem_001", "mem_002"])
        detail: "abstract" | "content" | "full" 
        memory_type: Optional filter by memory type(s). If memory_ids is "all"/"*", returns all memories (optionally filtered).
    """
    # Handle incorrect dict input from some LLMs
    if isinstance(detail, dict):
        if detail.get('abstract'):
            detail = 'abstract'
        elif detail.get('full'):
            detail = 'full'
        else:
            detail = 'content'
    elif isinstance(detail, str) and detail not in ["abstract", "content", "full"]:
        # Fallback for other weird string inputs
        detail = "content"

    mem_file = Path("mem.json")
    
    if not mem_file.exists():
        return {"status": "error", "message": "mem.json not found"}
    
    with open(mem_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    memories = data.get('memory', [])
    
    # Normalize to list
    if isinstance(memory_ids, str):
        # Special: allow reading all memories (optionally filtered by type)
        if memory_ids.strip().lower() in {"all", "*"}:
            memory_ids = [m.get("id") for m in memories if isinstance(m, dict) and m.get("id")]
        else:
            memory_ids = [memory_ids]

    # Normalize type filter
    type_filter = None
    if memory_type is not None:
        if isinstance(memory_type, str):
            type_filter = {memory_type}
        elif isinstance(memory_type, list):
            type_filter = {t for t in memory_type if isinstance(t, str)}
    
    # Build ID to memory mapping
    id_to_mem = {mem['id']: mem for mem in memories}
    
    # Collect results
    results = []
    
    for mem_id in memory_ids:
        if mem_id not in id_to_mem:
            results.append({
                "id": mem_id,
                "error": "Not found"
            })
            continue
        
        mem = id_to_mem[mem_id]
        if type_filter is not None and mem.get("type") not in type_filter:
            # Skip mismatched types when filter is provided
            continue
        
        if detail == "abstract":
            results.append({
                "id": mem['id'],
                "type": mem['type'],
                "abstract": mem.get('abstract', ''),
            })
        
        elif detail == "content":
            results.append({
                "id": mem['id'],
                "type": mem['type'],
                "abstract": mem.get('abstract', ''),
                "keyword": mem.get('keyword', ''),
                "content": mem.get('content', ''),
                "connections": mem.get('connections', {})
            })
        
        elif detail == "full":
            # Only knowledge type has full paper content via local_path
            if mem['type'] == 'knowledge':
                content_data = mem.get('content', '')
                full_file_content = None
                
                # Read from local_path if content is dict and has local_path
                if isinstance(content_data, dict) and 'local_path' in content_data:
                    local_path = content_data['local_path']
                    local_file = Path(local_path)
                    if local_file.exists():
                        try:
                            with open(local_file, 'r', encoding='utf-8') as f:
                                full_file_content = f.read()
                        except Exception as e:
                            full_file_content = f"Error reading file: {e}"
                
                results.append({
                    "id": mem['id'],
                    "type": mem['type'],
                    "abstract": mem.get('abstract', ''),
                    "keyword": mem.get('keyword', ''),
                    "content": content_data,
                    "connections": mem.get('connections', {}),
                    "metadata": mem.get('metadata', {}),
                    "full_file_content": full_file_content
                })
            else:
                # For non-knowledge types (idea, experience), full is same as content + metadata
                results.append({
                    "id": mem['id'],
                    "type": mem['type'],
                    "abstract": mem.get('abstract', ''),
                    "keyword": mem.get('keyword', ''),
                    "content": mem.get('content', ''),
                    "connections": mem.get('connections', {}),
                    "metadata": mem.get('metadata', {})
                })
    
    return {
        "status": "success",
        "count": len(results),
        "memories": results
    }


if __name__ == "__main__":
    # Test multiple
    result = read_memory(["mem_001", "mem_002"], detail="abstract")
    print(result)

