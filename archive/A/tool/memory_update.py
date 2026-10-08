"""
Memory Update Tool - Update connections and content in existing memories
Agent tool for establishing relationships between memories
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


def update_memory_connections(
    memory_id: str,
    connections: Dict[str, Dict]
) -> Dict:
    """
    Add/update connections for a memory (merges with existing)
    
    Args:
        memory_id: Target memory ID
        connections: Connections to add/update
    
    Returns:
        Status dict
    """
    mem_file = Path("mem.json")
    
    if not mem_file.exists():
        return {"status": "error", "message": "mem.json not found"}
    
    with open(mem_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    memories = data.get('memory', [])
    
    # Find and update
    for mem in memories:
        if mem['id'] == memory_id:
            existing = mem.get('connections', {})
            existing.update(connections)
            mem['connections'] = existing
            # Be robust to older/hand-edited memories that may not have metadata.
            if not isinstance(mem.get("metadata"), dict):
                mem["metadata"] = {}
            mem["metadata"].setdefault("created", datetime.now().isoformat())
            mem["metadata"]["last_changed"] = datetime.now().isoformat()
            
            with open(mem_file, 'w', encoding='utf-8') as f:
                json.dump({"memory": memories}, f, indent=2, ensure_ascii=False)
            
            return {
                "status": "success",
                "memory_id": memory_id,
                "connections_count": len(mem['connections'])
            }
    
    return {"status": "error", "message": f"Memory {memory_id} not found"}


def batch_update_connections(updates: List[Dict]) -> Dict:
    """
    Batch update connections for multiple memories
    
    Args:
        updates: List of {"memory_id": "mem_001", "connections": {...}}
    
    Returns:
        Status dict
    """
    mem_file = Path("mem.json")
    
    if not mem_file.exists():
        return {"status": "error", "message": "mem.json not found"}
    
    with open(mem_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    memories = data.get('memory', [])
    
    id_to_index = {mem['id']: i for i, mem in enumerate(memories)}
    updated_ids = []
    
    for update in updates:
        mem_id = update.get('memory_id')
        connections = update.get('connections', {})
        
        if mem_id in id_to_index:
            idx = id_to_index[mem_id]
            existing = memories[idx].get('connections', {})
            existing.update(connections)
            memories[idx]['connections'] = existing
            if not isinstance(memories[idx].get("metadata"), dict):
                memories[idx]["metadata"] = {}
            memories[idx]["metadata"].setdefault("created", datetime.now().isoformat())
            memories[idx]["metadata"]["last_changed"] = datetime.now().isoformat()
            updated_ids.append(mem_id)
    
    with open(mem_file, 'w', encoding='utf-8') as f:
        json.dump({"memory": memories}, f, indent=2, ensure_ascii=False)
    
    return {
        "status": "success",
        "updated_count": len(updated_ids),
        "updated_ids": updated_ids
    }


def update_memory_content(
    memory_id: str,
    content_updates: Dict
) -> Dict:
    """
    Update content fields of a memory
    
    Args:
        memory_id: Target memory ID
        content_updates: Content fields to update
    
    Returns:
        Status dict
    """
    mem_file = Path("mem.json")
    
    if not mem_file.exists():
        return {"status": "error", "message": "mem.json not found"}
    
    with open(mem_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    memories = data.get('memory', [])
    
    for mem in memories:
        if mem['id'] == memory_id:
            if 'content' not in mem:
                mem['content'] = {}
            mem['content'].update(content_updates)
            # Be robust to older/hand-edited memories that may not have metadata.
            if not isinstance(mem.get("metadata"), dict):
                mem["metadata"] = {}
            mem["metadata"].setdefault("created", datetime.now().isoformat())
            mem["metadata"]["last_changed"] = datetime.now().isoformat()
            
            with open(mem_file, 'w', encoding='utf-8') as f:
                json.dump({"memory": memories}, f, indent=2, ensure_ascii=False)
            
            return {"status": "success", "memory_id": memory_id}
    
    return {"status": "error", "message": f"Memory {memory_id} not found"}


if __name__ == "__main__":
    # Test
    test_result = update_memory_connections(
        memory_id="mem_001",
        connections={
            "mem_002": {
                "relation": "related_to",
                "strength": 0.8,
                "reason": "Both discuss polymer ML"
            }
        }
    )
    print(test_result)

