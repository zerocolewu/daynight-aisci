"""
Memory Write Tool - Write new memories to mem.json
Agent tool for adding knowledge, ideas, experiences, or local files to memory
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Literal, Optional


def write_memory(
    memory_type: Literal["knowledge", "idea", "experience", "resource", "integration"],
    abstract: str,
    keyword: str,
    content: Dict,
    day: int = 0
) -> Dict:
    """
    Write a new memory to mem.json
    
    Args:
        memory_type: Type of memory (knowledge/idea/experience/resource/integration)
        abstract: Brief summary of the memory (100-200 words)
        keyword: Space-separated keywords
        content: Content dictionary (structure depends on type)
    
    Returns:
        Dict with status and memory_id
    """
    mem_file = Path("mem.json")
    
    try:
        # Load existing memories
        if mem_file.exists():
            with open(mem_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            memories = data.get('memory', [])
        else:
            memories = []
        
        # Generate unique ID
        mem_id = f"mem_{len(memories)+1:03d}"
        
        # Create new memory object
        new_memory = {
            "id": mem_id,
            "type": memory_type,
            "abstract": abstract,
            "keyword": keyword,
            "content": content,
            "connections": {},  # Will be filled by internalize_agent
            "metadata": {
                "created": datetime.now().isoformat(),
                "last_changed": datetime.now().isoformat(),
                "embedding": None,  # Will be filled by embed_memories.py
                "day": day  # Track which day this memory was created
            }
        }
        
        # Add to memories
        memories.append(new_memory)
        
        # Save back to file
        with open(mem_file, 'w', encoding='utf-8') as f:
            json.dump({"memory": memories}, f, indent=2, ensure_ascii=False)
        
        return {
            "status": "success",
            "memory_id": mem_id,
            "message": f"Memory {mem_id} created successfully",
            "total_memories": len(memories)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to write memory: {str(e)}"
        }


def batch_write_memories(memories: list) -> Dict:
    """
    Write multiple memories at once
    
    Args:
        memories: List of memory dicts, each with keys:
                  [memory_type, abstract, keyword, content, source]
    
    Returns:
        Dict with status and created memory_ids
    """
    mem_file = Path("mem.json")
    created_ids = []
    
    try:
        # Load existing memories
        if mem_file.exists():
            with open(mem_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            existing_memories = data.get('memory', [])
        else:
            existing_memories = []
        
        # Add each new memory
        for i, mem_data in enumerate(memories):
            mem_id = f"mem_{len(existing_memories)+i+1:03d}"
            
            new_memory = {
                "id": mem_id,
                "type": mem_data.get("memory_type", "knowledge"),
                "abstract": mem_data.get("abstract", ""),
                "keyword": mem_data.get("keyword", ""),
                "content": mem_data.get("content", {}),
                "connections": {},
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "last_changed": datetime.now().isoformat(),
                    "source": mem_data.get("source", "agent_generated"),
                    "embedding": None
                }
            }
            
            existing_memories.append(new_memory)
            created_ids.append(mem_id)
        
        # Save all at once
        with open(mem_file, 'w', encoding='utf-8') as f:
            json.dump({"memory": existing_memories}, f, indent=2, ensure_ascii=False)
        
        return {
            "status": "success",
            "created_ids": created_ids,
            "count": len(created_ids),
            "message": f"Successfully created {len(created_ids)} memories"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to batch write memories: {str(e)}"
        }


if __name__ == "__main__":
    # Test the tool
    test_result = write_memory(
        memory_type="idea",
        abstract="Test idea for polymer prediction using attention mechanisms",
        keyword="test polymer attention prediction",
        content={
            "motivation": "This is a test",
            "approach": "Test approach",
            "expected_impact": "Test impact"
        },
        source="test"
    )
    
    print(json.dumps(test_result, indent=2))

