"""
File Reader Tool - Read and parse local files
Agent tool for reading markdown, code, data files from the project
"""

import json
from pathlib import Path
from typing import Dict, Optional, List


def read_local_file(
    file_path: str,
    max_lines: Optional[int] = None,
    return_format: str = "text"
) -> Dict:
    """
    Read a local file from the project
    
    Args:
        file_path: Path to file (relative to project root or absolute)
        max_lines: Maximum number of lines to return (None = all)
        return_format: "text" or "json" or "lines"
    
    Returns:
        Dict with file content and metadata
    """
    try:
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            return {
                "status": "error",
                "message": f"File not found: {file_path}"
            }
        
        # Read file
        with open(file_path_obj, 'r', encoding='utf-8') as f:
            if max_lines:
                lines = [f.readline() for _ in range(max_lines)]
                content = ''.join(lines)
            else:
                content = f.read()
        
        # Format output
        if return_format == "json":
            try:
                parsed_content = json.loads(content)
                return {
                    "status": "success",
                    "file_path": str(file_path_obj),
                    "content": parsed_content,
                    "size": len(content),
                    "format": "json"
                }
            except json.JSONDecodeError:
                return {
                    "status": "error",
                    "message": "Invalid JSON file"
                }
        
        elif return_format == "lines":
            lines = content.split('\n')
            return {
                "status": "success",
                "file_path": str(file_path_obj),
                "content": lines,
                "line_count": len(lines),
                "format": "lines"
            }
        
        else:  # text
            return {
                "status": "success",
                "file_path": str(file_path_obj),
                "content": content,
                "size": len(content),
                "format": "text"
            }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to read file: {str(e)}"
        }


def list_markdown_files(
    directory: str = "lamnew-markdown",
    pattern: str = "*.md"
) -> Dict:
    """
    List all markdown files in a directory
    
    Args:
        directory: Directory to search
        pattern: File pattern (e.g., "*.md", "full.md")
    
    Returns:
        Dict with list of file paths
    """
    try:
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return {
                "status": "error",
                "message": f"Directory not found: {directory}"
            }
        
        # Find files
        if pattern == "full.md":
            files = list(dir_path.glob("*/full.md"))
        else:
            files = list(dir_path.rglob(pattern))
        
        file_list = [str(f) for f in files]
        
        return {
            "status": "success",
            "directory": str(dir_path),
            "pattern": pattern,
            "files": file_list,
            "count": len(file_list)
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to list files: {str(e)}"
        }


def read_paper_from_markdown(
    paper_path: str
) -> Dict:
    """
    Read a paper from markdown format (full.md)
    
    Args:
        paper_path: Path to full.md file
    
    Returns:
        Dict with paper content and metadata
    """
    try:
        path_obj = Path(paper_path)
        
        if not path_obj.exists():
            return {
                "status": "error",
                "message": f"Paper not found: {paper_path}"
            }
        
        # Read content
        with open(path_obj, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract title from folder name
        folder_name = path_obj.parent.name
        # Remove .pdf-uuid suffix
        import re
        title = re.sub(r'\.pdf-[a-f0-9\-]+$', '', folder_name)
        
        return {
            "status": "success",
            "paper_path": str(path_obj),
            "title": title,
            "content": content,
            "size": len(content),
            "folder": str(path_obj.parent)
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to read paper: {str(e)}"
        }


def search_files_by_keyword(
    keyword: str,
    directory: str = "lamnew-markdown",
    search_in_content: bool = False
) -> Dict:
    """
    Search files by keyword in filename or content
    
    Args:
        keyword: Search keyword
        directory: Directory to search in
        search_in_content: If True, search in file content (slower)
    
    Returns:
        Dict with matching files
    """
    try:
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return {
                "status": "error",
                "message": f"Directory not found: {directory}"
            }
        
        matches = []
        
        if search_in_content:
            # Search in content (only .md files)
            for md_file in dir_path.rglob("*.md"):
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if keyword.lower() in content.lower():
                        matches.append({
                            "file": str(md_file),
                            "match_type": "content"
                        })
                except Exception:
                    continue
        else:
            # Search in filename only
            for item in dir_path.rglob("*"):
                if keyword.lower() in item.name.lower():
                    matches.append({
                        "file": str(item),
                        "match_type": "filename"
                    })
        
        return {
            "status": "success",
            "keyword": keyword,
            "directory": str(dir_path),
            "matches": matches,
            "count": len(matches)
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to search files: {str(e)}"
        }


if __name__ == "__main__":
    # Test reading a file
    result = list_markdown_files(directory="lamnew-markdown", pattern="full.md")
    print(json.dumps(result, indent=2))

