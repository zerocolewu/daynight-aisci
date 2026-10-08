"""
Claude Code Tool - Simplified wrapper for executing code tasks
Core functionality: Run Claude Code CLI with auto project folder management
"""

import json
import subprocess
from typing import Dict
from pathlib import Path
from datetime import datetime


def run_claude_code(
    prompt: str,
    timeout: int = 600
) -> Dict:
    """
    Run Claude Code in headless mode
    
    Args:
        prompt: Task prompt for Claude Code
        timeout: Timeout in seconds (default 600s = 10min)
    
    Returns:
        Dict with execution result
        
    Example:
        result = run_claude_code("Write a Python script to analyze data")
    """
    try:
        # Build command (always use text output, simpler)
        cmd = ["claude", "-p", prompt, "--output-format", "text"]
        
        # Execute Claude Code
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path.cwd()
        )
        
        if process.returncode != 0:
            return {
                "status": "error",
                "message": "Claude Code execution failed",
                "stderr": process.stderr,
                "stdout": process.stdout
            }
        
        return {
            "status": "success",
            "result": process.stdout,
            "message": "Task completed successfully"
        }
    
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": f"Claude Code timed out after {timeout}s"
        }
    
    except FileNotFoundError:
        return {
            "status": "error",
            "message": "Claude CLI not found. Install: https://claude.ai/download"
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to run Claude Code: {str(e)}"
        }


def run_project_task(
    task: str,
    project_name: str = None,
    timeout: int = 72000
) -> Dict:
    """
    Run a task with auto project folder creation in AIscientist/src/
    
    Args:
        task: Task description
        project_name: Project folder name (auto-generated if None)
        timeout: Timeout in seconds
    
    Returns:
        Dict with execution result and project path
        
    Example:
        result = run_project_task(
            task="Train a GNN model on molecular data",
            project_name="gnn_experiment"
        )
    """
    try:
        # Auto-generate project name if not provided
        if not project_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = f"project_{timestamp}"
        
        # Create project path
        base_dir = Path(__file__).parent.parent / "src"
        project_path = base_dir / project_name
        
        # Build enhanced prompt with pre-injected requirements
        enhanced_prompt = f"""
IMPORTANT: Create all files in the directory: {project_path.absolute()}

TASK: {task}

REQUIREMENTS:
1. Create a new project folder at: {project_path.absolute()}
2. Generate all code files in this folder
3. Include proper documentation and error handling
4. Save results (plots, metrics, logs) in the same folder
5. Create a README.md explaining what was done

Please complete this task and organize all files in the specified directory.
"""
        
        # Run Claude Code
        result = run_claude_code(prompt=enhanced_prompt, timeout=timeout)
        
        if result["status"] == "success":
            return {
                "status": "success",
                "task": task,
                "project_name": project_name,
                "project_path": str(project_path.absolute()),
                "result": result.get("result"),
                "message": f"Project created at: {project_path}"
            }
        else:
            return result
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to run project task: {str(e)}"
        }



if __name__ == "__main__":
    # Test the simplified tool
    print("Testing simplified Claude Code tool...")
    

    test_result = run_claude_code(
        prompt="Create a hello_world.py script that prints 'Hello World'"
    )
    print("\n=== Simple Task Test ===")
    print(json.dumps(test_result, indent=2))
