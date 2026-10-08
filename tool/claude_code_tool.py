"""
Claude Code Tool - Simplified wrapper for executing code tasks
Core functionality: Run Claude Code CLI with auto project folder management
"""

import json
import subprocess
from typing import Dict, Optional, Union
from pathlib import Path
from datetime import datetime


def _is_context_limit_text(text: str) -> bool:
    t = (text or "").lower()
    return "context window limit" in t or "context length" in t or "maximum context" in t


def run_claude_code(
    prompt: str,
    timeout: int = 72000,
    cwd: Optional[Union[str, Path]] = None,
    continue_session: bool = False,
) -> Dict:
    """
    Run Claude Code in headless mode
    
    Args:
        prompt: Task prompt for Claude Code
        timeout: Timeout in seconds (default 72000s = 20h)
        cwd: Working directory for Claude Code (recommended: the project folder)
        continue_session: If True, run with --continue to resume the most recent session under cwd
    
    Returns:
        Dict with execution result
        
    Example:
        result = run_claude_code("Write a Python script to analyze data")
    """
    try:
        # Build command (use -p for non-interactive/print mode so subprocess can capture output reliably)
        cmd = ["claude", "-p"]
        if continue_session:
            cmd.append("--continue")
        cmd.append(prompt)
        
        # Execute Claude Code
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else str(Path.cwd()),
        )
        
        stdout = process.stdout or ""
        stderr = process.stderr or ""
        # Some Claude CLI/API errors appear in stdout; keep both for inspection.
        is_ctx = _is_context_limit_text(stdout) or _is_context_limit_text(stderr)

        # Treat empty stdout as failure too (helps catch "silent stop" cases)
        if process.returncode != 0 or not stdout.strip() or is_ctx:
            return {
                "status": "error",
                "error_type": "context_limit" if is_ctx else "execution_failed",
                "message": (
                    "Claude Code reached its context window limit. "
                    "Try continue_session=True with a SHORT follow-up prompt, or start a new session."
                    if is_ctx
                    else "Claude Code execution failed"
                ),
                "stderr": stderr,
                "stdout": stdout,
                "returncode": process.returncode,
                "cmd": cmd,
                "cwd": str(cwd) if cwd else str(Path.cwd()),
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
    timeout: int = 72000,
    continue_session: bool = False,
) -> Dict:
    """
    Run a task with auto project folder creation in AIscientist/src/
    """
    try:
        # Auto-generate project name if not provided
        if not project_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = f"project_{timestamp}"
        
        # Create project path
        base_dir = Path(__file__).parent.parent / "src"
        project_path = base_dir / project_name
        project_path.mkdir(parents=True, exist_ok=True)
        dataset_root = base_dir / "alldataset"
        
        # Build enhanced prompt with clear boundaries
        # IMPORTANT: when continuing a session, keep the prompt short to avoid re-triggering context limits.
        enhanced_prompt = (
            f"""
=== EXECUTION CONTEXT ===
You are executing a computational experiment for a research project.
Your working directory: {project_path.absolute()}
Your available datasets: {dataset_root.absolute()} (read datainfo.md if present; use ONLY these datasets)

=== CRITICAL CONSTRAINTS ===
MUST:
- Create all files ONLY in: {project_path.absolute()}
- Follow the task description precisely
- Report results honestly based on actual execution outcomes
- Use only libraries available in the current Python environment
- Follow the content in CLAUDE.md

MUST NOT:
- Reference or copy code from directories outside AIscientist/
- Let the parent folder name or unrelated files influence your implementation
- Fabricate datasets, including generating molecular data randomly and other dishonest experiments behaviors
- Fabricate or beautify experimental results
- Use datasets other than those specified in the task or outside {dataset_root.absolute()}
- Create new project folders if this is an iteration (work in the same directory)
- Simplify the experiment code

MAY:
- Make reasonable technical decisions for implementation details
- Add visualization and logging code
- Supplement evaluation metrics beyond the required ones

=== TASK ===
{task}

=== OUTPUT REQUIREMENTS ===
After completing the task, provide a COMPLETE and HONEST report including:
1. **Implementation Summary**: What was actually implemented
2. **Execution Status**: Success/Failure/Partial
3. **Actual Results**: Real metrics, not fabricated numbers
4. **Issues Encountered**: Any errors, warnings, or unexpected behaviors
5. **File Structure**: List of files created/modified
6. **Next Steps Suggestion**: What could be improved (if applicable)

Be thorough in your report - the research supervisor needs complete information to make decisions.
"""
            if not continue_session
            else f"""
Continue the most recent Claude Code session for this project.
Working directory: {project_path.absolute()}
Datasets root: {dataset_root.absolute()}

Instruction (delta):
{task}

Output requirement: provide an honest update of current status, what you completed before, what remains, and then continue executing.
"""
        )
        
        # Run Claude Code
        result = run_claude_code(
            prompt=enhanced_prompt,
            timeout=timeout,
            cwd=project_path,
            continue_session=continue_session,
        )
        
        if result["status"] == "success":
            return {
                "status": "success",
                "task": task,
                "project_name": project_name,
                "project_path": str(project_path.absolute()),
                "result": result.get("result"),
                "message": f"Task executed in: {project_path}"
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
