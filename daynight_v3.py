"""
AI Scientist Day-Night System (v2)

Key differences from daynight.py:
- Do NOT modify daynight.py; this is a copied & evolved controller.
- Assumes mem.json has been seeded by cold_start_agent:
  - mem_001: TopicState
  - mem_002: MetaMethod
- Learning pipeline: learn_search (pool+arXiv selection+convert) -> learn_read (structured reading + topic link)
- `read_deepdetail(mem_id)` exists as a tool for reading verbatim paper markdown via knowledge memory's local_path (only when exact details are needed).
- Work is gated by ExperimentCard (must be written before run_experiment) and writes a separate post-run Experience.
- Internalize runs two agents sequentially: internalize_integrate + internalize_meta (updates mem_001/mem_002).
"""

from __future__ import annotations

from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.tools.thinking import ThinkingTools

import os
import sys
import re
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any, Union

from dotenv import load_dotenv

# ==================== LOGGING SETUP ====================


class TeeOutput:
    """Write to both console and file simultaneously"""

    def __init__(self, log_file: str):
        self.terminal = sys.stdout
        self.log = open(log_file, "a", encoding="utf-8")
        # Write session header
        self.log.write(f"\n{'='*60}\n")
        self.log.write(f"Session started: {datetime.now().isoformat()}\n")
        self.log.write(f"{'='*60}\n\n")
        self.log.flush()

    def write(self, message):
        self.terminal.write(message)
        # Strip ANSI escape codes before writing to file
        clean_message = re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", message)
        self.log.write(clean_message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# Redirect stdout and stderr to full.log
LOG_FILE = Path(__file__).parent / "full.log"
sys.stdout = TeeOutput(str(LOG_FILE))
sys.stderr = sys.stdout  # Also capture stderr (DEBUG output)


# Write concise summaries to lite.log (separate from stdout tee)
LITE_LOG_FILE = Path(__file__).parent / "lite.log"


def _append_lite_log(text: str) -> None:
    try:
        with open(LITE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
    except Exception:
        # Don't fail the run for lite logging issues
        pass


def _safe_to_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        try:
            return json.dumps(x, ensure_ascii=False, indent=2)
        except Exception:
            return str(x)
    return str(x)


# ==================== TOOLS ====================

from tool.memory_search import search_memory
from tool.memory_read import read_memory
from tool.memory_write import write_memory
from tool.memory_update import update_memory_content

from tool.claude_code_tool import run_project_task
from tool.arxiv_tool import search_arxiv, download_arxiv_paper
from tool.mineru import convert_pdfs_to_markdown


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


TOPIC_STATE_ID = "mem_001"
META_METHOD_ID = "mem_002"

# ==================== read_deepdetail (tool) ====================

def read_deepdetail(mem_id: str, max_chars: int = 20000) -> Dict[str, Any]:
    """
    Read verbatim paper markdown for a specific memory id.
    Only use this when exact textual details from a paper are needed.

    Policy:
    - Only supported for knowledge memories whose content is a dict containing "local_path".
    - Returns truncated markdown to keep context manageable.
    """
    only_msg = "read_deepdetail only for specific knowledge memory"
    try:
        r = read_memory(mem_id, detail="content")
        if not (isinstance(r, dict) and r.get("status") == "success"):
            return {"status": "success", "message": only_msg, "mem_id": mem_id, "raw": r}
        mems = r.get("memories", [])
        if not mems or not isinstance(mems[0], dict):
            return {"status": "success", "message": only_msg, "mem_id": mem_id}

        mem = mems[0]
        if mem.get("type") != "knowledge":
            return {
                "status": "success",
                "message": only_msg,
                "mem_id": mem_id,
                "type": mem.get("type"),
            }

        c = mem.get("content", {})
        local_path = c.get("local_path") if isinstance(c, dict) else None
        if not local_path:
            return {"status": "success", "message": only_msg, "mem_id": mem_id}

        p = Path(str(local_path))
        if not p.exists():
            return {"status": "success", "message": only_msg, "mem_id": mem_id, "local_path": str(local_path)}

        text = p.read_text(encoding="utf-8", errors="ignore")
        if isinstance(max_chars, int) and max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars] + "\n...(truncated)..."

        return {"status": "success", "mem_id": mem_id, "local_path": str(local_path), "paper_markdown": text}
    except Exception as e:
        return {"status": "success", "message": only_msg, "mem_id": mem_id, "error": str(e)}


# ==================== AIScientist ====================


class AIScientist:
    """
    AI Scientist System with Day-Night operational cycles (v2)
    """

    def __init__(self, day: int = 1):
        self.api_key = openai_api_key
        self.day = day

        # Create directories
        self.papers_dir = Path(f"papers/day_{self.day}")
        self.papers_dir.mkdir(parents=True, exist_ok=True)

        self.src_dir = Path("src")
        self.src_dir.mkdir(parents=True, exist_ok=True)

        self.model_config = {
            "id": "glm-5.2",
            "api_key": self.api_key,
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        }

        self.description = """
Act as an objective scientific reasoning system specialized in deep learning for science.
Prioritize scrutiny-resistant evidence; do not hallucinate.
Be faithful to the facts, and never beautify results.
Always thoroughly reasoning and fully understand the meaning of prompts before you act.
"""

        # Lazy agents
        self._learn_agent_search = None
        self._learn_agent_read = None
        self._learn_agent_compile = None
        self._work_agent = None
        self._best_of_n_agent = None
        self._internalize_integrate_agent = None
        self._internalize_meta_agent = None
        self._associate_agent_reason = None
        self._associate_agent_random = None
        self._daily_summary_agent = None
        self._current_paper_local_path: Optional[str] = None

    # ==================== REAL-TIME MONITORING (memory type counts) ====================

    def _memory_type_counts_line(self) -> str:
        """
        Compute a one-line summary of mem.json type counts (excluding resource).
        Format requirement (fixed first line in agent instructions):
          memory: knowledge N, idea N, experience N, integration N
        """
        counts = {"knowledge": 0, "idea": 0, "experience": 0, "integration": 0}
        try:
            with open("mem.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            mems = data.get("memory", [])
            if isinstance(mems, list):
                for m in mems:
                    if not isinstance(m, dict):
                        continue
                    t = m.get("type")
                    if t in counts:
                        counts[t] += 1
        except Exception:
            # Never block agent execution due to monitoring.
            pass
        return (
            f"memory: knowledge {counts['knowledge']}, idea {counts['idea']}, "
            f"experience {counts['experience']}, integration {counts['integration']}"
        )

    def _inject_memory_counts_into_instructions(self, agent: Any) -> None:
        """
        Insert the 'memory: ...' line as the FIRST LINE of agent instructions.
        If instructions already start with a memory line, replace it to keep it real-time.
        """
        try:
            instr = getattr(agent, "instructions", "") or ""
            # Remove existing prefix if present to avoid stacking
            instr = re.sub(r"^memory:.*\n+", "", instr, flags=re.IGNORECASE)
            agent.instructions = self._memory_type_counts_line() + "\n" + instr.lstrip()
        except Exception:
            # Monitoring must not break a run.
            return

    def _append_daily_journal(self, lite_trace: str) -> Dict[str, Any]:
        """
        Summarize a day's trace and append a rolling 3-entry journal into TopicState (mem_001).
        Journal schema: list[{"day": int, "journal": str}]
        """
        try:
            if self._daily_summary_agent is None:
                self._daily_summary_agent = Agent(
                    model=OpenAILike(**self.model_config),
                    description="You are a daily journal agent that preserves continuity across days.",
                    instructions=f"""
You read today's lite trace and produce ONE concise daily journal entry.

First read:
- read_memory("{TOPIC_STATE_ID}", detail="content")
- read_memory("{META_METHOD_ID}", detail="content")

Output MUST be plain text ONLY (no JSON), faithful to the trace (no fabrication).
Keep it **very brief and concise**, only capture the most important information (what was done and what is the result), **not exceed 100 words**, definitely do not fabricate.

If a tool call fails, it may be due to tool-call formatting; try adjusting the format. If it still fails after multiple attempts, skip using that tool.
""",
                    tools=[read_memory],
                    show_tool_calls=True,
                    debug_mode=True,
                    markdown=False,
                )

            max_chars = 80000
            trace_for_agent = lite_trace[-max_chars:] if len(lite_trace) > max_chars else lite_trace
            self._inject_memory_counts_into_instructions(self._daily_summary_agent)
            resp = self._daily_summary_agent.run(
                f"Day {self.day} lite trace (possibly truncated to last {max_chars} chars):\n\n{trace_for_agent}"
            )
            journal_text = (resp.content or "").strip()
            if not journal_text:
                journal_text = "(empty journal)"

            ts_read = read_memory(TOPIC_STATE_ID, detail="content")
            ts_content = {}
            if isinstance(ts_read, dict) and ts_read.get("status") == "success":
                mems = ts_read.get("memories", [])
                if mems and isinstance(mems[0], dict):
                    ts_content = mems[0].get("content", {}) or {}

            journal = ts_content.get("daily_journal", [])
            if not isinstance(journal, list):
                journal = []
            journal.append({"day": self.day, "journal": journal_text})
            while len(journal) > 3:
                journal.pop(0)

            patch = {"daily_journal": journal}

            upd = update_memory_content(TOPIC_STATE_ID, patch)
            return {"status": "success", "topicstate_update": upd, "journal_entry": {"day": self.day, "journal": journal_text}}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ==================== DAY: LEARN SEARCH ====================

    def learn_search(self, task: str) -> str:
        """
        LEARN SEARCH v2: select papers (pool+arXiv) and convert PDFs to markdown in papers/day_{day}/
        """

        pool_dir = Path(__file__).parent / "papers" / "pool"

        def list_pool_papers() -> List[str]:
            if not pool_dir.exists():
                return []
            return sorted([p.name for p in pool_dir.glob("*.pdf")])

        def copy_pool_papers_to_day(filenames: List[str]) -> Dict[str, Any]:
            copied = []
            missing = []
            for name in filenames:
                src = pool_dir / name
                if not src.exists():
                    missing.append(name)
                    continue
                dst = self.papers_dir / name
                # Default policy: once selected, remove from pool (move into day folder)
                shutil.move(str(src), str(dst))
                copied.append(str(dst))
            return {"status": "success", "copied": copied, "missing": missing, "day_dir": str(self.papers_dir)}

        def download_to_day_dir(arxiv_id: str, filename: Optional[str] = None):
            return download_arxiv_paper(arxiv_id=arxiv_id, download_dir=str(self.papers_dir), filename=filename)

        def convert_papers_to_markdown():
            return convert_pdfs_to_markdown(str(self.papers_dir))

        if self._learn_agent_search is None:
            self._learn_agent_search = Agent(
                model=OpenAILike(**self.model_config),
                description=self.description,
                instructions=f"""
You are now focused on literature selection (not summarization).

First read TopicState and MetaMethod which are your metacognition and you should align with them:
- read_memory("{TOPIC_STATE_ID}", detail="content")
- read_memory("{META_METHOD_ID}", detail="content")


You have two sources:
1) Local pool: list_pool_papers() then choose 1–2 PDFs to move into {self.papers_dir} (selected PDFs will be removed from pool)
2) arXiv: use search_arxiv (sort_by=relevance) and download_to_day_dir for 1–2 high-quality papers, not superficial review articles.

Hard rules:
- You must justify each selected paper against TopicState (research directions and active hypotheses).
- Selection discipline (quotas, quality gating) is defined in MetaMethod; follow it strictly.
- After placing PDFs into {self.papers_dir}, you MUST call convert_papers_to_markdown() to produce markdown for reading.

Task focus: {task}

If a tool call fails, it may be due to tool-call formatting; try adjusting the format. If it still fails after multiple attempts, skip using that tool (Include arxiv network request timeout).
""",
                tools=[
                    read_memory,
                    list_pool_papers,
                    copy_pool_papers_to_day,
                    search_arxiv,
                    download_to_day_dir,
                    convert_papers_to_markdown,
                    search_memory
                ],
                show_tool_calls=True,
                debug_mode=True,
                markdown=False,
            )

        print(f"\n{'='*60}")
        print("🌅 DAY MODE: LEARNING SEARCH (v2)")
        print(f"{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}\n")
        self._inject_memory_counts_into_instructions(self._learn_agent_search)
        response = self._learn_agent_search.run(task)
        return response.content

    # ==================== DAY: LEARN READ (SKIM CARDS) ====================

    def learn_read(self, task: str = "") -> str:
        """
        LEARN READ v2: structured paper reading (daynight.py-style) + TopicState/MetaMethod link fields
        """
        
        print(f"\n{'='*60}")
        print("🌅 DAY MODE: LEARNING READ (v2)")
        print(f"{'='*60}\n")

        md_files = list(self.papers_dir.rglob("*.md"))
        if not md_files:
            return f"No markdown files found in {self.papers_dir}. Run learn_search first."

        def write_memory_with_day(memory_type, abstract, keyword, content):
            # Hard guardrail: inject local_path from the current paper context (do not rely on LLM)
            if isinstance(content, dict):
                if self._current_paper_local_path:
                    content["local_path"] = self._current_paper_local_path
            return write_memory(memory_type, abstract, keyword, content, day=self.day)

        results = []
        for md_file in md_files:
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                if len(content) > 100000:
                    content = content[:100000] + "\n...(truncated)..."

                paper_title = md_file.stem
                self._current_paper_local_path = str(md_file)
                if self._learn_agent_read is None:
                    self._learn_agent_read = Agent(
                        model=OpenAILike(**self.model_config),
                        description=self.description,
                        instructions=f"""
You are now focused on reading papers and committing them to memory.

First read TopicState and MetaMethod which are your metacognition and you should align with them:
- read_memory("{TOPIC_STATE_ID}", detail="content")
- read_memory("{META_METHOD_ID}", detail="content")

Stay faithful to the original text and do not add content not present in the paper.

Current Paper Title: {paper_title}
Paper Content is provided in the user prompt.

You MUST save ONE memory per paper via write_memory_with_day with:
"type": "knowledge"
"abstract": a one-sentence summary of the core contribution (<=100 words)
"keyword": 3-5 keywords 
"content": a dict with EXACT keys:

"background": "Articulate the current state of the field, core limitations of existing methods, and the motivation for this research. Explain why this problem matters and how the new approach addresses existing bottlenecks. (Approximately 300-500 words)",
"method": "Systematically describe the core idea, specific technical design, and theoretical foundation. Explain why this design was chosen, why it works, and the synergistic mechanisms among key components, etc. (Approximately 500-1500 words)",
"experiment": "Detail the specific experimental setup: dataset selection, experimental configuration, baseline comparisons, main experiments, ablation studies, etc. (Approximately 500-1500 words)",
"result": "Present main experimental findings and explain the principles behind performance improvements. Discuss the method's advantages, limitations, and future directions, etc. (Approximately 300-500 words)",
"github_url": "if the paper provides a repo link, extract it; otherwise N/A".
"dataset": "what dataset(s) are used; if unclear, N/A".
"topic_relation": "how this paper relates to current TopicState (which hypothesis/evidence need it touches)."
"topic_contribution": "what actionable contribution it provides to TopicState (what decision/experiment it enables)."

If a tool call fails, it may be due to tool-call formatting; try adjusting the format. If it still fails after multiple attempts, skip using that tool.
""",
                        tools=[read_memory, write_memory_with_day],
                        show_tool_calls=True,
                        debug_mode=True,
                        markdown=False,
                    )

                self._inject_memory_counts_into_instructions(self._learn_agent_read)
                resp = self._learn_agent_read.run(f"Paper title: {paper_title}\n\nPaper content:\n\n{content}")
                results.append(f"Skimmed {paper_title}: {resp.content}")
            except Exception as e:
                results.append(f"Failed to skim {md_file}: {e}")
            finally:
                self._current_paper_local_path = None

        return "\n".join(results)

    # ==================== DAY: WORK ====================

    def work(self, task: str = "") -> str:
        """
        WORK v2: requires ExperimentCard gating and writes separate Experience/Result post-run.
        """

        def write_memory_with_day(memory_type, abstract, keyword, content):
            return write_memory(memory_type, abstract, keyword, content, day=self.day)

        def run_experiment(task_description: str, project_name: str = None):
            if project_name:
                full_project_name = f"{self.day}_{project_name}"
            else:
                full_project_name = f"{self.day}_experiment"

            return run_project_task(
                task=task_description,
                project_name=full_project_name,
                timeout=108000,
            )

        def run_experiment_continue(task_description: str, project_name: str = None):
            """
            Same as run_experiment, but resumes the most recent Claude Code session under the project folder.
            Use this when the previous run silently stopped (e.g., network problem) and you need to continue.
            """
            if project_name:
                full_project_name = f"{self.day}_{project_name}"
            else:
                full_project_name = f"{self.day}_experiment"

            return run_project_task(
                task=task_description,
                project_name=full_project_name,
                timeout=108000,
                continue_session=True,
            )

        def best_of_n_select(query: str = "Select the best next idea for work") -> str:
            """
            Tool wrapper: pick one idea/hypothesis to execute.
            Candidate ideas should come from associate outputs and from TopicState trajectory/directions.
            """
            def _extract_mem_id(text: str) -> Optional[str]:
                if not isinstance(text, str):
                    return None
                m = re.search(r"\bmem_\d+\b", text)
                return m.group(0) if m else None

            if self._best_of_n_agent is None:
                self._best_of_n_agent = Agent(
                    model=OpenAILike(**self.model_config),
                    description="You are a strict PI assistant that selects ONE executable idea.",
                    instructions=f"""
You select ONE best next executable idea for work, based on TopicState and current memory.

First read TopicState and MetaMethod which are your metacognition and you should align with them:
- read_memory("{TOPIC_STATE_ID}", detail="content")
- read_memory("{META_METHOD_ID}", detail="content")

Then retrieve candidate ideas:
- Use search_memory(..., memory_type="idea", return_full_content=False) to find recent idea memories relevant to TopicState.
- If you have multiple candidates, you MUST read the top candidates with read_memory(..., detail="content", memory_type="idea") before deciding.
- If ideas are missing, you MUST return selected_idea_id="N/A" and explain that the system should run association_reason to write new idea memory first. Do NOT invent an un-stored idea.

Output MUST be plain text ONLY (no JSON, no markdown code fences), exactly ONE line:
mem_### | <short reason>
If no idea exists, output:
N/A | <short reason>

If a tool call fails, it may be due to tool-call formatting; try adjusting the format. If it still fails after multiple attempts, skip using that tool.
""",
                    tools=[read_memory, search_memory],
                    show_tool_calls=True,
                    debug_mode=True,
                    markdown=False,
                )
            self._inject_memory_counts_into_instructions(self._best_of_n_agent)
            resp = self._best_of_n_agent.run(query)
            out = (resp.content or "").strip()
            if not out:
                return "N/A | empty output from best-of-n"
            # If the model violated the one-line format, salvage a mem_id but still return plain text.
            mid = _extract_mem_id(out)
            if mid:
                reason = out.replace(mid, "").strip()
                if reason.startswith("|"):
                    reason = reason[1:].strip()
                if not reason:
                    reason = "(no reason provided)"
                return f"{mid} | {reason[:300]}"
            return f"N/A | {out[:300]}"

        if self._work_agent is None:
            self._work_agent = Agent(
                model=OpenAILike(**self.model_config),
                description=self.description,
                instructions=f"""
You are now acting as a **Research Supervisor**. Fully execute one best idea, including the complete deep learning computational experiment workflow such as code generation and training, and save the experience to memory.

=== WORKFLOW ===
First read TopicState and MetaMethod which are your metacognition and you should align with them:
read_memory("{TOPIC_STATE_ID}", detail="content")
read_memory("{META_METHOD_ID}", detail="content")

Call best_of_n_select() to select one idea that relevant to task to execute.
best_of_n_select returns idea mem id and reason.
After you get a mem_id, you need call read_memory(mem_id, detail="content", memory_type="idea") to fetch the full idea content before planning the experiment.

Fully understand this idea, transform it into an executable experimental plan through "mental simulation" like a human researcher, with focus on the **model and experimental strategy** mentioned below:

Your experiment is executed through the run_experiment tool (this is actually Claude Code, with full code editing and execution capabilities). Provide it with a complete task description, including all the information it needs.

Honestly analyze the returned experimental results:
- If the experiment validates/invalidates the theory → Write experience to memory and stop
- If results are inconclusive or improvable → Iterate with refined instructions (using the SAME project folder)

=== TASK SPECIFICATION FORMAT ===
When calling run_experiment, your task_description MUST include:

**Idea**:
- What theoretical idea is being tested
- What hypothesis needs to be validated/invalidated

**Model and Experimental Strategy**:
- Dataset: Tg/Tm/Egc/PI1M, already downloaded locally to src/alldataset/
- Experiment working directory: src/{{day}}_{{project_name}}/
- Model: Core architecture requirements. This is the most critical part in deep learning; the model architecture must be clearly constructed strictly according to the idea
- Experimental strategy: Specifically how to train and evaluate this deep learning model, what processing needs to be done on the data and results, and a series of other very specific detailed issues
- Diagnostic analysis: design diagnostic part of the experiment to explain *why* the model is better or worse, not just whether it is.

**Constraints**:
- Libraries: Only use PyTorch, PyTorch Geometric, etc. (there are many other libraries available in environment)
- Data: Use ONLY the specified dataset, do NOT substitute with others

**Expected Output**:
- What files should be created
- What results should be reported

=== ITERATION RULES ===
- If the experiment is not successful(**including bad performance**), deeply analyze the reason and you need to iterate with run_experiment, do not iterate more than 2 times.
- When iterating, focus on the SAME idea and use the SAME project_name to continue in the existing folder
- Be specific about what to FIX or IMPROVE based on previous results

=== HONESTY PRINCIPLE ===
- If an experiment fails, report it honestly as failure
- If results are negative (theory not supported), that is still valuable
- Do NOT fabricate positive results to make the "story" better
- A negative result with clear analysis is better than a fake positive result

If a tool call fails, it may be due to tool-call formatting; try adjusting the format. If it still fails after multiple attempts, skip using that tool.
If encounter context limit, use the same project_name and run_experiment. Do not use run_experiment_continue, which still leads to context limit
If encounter network problem, use the same project_name and run_experiment_continue.

The long training period is perfectly acceptable; do not voluntarily stop the experiment.
""",
                tools=[
                    read_memory,
                    best_of_n_select,
                    write_memory_with_day,
                    run_experiment,
                    run_experiment_continue,
                ],
                show_tool_calls=True,
                debug_mode=True,
                markdown=False,
            )

        print(f"\n{'='*60}")
        print("🌅 DAY MODE: WORKING (v2)")
        print(f"{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}\n")

        self._inject_memory_counts_into_instructions(self._work_agent)
        response = self._work_agent.run(task)
        return response.content

    # ==================== NIGHT: INTERNALIZE (INTEGRATE + META) ====================

    def internalize(self, task: str = "Internalize (integrate + meta)") -> str:
        """
        Internalize v3: integrate/restructure memories (no explicit connection graph building)
        """

        # Collect today's memories ids/abstracts
        try:
            with open("mem.json", "r", encoding="utf-8") as f:
                mem_data = json.load(f)
            today_memories = [
                {"id": m.get("id"), "abstract": m.get("abstract")}
                for m in mem_data.get("memory", [])
                if m.get("metadata", {}).get("day") == self.day
            ]
        except Exception:
            today_memories = []

        memories_context = json.dumps(today_memories[:50], ensure_ascii=False, indent=2)

        # Narrow tool wrapper: keep it minimal (similar to write_memory_with_day)
        def write_integration_with_day(abstract: str, keyword: str, content: Any):
            # 用{"text": "..."} 包裹了一层，使得如果传入的是字符串，也可以变成dict，符合write_memory中的content的数据类型要求
            payload = content if isinstance(content, dict) else {"text": str(content)}
            return write_memory("integration", abstract, keyword, payload, day=self.day)

        # internalize_integrate agent
        if self._internalize_integrate_agent is None:
            self._internalize_integrate_agent = Agent(
                model=OpenAILike(**self.model_config),
                description=self.description,
                instructions=f"""
You are the internalize_integrate_agent.

First, read TopicState and MetaMethod:
- read_memory("{TOPIC_STATE_ID}", detail="content")
- read_memory("{META_METHOD_ID}", detail="content")

You are internalizing memories through integration. Your goal is to compress and reorganize today’s new atomic memories together with a set of relevant past memories into higher-level integration memories—like forming a coherent memory cluster in a human brain.

Today’s memories (id + abstract):
{memories_context}

Your tasks:
1) Read the most relevant memories from today (read_memory).
2) Retrieve relevant past memories when needed (search_memory), guided by today’s memories and the active hypotheses in TopicState.
3) For groups of memories with deep, strong relevance (not superficial similarity), write **1–3** new memories of type="integration" using write_integration_with_day. **Do not write more than 3**.
4) The integration memory’s content should be a dict. It should include consensus / conflicts / alignment / mentioned_memory_ids. Use deep narrative statements and explicitly cite the referenced memories using parentheses (mem_###). Turn messy atomic memories into a clear, structured understanding, and let key insights emerge from this reorganization.

After writing the integration memories, output a short plain-text integration report including:
- A summary of what was integrated
- Suggestions for updating the active hypotheses in TopicState (do not directly edit TopicState; only propose)

If a tool call fails, it may be due to tool-call formatting; try adjusting the format. If it still fails after multiple attempts, skip using that tool.
""",
                tools=[read_memory, search_memory, write_integration_with_day],
                show_tool_calls=True,
                debug_mode=True,
                markdown=False,
            )

        # internalize_meta agent: update mem_001 and mem_002
        if self._internalize_meta_agent is None:
            self._internalize_meta_agent = Agent(
                model=OpenAILike(**self.model_config),
                description=self.description,
                instructions=f"""
You are internalize_meta_agent.

You update:
- TopicState (mem_001): research direction + active hypotheses
Do not update daily_journal, connections, and other content in TopicState that is not belong to research direction and active hypotheses

First read TopicState and MetaMethod which are your metacognition and you should align with them:
- read_memory("{TOPIC_STATE_ID}", detail="content")
- read_memory("{META_METHOD_ID}", detail="content")

Then read relevant today memories and integrate-agent report (provided in user prompt).
Don't update TopicState too much, not all memories are really useful, keep the research continuity.
Don't change the theme of active hypotheses, only update the details.

Rules:
- Update mem_001 via update_memory_content (do NOT create new IDs).

Output STRICT JSON with keys:
- topic_state_updates (dict)

If a tool call fails, it may be due to tool-call formatting; try adjusting the format. If it still fails after multiple attempts, skip using that tool.
""",
                tools=[read_memory, update_memory_content],
                show_tool_calls=True,
                debug_mode=True,
                markdown=False,
            )

        print(f"\n{'='*60}")
        print(f"🌙 NIGHT MODE: INTERNALIZING (v3-integrate) (Day {self.day})")
        print(f"{'='*60}\n")

        self._inject_memory_counts_into_instructions(self._internalize_integrate_agent)
        integrate_report = self._internalize_integrate_agent.run(task).content
        self._inject_memory_counts_into_instructions(self._internalize_meta_agent)
        meta_resp = self._internalize_meta_agent.run(
            f"Internalize task: {task}\n\ninternalize_integrate_report:\n{integrate_report}\n"
        )

        # Apply updates if JSON is valid
        try:
            upd = json.loads(meta_resp.content)
            ts_upd = upd.get("topic_state_updates", {})
            mm_upd = upd.get("meta_method_updates", {})
            if isinstance(ts_upd, dict) and ts_upd:
                # Guard: never let internalize_meta overwrite daily_journal or connections.
                # The agent frequently violates its instruction by including these fields,
                # which corrupts the rolling journal list (converts it to a plain string).
                ts_upd.pop("daily_journal", None)
                ts_upd.pop("connections", None)
                if ts_upd:
                    update_memory_content(TOPIC_STATE_ID, ts_upd)
            if isinstance(mm_upd, dict) and mm_upd:
                update_memory_content(META_METHOD_ID, mm_upd)
        except Exception:
            pass

        return f"internalize_integrate:\n{integrate_report}\n\ninternalize_meta:\n{meta_resp.content}"

    # ==================== NIGHT: ASSOCIATE ====================

    def associate_reason(self, task: str) -> str:
        def write_memory_with_day(memory_type, abstract, keyword, content):
            return write_memory(memory_type, abstract, keyword, content, day=self.day)

        if self._associate_agent_reason is None:
            self._associate_agent_reason = Agent(
                model=OpenAILike(**self.model_config),
                description=self.description,
                instructions=f"""
You are generating executable ideas (Idea = operational wrapper of hypotheses). **No more than 2 ideas**.

First read TopicState and MetaMethod which are your metacognition and you should align with them:
read_memory("{TOPIC_STATE_ID}", detail="content")
read_memory("{META_METHOD_ID}", detail="content")
Your tools are read_memory, search_memory, write_memory_with_day:

read_memory(memory_ids, detail="content", memory_type=None)
Purpose: Read memories by ID
search_memory(query, top_k=10, return_full_content=False, memory_type=None)
Purpose: Semantic retrieval of memories; can use memory_type="knowledge" to search only for ideas

When proposing ideas, you should maintain consistency with topic state and metamethod, and you should conduct "mental simulation" like a human scientist, thinking through many devilish details, so that your ideas are truly built on thinking rather than simple patchwork.
Idea (content must be a dict type, not a string), containing the following required keys:
H: Related hypotheses (can be multiple)
X: Novel experimental ideas on top of H (cannot be replication of existing work, should be research increment; linked to H)
Y: List of expected observable phenomena/results on top of X, and what kind of results will have what kind of association with H

Save ideas via write_memory_with_day, type "idea", containing:
abstract: 1 sentence naming the idea
keyword: 3-5 words
content: the dict above

If a tool call fails, it may be due to tool-call formatting; try adjusting the format. If it still fails after multiple attempts, skip using that tool.
""",
                tools=[read_memory, search_memory, write_memory_with_day],
                show_tool_calls=True,
                debug_mode=True,
                markdown=False,
            )

        print(f"\n{'='*60}")
        print("🌙 NIGHT MODE: ASSOCIATING (Reasoning) (v2)")
        print(f"{'='*60}\n")
        self._inject_memory_counts_into_instructions(self._associate_agent_reason)
        return self._associate_agent_reason.run(task).content

    def associate_random(self, task: str) -> str:
        def write_memory_with_day(memory_type, abstract, keyword, content):
            return write_memory(memory_type, abstract, keyword, content, day=self.day)

        if self._associate_agent_random is None:
            self._associate_agent_random = Agent(
                model=OpenAILike(**self.model_config),
                description=self.description,
                instructions=f"""
You are generating executable ideas (Idea = operational wrapper of hypotheses). **only 1 idea**.

First read TopicState and MetaMethod which are your metacognition and you should align with them:
read_memory("{TOPIC_STATE_ID}", detail="content")
read_memory("{META_METHOD_ID}", detail="content")
Your tools are read_memory, search_memory, write_memory_with_day:

read_memory(memory_ids, detail="content", memory_type=None)
Purpose: Read memories by ID
search_memory(query, top_k=10, return_full_content=False, memory_type=None)
Purpose: Semantic retrieval of memories; can use memory_type="knowledge" to search only for ideas

When proposing ideas, you should maintain consistency with topic state and metamethod, and you should conduct "mental simulation" like a human scientist, thinking through many devilish details, so that your ideas are truly built on thinking rather than simple patchwork.
The ideas you generate should lean more towards conceptual leaping, seeking new problem perspectives from shared underlying principles, finding ideas that make people ask "why has no one ever thought of this before". 
If the information you need does not exist in memory, please use your own world knowledge.

Idea (content must be a dict type, not a string), containing the following required keys:
H: Related hypotheses (can be multiple)
X: Novel experimental ideas on top of H (cannot be replication of existing work, should be research increment; linked to H)
Y: List of expected observable phenomena/results on top of X, and what kind of results will have what kind of association with H

Save ideas via write_memory_with_day, type "idea", containing:
abstract: 1 sentence naming the idea (what intervention + what phenomenon)
keyword: 3-5 words
content: the dict above

If a tool call fails, it may be due to tool-call formatting; try adjusting the format. If it still fails after multiple attempts, skip using that tool.
""",
                tools=[read_memory, search_memory, write_memory_with_day],
                show_tool_calls=True,
                debug_mode=True,
                markdown=False,
            )

        print(f"\n{'='*60}")
        print("🌙 NIGHT MODE: ASSOCIATING (Random) (v2)")
        print(f"{'='*60}\n")
        self._inject_memory_counts_into_instructions(self._associate_agent_random)
        return self._associate_agent_random.run(task).content

    # ==================== CYCLE CONTROL ====================

    def full_cycle(self, instruction: str = "Execute an adaptive research cycle.") -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print(f"🌍 FULL CYCLE - ORCHESTRATOR (v2) (Day {self.day})")
        print("=" * 60 + "\n")

        orchestrator = Agent(
            model=OpenAILike(**self.model_config),
            description="You are the Principal Investigator (PI) and Chief Architect of this research project.",
            instructions=f"""
You are the PI of a long-term research initiative focusing on Deep Learning for Science.
Today is Day {self.day} of a consecutive multi-day run. Treat this as a continuation: inherit context from TopicState and maintain the long-term consistency.
On Day 1, prioritize “Data Centric Foundation” hypotheses; on subsequent days, explore other promising hypotheses.

Innovation requires logic; too much complexity isn't necessarily good, nor is too little simplicity. The best algorithm is perhaps a delicate balance.

First, always read:
- TopicState: read_memory("{TOPIC_STATE_ID}", detail="content")
- MetaMethod: read_memory("{META_METHOD_ID}", detail="content")

You have these behaviors (tools):
- learn_search(task): selection+conversion from pool+arXiv into papers/day_{self.day}/
- learn_read(task): structured reading (background/method/experiment/result/github_url/dataset) + topic_relation/topic_contribution, saved to memory
- associate_reason(task): generate reasonable executable ideas
- associate_random(task): generate Concept leaping executable ideas
- work(task): Fully execute one best idea (it will find it on its own), including the complete deep learning computational experiment workflow such as code generation and training, and save the experience to memory.
- internalize(task): internalize_integrate + internalize_meta (update TopicState)

Core Research Workflow (must maintain consistency of topic state and metamethod):
IN EACH DAY:
Run learn_search -> learn_read -> work -> internalize -> associate_reason/random. IN EACH DAY, Each behavior runs only once Except work which can be run 2-3 times.

User instruction: {instruction}

If a tool call fails, it may be due to tool-call formatting; try adjusting the format. If it still fails after multiple attempts, skip using that tool.
""",
            tools=[
                read_memory,
                self.learn_search,
                self.learn_read,
                self.associate_reason,
                self.associate_random,
                self.work,
                self.internalize,
            ],
            show_tool_calls=True,
            debug_mode=True,
            markdown=False,
        )

        self._inject_memory_counts_into_instructions(orchestrator)
        day_start_time = datetime.now().isoformat()
        response = orchestrator.run(instruction)
        response_light = (
            f"\n{'='*60}\n"
            f"Day {self.day} | start: {day_start_time} | end: {datetime.now().isoformat()}\n"
            f"{'='*60}\n"
        )
        for message in response.messages:
            role = getattr(message, "role", "")
            if role == "system":
                continue
            content = _safe_to_text(getattr(message, "content", ""))
            response_light += f"角色: \n{role}\n内容: \n{content}\n\n"

        _append_lite_log(response_light)
        # Continuity bridge: summarize day and write rolling journal into TopicState (mem_001)
        try:
            jr = self._append_daily_journal(response_light)
            # Important: do not silently swallow failures; print to full.log via stdout tee.
            if isinstance(jr, dict) and jr.get("status") == "success":
                print(f"[daily_journal] success: { _safe_to_text(jr) }")
            else:
                print(f"[daily_journal] error: { _safe_to_text(jr) }")
        except Exception as e:
            print(f"[daily_journal] exception: {e}")

        print(f"\n{'='*60}")
        print(f"Day {self.day} ended: {datetime.now().isoformat()}")
        print(f"{'='*60}\n")

        return {"mode": "full_cycle_v3", "result": response.content, "messages": response.messages}


def initialize_system(day: int = 1) -> AIScientist:
    print("\n" + "=" * 60)
    print(f"🤖 Initializing AI Scientist System (v2) (Day {day})")
    print("=" * 60)
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment")
    scientist = AIScientist(day=day)
    print("✓ System initialized successfully")
    print(f"  Papers directory: {scientist.papers_dir}")
    print(f"  Src directory: {scientist.src_dir}")
    print("=" * 60 + "\n")
    return scientist


if __name__ == "__main__":
    # Run consecutive days in a single execution (no CLI args by design).
    for day in range(1, 8):
        scientist = initialize_system(day=day)
        scientist.full_cycle(
            "Run an adaptive day"
        )
        mem_src = Path(__file__).parent / "mem.json"
        if mem_src.exists():
            shutil.copy2(str(mem_src), str(mem_src.parent / f"mem_day{day}.json"))