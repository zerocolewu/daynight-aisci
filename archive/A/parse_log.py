"""
Full.log Parser for AI Scientist Day-Night System v3

Extracts structured statistics from full.log:
- Token usage per agent
- LLM execution time per agent
- Tool calls statistics (counts only, not execution time)
- Summary by day and agent type

Note: Tool execution time is NOT captured due to log structure limitations.
See TOOL_TIME_LIMITATION.md for details.
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from collections import defaultdict


@dataclass
class ToolCall:
    """Single tool call record"""
    tool_id: str
    tool_name: str
    arguments: str


@dataclass
class Metrics:
    """Token and time metrics"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    llm_time_seconds: float = 0.0
    tokens_per_second: float = 0.0


@dataclass
class AgentRun:
    """Single Agent run record"""
    agent_type: str
    agent_id: str = ""
    session_id: str = ""
    run_id: str = ""
    day_number: int = 0
    metrics: Metrics = field(default_factory=Metrics)
    tool_calls: List[ToolCall] = field(default_factory=list)


@dataclass
class DayCycle:
    """One complete day cycle"""
    day_number: int
    start_time: str = ""
    end_time: str = ""
    orchestrator_runs: List[AgentRun] = field(default_factory=list)
    behaviour_runs: List[AgentRun] = field(default_factory=list)


class FullLogParser:
    """Parser for full.log files"""

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.content = ""
        self.lines = []
        self.day_cycles: List[DayCycle] = []

    def load(self):
        """Load log file"""
        with open(self.log_path, 'r', encoding='utf-8') as f:
            self.content = f.read()
        self.lines = self.content.split('\n')
        print(f"✓ Loaded {len(self.lines)} lines from {self.log_path}")

    def _normalize_agent_type(self, behaviour_name: str) -> str:
        """Normalize behaviour name to agent type"""
        name_lower = behaviour_name.lower().strip()
        mapping = {
            "learning search":  "learn_search",
            "learning read":  "learn_read",
            "learning (skim cards)":  "learn_read",
            "working":  "work",
            "internalizing":  "internalize",
            "associating (reasoning)":  "associate_reason",
            "associating (random)":  "associate_random",
        }
        for key, value in mapping.items():
            if key in name_lower:
                return value
        return name_lower.replace(" ", "_").replace("(", "").replace(")", "")

    def _save_agent(self, day: DayCycle, agent: AgentRun):
        """Save agent to appropriate slot"""
        if agent.agent_type == 'orchestrator':
            day.orchestrator_runs.append(agent)
        else:
            day.behaviour_runs.append(agent)

    def parse(self) -> List[DayCycle]:
        """Parse the log file and return structured data"""
        if not self.lines:
            self.load()

        self.day_cycles = []
        current_day: Optional[DayCycle] = None
        current_agent: Optional[AgentRun] = None
        orchestrator_agent: Optional[AgentRun] = None  # Track parent ORCHESTRATOR
        last_behaviour_type: Optional[str] = None  # Track last behaviour for multi-run agents (learn_read)
        in_metrics = False
        in_tool_calls = False
        pending_tool_call = {}
        in_experiment_tool_metrics = False  # Track run_experiment TOOL METRICS

        for line in self.lines:
            clean_line = re.sub(r'^DEBUG\s*', '', line).strip()
            # Remove ANSI color codes
            clean_line = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', "", clean_line)

            # ====== DAY STRUCTURE ======
            # Session start time
            match = re.match(r'Session started: (.+)', clean_line)
            if match and current_day and not current_day.start_time:
                current_day.start_time = match.group(1)
                continue

            # Day initialization
            match = re.search(r'🤖 Initializing AI Scientist System \(v2\) \(Day (\d+)\)', clean_line)
            if match:
                day_num = int(match.group(1))
                if not any(d.day_number == day_num for d in self.day_cycles):
                    current_day = DayCycle(day_number=day_num)
                    self.day_cycles.append(current_day)
                continue

            # Orchestrator start
            match = re.search(r'🌍 FULL CYCLE - ORCHESTRATOR \(v2\) \(Day (\d+)\)', clean_line)
            if match and current_day:
                if current_agent:
                    self._save_agent(current_day, current_agent)
                current_agent = AgentRun(
                    agent_type='orchestrator',
                    day_number=current_day.day_number
                )
                orchestrator_agent = current_agent  # Track as parent
                last_behaviour_type = None
                in_metrics = False
                in_tool_calls = False
                continue

            # Day/Night mode (behaviour agents) - ORCHESTRATOR is still running
            match = re.search(r'[🌅🌙] (DAY|NIGHT) MODE: (.+?)(?:\s*\(v[\d\w-]+\))?(?:\s*\(Day (\d+)\))?$', clean_line)
            if match and current_day:
                # Don't save ORCHESTRATOR - it's still running, will resume after behaviour ends
                behaviour_name = match.group(2).strip()
                agent_type = self._normalize_agent_type(behaviour_name)
                current_agent = AgentRun(
                    agent_type=agent_type,
                    day_number=current_day.day_number
                )
                last_behaviour_type = agent_type
                in_metrics = False
                in_tool_calls = False
                continue

            # Fallback: learn_read has no DAY MODE banner, detect via Running line
            match = re.search(r'Running: learn_read\(task=', clean_line)
            if match and current_day and current_agent and current_agent.agent_type == 'orchestrator':
                current_agent = AgentRun(
                    agent_type='learn_read',
                    day_number=current_day.day_number
                )
                last_behaviour_type = 'learn_read'
                in_metrics = False
                in_tool_calls = False
                continue

            # ====== RUN EXPERIMENT TIME CAPTURE ======
            # Track when run_experiment starts running
            if re.search(r'Running: run_experiment\(', clean_line):
                in_experiment_tool_metrics = True
                continue

            # ====== AGENT METADATA ======
            match = re.search(r'Agent ID: ([a-f0-9-]+)', clean_line)
            if match and current_agent:
                current_agent.agent_id = match.group(1)
                continue

            match = re.search(r'Session ID: ([a-f0-9-]+)', clean_line)
            if match and current_agent:
                current_agent.session_id = match.group(1)
                continue

            match = re.search(r'Agent Run Start: ([a-f0-9-]+)', clean_line)
            if match:
                if current_agent and current_agent.agent_type == 'orchestrator' and last_behaviour_type:
                    # Same behaviour continuing (e.g. learn_read next paper) - create new run
                    current_agent = AgentRun(
                        agent_type=last_behaviour_type,
                        day_number=current_day.day_number if current_day else 0
                    )
                if current_agent:
                    current_agent.run_id = match.group(1)
                continue

            # Run end - finalize current agent, handle next run
            match = re.search(r'Agent Run End: ([a-f0-9-]+)', clean_line)
            if match and current_agent and current_day:
                self._save_agent(current_day, current_agent)
                ended_type = current_agent.agent_type
                if ended_type == 'orchestrator':
                    current_agent = None
                    orchestrator_agent = None
                    last_behaviour_type = None
                else:
                    # Behaviour ended - restore ORCHESTRATOR but keep last_behaviour_type
                    # in case there are more runs of the same type (e.g. learn_read per paper)
                    if orchestrator_agent:
                        current_agent = orchestrator_agent
                    else:
                        current_agent = None
                        last_behaviour_type = None
                continue

            # ====== METRICS PARSING ======
            # TOOL METRICS should NOT be parsed as LLM metrics
            if re.search(r'\*+  TOOL METRICS  \*+', clean_line):
                in_metrics = False
                # If we're tracking a run_experiment, start capturing its tool time
                if in_experiment_tool_metrics and current_day:
                    in_experiment_tool_metrics = 'capturing'
                continue

            # Capture run_experiment time from TOOL METRICS
            if in_experiment_tool_metrics == 'capturing' and current_day:
                match = re.search(r'\* Time:\s+([\d.]+)s', clean_line)
                if match:
                    exp_run = AgentRun(
                        agent_type='run_experiment',
                        day_number=current_day.day_number
                    )
                    exp_run.metrics.llm_time_seconds = float(match.group(1))
                    current_day.behaviour_runs.append(exp_run)
                    in_experiment_tool_metrics = False
                    continue

            # LLM METRICS block start (token stats)
            if re.search(r'\*+  METRICS  \*+', clean_line):
                in_metrics = True
                continue

            # Parse LLM metrics (tokens, time)
            # IMPORTANT: ACCUMULATE metrics values (not replace)
            if in_metrics and current_agent:
                match = re.search(r'\* Tokens:\s+input=(\d+), output=(\d+), total=(\d+)(?:, cached=(\d+))?(?:, reasoning=(\d+))?', clean_line)
                if match:
                    current_agent.metrics.input_tokens += int(match.group(1))
                    current_agent.metrics.output_tokens += int(match.group(2))
                    current_agent.metrics.total_tokens += int(match.group(3))
                    if match.group(4):
                        current_agent.metrics.cached_tokens += int(match.group(4))
                    if match.group(5):
                        current_agent.metrics.reasoning_tokens += int(match.group(5))
                    continue

                match = re.search(r'\* Time:\s+([\d.]+)s', clean_line)
                if match:
                    current_agent.metrics.llm_time_seconds += float(match.group(1))
                    continue

                match = re.search(r'\* Tokens per second:\s+([\d.]+)', clean_line)
                if match:
                    # tokens_per_second is an average, take the last one
                    current_agent.metrics.tokens_per_second = float(match.group(1))
                    continue

            # ====== TOOL CALLS PARSING (counts only) ======
            if clean_line == 'Tool Calls:':
                in_tool_calls = True
                continue

            if in_tool_calls and current_agent:
                match = re.search(r"- ID: '([^']+)'", clean_line)
                if match:
                    pending_tool_call['tool_id'] = match.group(1)
                    continue

                match = re.search(r"Name: '([^']+)'", clean_line)
                if match:
                    pending_tool_call['tool_name'] = match.group(1)
                    continue

                match = re.search(r"Arguments: '(.+)'", clean_line)
                if match:
                    pending_tool_call['arguments'] = match.group(1)
                    if 'tool_id' in pending_tool_call and 'tool_name' in pending_tool_call:
                        current_agent.tool_calls.append(ToolCall(
                            tool_id=pending_tool_call['tool_id'],
                            tool_name=pending_tool_call['tool_name'],
                            arguments=pending_tool_call.get('arguments', '')
                        ))
                    pending_tool_call = {}
                    continue

                # End of tool calls section
                if clean_line.startswith('*') or clean_line.startswith('=') or clean_line.startswith('Running:'):
                    in_tool_calls = False

            # ====== DAY END ======
            match = re.search(r'Day (\d+) ended: (.+)', clean_line)
            if match and current_day:
                current_day.end_time = match.group(2)
                if current_agent:
                    self._save_agent(current_day, current_agent)
                current_day = None
                current_agent = None
                in_experiment_tool_metrics = False
                continue

        # Finalize last agent if needed
        if current_agent and current_day:
            self._save_agent(current_day, current_agent)

        return self.day_cycles

    def get_summary(self) -> Dict:
        """Get summary statistics"""
        summary = {
            'total_days': len(self.day_cycles),
            'by_day': {},
            'by_agent_type': defaultdict(lambda: {
                'count': 0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_tokens': 0,
                'total_llm_time': 0.0,
                'tool_calls': defaultdict(int)
            }),
            'total_tokens': 0,
            'total_llm_time': 0.0,
            'total_runs_by_type': defaultdict(int)
        }

        for day in self.day_cycles:
            day_summary = {
                'day_number': day.day_number,
                'start_time': day.start_time,
                'end_time': day.end_time,
                'orchestrator_runs': [],
                'behaviours': [],
                'day_total_tokens': 0,
                'day_llm_time': 0.0
            }

            # Process orchestrator runs
            for orch in day.orchestrator_runs:
                orch_summary = {
                    'agent_id': orch.agent_id,
                    'run_id': orch.run_id,
                    'input_tokens': orch.metrics.input_tokens,
                    'output_tokens': orch.metrics.output_tokens,
                    'total_tokens': orch.metrics.total_tokens,
                    'llm_time': orch.metrics.llm_time_seconds,
                    'tool_calls': [tc.tool_name for tc in orch.tool_calls]
                }
                day_summary['orchestrator_runs'].append(orch_summary)
                day_summary['day_total_tokens'] += orch.metrics.total_tokens
                day_summary['day_llm_time'] += orch.metrics.llm_time_seconds

                # For by_agent_type, aggregate ALL runs
                summary['total_runs_by_type']['orchestrator'] += 1
                agent_stats = summary['by_agent_type']['orchestrator']
                agent_stats['count'] += 1
                agent_stats['total_input_tokens'] += orch.metrics.input_tokens
                agent_stats['total_output_tokens'] += orch.metrics.output_tokens
                agent_stats['total_tokens'] += orch.metrics.total_tokens
                agent_stats['total_llm_time'] += orch.metrics.llm_time_seconds
                for tc in orch.tool_calls:
                    agent_stats['tool_calls'][tc.tool_name] += 1

            # Process behaviours
            for run in day.behaviour_runs:
                behaviour_summary = {
                    'agent_type': run.agent_type,
                    'agent_id': run.agent_id,
                    'run_id': run.run_id,
                    'input_tokens': run.metrics.input_tokens,
                    'output_tokens': run.metrics.output_tokens,
                    'total_tokens': run.metrics.total_tokens,
                    'llm_time': run.metrics.llm_time_seconds,
                    'tool_calls': [tc.tool_name for tc in run.tool_calls]
                }
                day_summary['behaviours'].append(behaviour_summary)
                day_summary['day_total_tokens'] += run.metrics.total_tokens
                day_summary['day_llm_time'] += run.metrics.llm_time_seconds

                agent_stats = summary['by_agent_type'][run.agent_type]
                agent_stats['count'] += 1
                agent_stats['total_input_tokens'] += run.metrics.input_tokens
                agent_stats['total_output_tokens'] += run.metrics.output_tokens
                agent_stats['total_tokens'] += run.metrics.total_tokens
                agent_stats['total_llm_time'] += run.metrics.llm_time_seconds
                for tc in run.tool_calls:
                    agent_stats['tool_calls'][tc.tool_name] += 1

            summary['by_day'][f'day_{day.day_number}'] = day_summary
            summary['total_tokens'] += day_summary['day_total_tokens']
            summary['total_llm_time'] += day_summary['day_llm_time']

        # Convert defaultdicts to regular dicts
        summary['by_agent_type'] = {k: dict(v) for k, v in summary['by_agent_type'].items()}
        for agent_type in summary['by_agent_type']:
            summary['by_agent_type'][agent_type]['tool_calls'] = dict(
                summary['by_agent_type'][agent_type]['tool_calls']
            )
        summary['total_runs_by_type'] = dict(summary['total_runs_by_type'])

        return summary

    def print_report(self):
        """Print a human-readable report"""
        summary = self.get_summary()
        print("\n" + "=" * 100)
        print("AI SCIENTIST LOG ANALYSIS REPORT")
        print("=" * 100)
        print(f"\n📊 OVERALL STATISTICS")
        print(f"   Total Days: {summary['total_days']}")
        print(f"   Total Tokens: {summary['total_tokens']:,}")
        print(f"   Total LLM Time: {summary['total_llm_time']:.1f}s ({summary['total_llm_time']/3600:.2f}h)")

        print(f"\n📈 BY AGENT TYPE (Aggregated across all runs)")
        print("-" * 100)
        header = f"{'Agent Type':<22} {'Runs':>6} {'Input':>10} {'Output':>8} {'Total':>10} {'LLM(s)':>10}"
        print(header)
        print("-" * 100)
        for agent_type, stats in sorted(summary['by_agent_type'].items()):
            runs = summary['total_runs_by_type'].get(agent_type, stats['count'])
            print(f"{agent_type:<22} {runs:>6} {stats['total_input_tokens']:>10,} {stats['total_output_tokens']:>8,} {stats['total_tokens']:>10,} {stats['total_llm_time']:>10.1f}")

        print(f"\n🔧 TOOL CALLS BY AGENT TYPE")
        print("-" * 100)
        for agent_type, stats in sorted(summary['by_agent_type'].items()):
            if stats['tool_calls']:
                print(f"\n{agent_type}:")
                for tool_name, count in sorted(stats['tool_calls'].items(), key=lambda x: -x[1]):
                    print(f"   - {tool_name}: {count}")

        print(f"\n📅 DAILY BREAKDOWN")
        print("-" * 100)
        for day_key in sorted(summary['by_day'].keys()):
            day_data = summary['by_day'][day_key]
            print(f"\n{day_key.upper()}:")
            print(f"   Tokens: {day_data['day_total_tokens']:,} | LLM Time: {day_data['day_llm_time']:.1f}s")

            if day_data['orchestrator_runs']:
                print(f"   ORCHESTRATOR RUNS ({len(day_data['orchestrator_runs'])} runs):")
                for i, orch in enumerate(day_data['orchestrator_runs']):
                    prefix = "   └─" if i == len(day_data['orchestrator_runs']) - 1 and not day_data['behaviours'] else "   ├─"
                    print(f"   {prefix} run_{i+1}: {orch['total_tokens']:,} tokens, {orch['llm_time']:.1f}s")

            if day_data['behaviours']:
                print(f"   BEHAVIOUR RUNS ({len(day_data['behaviours'])} runs):")
                for j, bhv in enumerate(day_data['behaviours']):
                    prefix = "   └─" if j == len(day_data['behaviours']) - 1 else "   ├─"
                    print(f"   {prefix} {bhv['agent_type']}: {bhv['total_tokens']:,} tokens, {bhv['llm_time']:.1f}s")

        print("\n" + "=" * 100)
        return summary

    def save_json(self, output_path=None):
        """Save summary to JSON file"""
        if output_path is None:
            output_path = self.log_path.parent / "log_analysis.json"
        summary = self.get_summary()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n✓ JSON summary saved to: {output_path}")


if __name__ == "__main__":
    import sys
    log_path = Path(__file__).parent / "full.log"
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        sys.exit(1)
    parser = FullLogParser(str(log_path))
    parser.parse()
    parser.print_report()
    parser.save_json()
