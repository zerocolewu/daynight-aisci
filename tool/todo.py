"""
简化版 TODO 工具 - 可直接用于 Agent 框架
适用于 LangChain, CrewAI 等框架的 tool 参数
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


# ============= 全局状态管理 =============

class TodoState:
    """全局 TODO 状态"""
    def __init__(self):
        self.storage_path = Path(".agent_todos.json")
        self.todos: List[Dict[str, Any]] = []
        self._load()
    
    def _load(self):
        """从磁盘加载"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self.todos = json.load(f)
            except:
                self.todos = []
    
    def _save(self):
        """保存到磁盘"""
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.todos, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存失败: {e}")
    
    def get_stats(self):
        """获取统计信息"""
        return {
            "total": len(self.todos),
            "completed": sum(1 for t in self.todos if t.get("status") == "completed"),
            "in_progress": sum(1 for t in self.todos if t.get("status") == "in_progress"),
            "pending": sum(1 for t in self.todos if t.get("status") == "pending")
        }


# 创建全局单例
_todo_state = TodoState()


# ============= 工具函数 =============

def todo_read() -> str:
    """
    读取当前的 TODO 任务列表
    
    Returns:
        JSON 字符串，包含所有任务和统计信息
    """
    stats = _todo_state.get_stats()
    
    result = {
        "todos": _todo_state.todos,
        "statistics": stats
    }
    
    return json.dumps(result, ensure_ascii=False, indent=2)


def todo_write(todos: List[Dict[str, Any]]) -> str:
    """
    更新完整的 TODO 任务列表
    
    Args:
        todos: 任务列表，每个任务包含：
            - content (必需): 任务描述
            - status (必需): "pending" | "in_progress" | "completed"
            - priority (可选): "high" | "medium" | "low"，默认 "medium"
            - id (可选): 任务ID，不提供则自动生成
    
    Returns:
        JSON 字符串，包含操作结果和重要的提醒信息
    
    Example:
        todo_write([
            {"content": "分析需求", "status": "completed", "priority": "high"},
            {"content": "编写代码", "status": "in_progress", "priority": "high"},
            {"content": "写测试", "status": "pending", "priority": "medium"}
        ])
    """
    # 验证和补全数据
    for todo in todos:
        # 必需字段检查
        if not todo.get("content"):
            return json.dumps({
                "success": False,
                "error": "每个任务必须包含 content 字段"
            }, ensure_ascii=False)
        
        if not todo.get("status"):
            return json.dumps({
                "success": False,
                "error": "每个任务必须包含 status 字段"
            }, ensure_ascii=False)
        
        # 状态验证
        if todo["status"] not in ["pending", "in_progress", "completed"]:
            return json.dumps({
                "success": False,
                "error": f"无效的状态: {todo['status']}"
            }, ensure_ascii=False)
        
        # 补全可选字段
        if not todo.get("id"):
            todo["id"] = str(uuid.uuid4())
        
        if not todo.get("priority"):
            todo["priority"] = "medium"
        elif todo["priority"] not in ["high", "medium", "low"]:
            return json.dumps({
                "success": False,
                "error": f"无效的优先级: {todo['priority']}"
            }, ensure_ascii=False)
        
        # 添加时间戳
        now = datetime.now().isoformat()
        if "created_at" not in todo:
            todo["created_at"] = now
        todo["updated_at"] = now
    
    # 确保只有一个 in_progress
    in_progress_count = sum(1 for t in todos if t["status"] == "in_progress")
    if in_progress_count > 1:
        return json.dumps({
            "success": False,
            "error": "只能有一个任务处于 in_progress 状态"
        }, ensure_ascii=False)
    
    # 保存
    _todo_state.todos = todos
    _todo_state._save()
    
    # 生成提醒
    reminder = _generate_reminder(todos)
    
    result = {
        "success": True,
        "count": len(todos),
        "statistics": _todo_state.get_stats(),
        "reminder": reminder
    }
    
    return json.dumps(result, ensure_ascii=False, indent=2)


def _generate_reminder(todos: List[Dict[str, Any]]) -> str:
    """生成系统提醒 - 这是让 agent 持续工作的关键"""
    
    pending = [t for t in todos if t["status"] == "pending"]
    in_progress = [t for t in todos if t["status"] == "in_progress"]
    completed = [t for t in todos if t["status"] == "completed"]
    total = len(todos)
    
    lines = [
        "\n" + "="*60,
        "📋 TODO 状态更新",
        "="*60,
        f"总计: {total} | 完成: {len(completed)} | 进行中: {len(in_progress)} | 待处理: {len(pending)}",
        ""
    ]
    
    # 关键：明确告诉 agent 下一步做什么
    if in_progress:
        task = in_progress[0]
        lines.extend([
            f"🔧 当前任务: {task['content']}",
            "",
            "⚠️ CRITICAL: 继续完成当前任务！",
            "完成后，立即将其状态改为 'completed'，",
            "然后开始下一个 'pending' 任务。",
            "不要停止工作！"
        ])
    elif pending:
        task = pending[0]
        lines.extend([
            f"⏳ 下一个任务: {task['content']}",
            "",
            "⚠️ CRITICAL: 立即开始这个任务！",
            "将其状态改为 'in_progress' 并开始工作。",
            "不要等待，不要犹豫！"
        ])
    elif total > 0 and len(completed) == total:
        lines.extend([
            "✅ 所有任务已完成！",
            "",
            "现在你可以：",
            "1. 向用户提供工作总结",
            "2. 询问是否需要优化或补充"
        ])
    else:
        lines.extend([
            "⚠️ 没有待处理的任务",
            "",
            "如果工作尚未完成，请创建新的 TODO 项。"
        ])
    
    lines.extend([
        "",
        "💡 提示: 不要向用户提及这个 TODO 更新，",
        "   除非用户明确询问进度。",
        "="*60
    ])
    
    return "\n".join(lines)


# ============= 工具描述（供 Agent 框架使用）=============

# 如果你的框架需要工具描述，使用这些
TODO_READ_DESCRIPTION = """读取当前的 TODO 任务列表。

使用场景：
- 查看所有任务的状态
- 了解当前进度
- 检查还有哪些任务需要完成

返回：包含所有任务和统计信息的 JSON 字符串"""


TODO_WRITE_DESCRIPTION = """更新完整的 TODO 任务列表。

⚠️ CRITICAL: 这是最重要的工具！必须频繁使用！

使用时机：
1. 收到多步骤任务时 - 立即创建 TODO 列表
2. 完成一个任务后 - 更新状态为 'completed'
3. 开始新任务前 - 将其状态改为 'in_progress'

重要规则：
- 每次更新必须提供完整列表（不是增量更新）
- 只能有一个任务处于 'in_progress' 状态
- 完成任务后立即标记，不要批量标记
- 必须遵守工具返回的 reminder 中的指示

参数：
- todos: 完整的任务列表
  - content (必需): 任务描述
  - status (必需): "pending" | "in_progress" | "completed"
  - priority (可选): "high" | "medium" | "low"

返回：操作结果和关键提醒信息"""


# ============= 使用示例 =============

if __name__ == "__main__":
    print("="*60)
    print("简化版 TODO 工具演示")
    print("="*60)
    
    # 示例 1: 创建任务列表
    print("\n1. 创建任务列表")
    print("-"*60)
    result = todo_write([
        {
            "content": "分析需求文档",
            "status": "completed",
            "priority": "high"
        },
        {
            "content": "设计系统架构",
            "status": "in_progress",
            "priority": "high"
        },
        {
            "content": "实现核心功能",
            "status": "pending",
            "priority": "high"
        },
        {
            "content": "编写测试用例",
            "status": "pending",
            "priority": "medium"
        }
    ])
    
    print(result)
    
    # 示例 2: 读取任务
    print("\n\n2. 读取当前任务")
    print("-"*60)
    result = todo_read()
    print(result)
    
    # 示例 3: 更新任务状态
    print("\n\n3. 完成当前任务，开始下一个")
    print("-"*60)
    result = todo_write([
        {
            "content": "分析需求文档",
            "status": "completed",
            "priority": "high"
        },
        {
            "content": "设计系统架构",
            "status": "completed",  # 完成了
            "priority": "high"
        },
        {
            "content": "实现核心功能",
            "status": "in_progress",  # 开始这个
            "priority": "high"
        },
        {
            "content": "编写测试用例",
            "status": "pending",
            "priority": "medium"
        }
    ])
    
    print(result)