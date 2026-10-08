"""
重新生成包含N/A的记忆记录
找出所有background、method、experiment、result中包含N/A的记录，然后重新用LLM生成
"""

import os
import json
from pathlib import Path
from datetime import datetime
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from dotenv import load_dotenv
from tqdm import tqdm

# 加载环境变量
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


class MemoryRegenerator:
    """重新生成N/A记忆的工具类"""
    
    def __init__(self, mem_file="mem.json"):
        # 初始化Agent
        self.agent = Agent(
            model=OpenAILike(
                id="qwen3-max",
                api_key=openai_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                temperature=0.6  
            ),
            instructions="""You are a professional scientific researcher""",
            show_tool_calls=False,
            debug_mode=False,
        )
        
        self.mem_file = mem_file
    
    def load_memories(self) -> dict:
        """加载记忆文件"""
        with open(self.mem_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_memories(self, data: dict):
        """保存记忆文件"""
        with open(self.mem_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def has_na(self, memory: dict) -> bool:
        """检查记忆是否包含N/A"""
        content = memory.get('content', {})
        fields = ['background', 'method', 'experiment', 'result']
        
        for field in fields:
            value = content.get(field, '')
            if value == 'N/A':
                return True
        
        return False
    
    def structure_literature_with_llm(self, title: str, full_content: str) -> dict:
        """
        使用LLM从文献中提取结构化信息
        """
        prompt = f"""Please act as a senior AI researcher and retell the original paper into the following four sections. Stay faithful to the original text and do not add content not present in the paper.

Title: {title}

Paper content:
{full_content}

You must strictly return your answer in the following JSON format, and do not include any content outside the JSON (without markdown code block markers):
{{
  "background": "Articulate the current state of the field, core limitations of existing methods, and the motivation for this research. Explain why this problem matters and how the new approach addresses existing bottlenecks. (Approximately 300-800 words)",
  
  "method": "Systematically describe the core idea, specific technical design, and theoretical foundation. Explain why this design, why it works, and the synergistic mechanisms among key components. (Approximately 500-1500 words)",
  
  "experiment": "Detail the specific experimental setup: dataset selection, experimental configuration, baseline comparisons, main experiments, and ablation studies. (Approximately 300-1500 words)",
  
  "result": "Present main experimental findings and explain the principles behind performance improvements. Discuss the method's advantages, limitations, and future directions. (Approximately 300-1000 words)",
  
  "keywords": "Extract 5 most important technical keywords from the paper (space-separated, e.g., 'machine_learning graph_neural_network molecular_property')",
  
  "abstract": "Generate a one-sentence summary of the paper's core contribution (within 100 words)"
}}

Requirements:
- Stay faithful to the original text and accurately extract key information, do not miss any important information in the paper.
- Maintain logical coherence with natural transitions between ideas
- Deeply explain core mechanisms and theoretical foundations
- Use fluent academic language, avoiding simple enumeration
- DO NOT use real line breaks in JSON string values - use \\n\\n to separate paragraphs and \\n for line breaks
"""
        
        try:
            response = self.agent.run(prompt)
            result_text = response.content
            
            # 清理可能的markdown代码块标记
            result_text = result_text.strip()
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.startswith('```'):
                result_text = result_text[3:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            
            # 解析JSON
            structured_data = json.loads(result_text)
            return structured_data
            
        except Exception as e:
            print(f"  ⚠️  LLM提取失败: {e}")
            return None
    
    def regenerate_na_memories(self):
        """重新生成所有包含N/A的记忆"""
        print(f"\n{'='*60}")
        print(f"🔄 重新生成N/A记忆任务")
        print(f"{'='*60}\n")
        
        # 加载记忆
        data = self.load_memories()
        memories = data.get('memory', [])
        
        # 找出所有包含N/A的记忆
        na_memories = []
        for i, memory in enumerate(memories):
            if self.has_na(memory):
                na_memories.append((i, memory))
        
        print(f"总记忆数: {len(memories)}")
        print(f"包含N/A的记忆数: {len(na_memories)}")
        print(f"{'='*60}\n")
        
        if len(na_memories) == 0:
            print("✅ 没有需要重新生成的记忆!")
            return
        
        # 询问用户是否继续
        print(f"找到 {len(na_memories)} 个包含N/A的记忆需要重新生成。")
        choice = input("是否继续? (y/n): ")
        if choice.lower() != 'y':
            print("取消操作。")
            return
        
        success_count = 0
        failed_count = 0
        
        # 重新生成每个N/A记忆
        for idx, (mem_idx, memory) in enumerate(tqdm(na_memories, desc="重新生成记忆")):
            try:
                # 获取本地文件路径
                local_path = memory['content'].get('local_path', '')
                if not local_path:
                    print(f"\n⚠️  记忆 {memory['id']} 没有local_path，跳过")
                    failed_count += 1
                    continue
                
                # 检查文件是否存在
                if not os.path.exists(local_path):
                    print(f"\n⚠️  文件不存在: {local_path}")
                    failed_count += 1
                    continue
                
                # 读取文件内容
                with open(local_path, 'r', encoding='utf-8') as f:
                    full_content = f.read()
                
                # 获取标题
                title = memory['content'].get('year_title', '')
                
                # 使用LLM重新生成
                print(f"\n处理 [{idx+1}/{len(na_memories)}]: {title[:50]}...")
                structured_data = self.structure_literature_with_llm(title, full_content)
                
                if structured_data:
                    # 更新记忆
                    memories[mem_idx]['content']['background'] = structured_data.get('background', 'N/A')
                    memories[mem_idx]['content']['method'] = structured_data.get('method', 'N/A')
                    memories[mem_idx]['content']['experiment'] = structured_data.get('experiment', 'N/A')
                    memories[mem_idx]['content']['result'] = structured_data.get('result', 'N/A')
                    memories[mem_idx]['keyword'] = structured_data.get('keywords', memory['keyword'])
                    memories[mem_idx]['abstract'] = structured_data.get('abstract', memory['abstract'])
                    memories[mem_idx]['metadata']['last_changed'] = datetime.now().isoformat()
                    
                    success_count += 1
                    print(f"  ✅ 成功重新生成")
                else:
                    failed_count += 1
                    print(f"  ❌ 生成失败")
                
            except Exception as e:
                print(f"\n❌ 处理失败 {memory.get('id', 'unknown')}: {e}")
                failed_count += 1
        
        # 保存更新后的记忆
        data['memory'] = memories
        self.save_memories(data)
        
        # 输出统计
        print(f"\n{'='*60}")
        print(f"✅ 重新生成完成!")
        print(f"{'='*60}")
        print(f"成功: {success_count}/{len(na_memories)}")
        print(f"失败: {failed_count}/{len(na_memories)}")
        print(f"\n💾 更新已保存到: {self.mem_file}")
        print(f"{'='*60}\n")


def main():
    """主函数"""
    print("\n🚀 AI Scientist - 重新生成N/A记忆模块")
    
    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 错误: 找不到OPENAI_API_KEY环境变量")
        print("请在.env文件中配置: OPENAI_API_KEY=your_key")
        return
    
    # 检查mem.json文件
    if not os.path.exists("mem.json"):
        print("❌ 错误: 找不到mem.json文件")
        return
    
    # 创建重新生成器并处理
    regenerator = MemoryRegenerator()
    regenerator.regenerate_na_memories()


if __name__ == "__main__":
    main()

