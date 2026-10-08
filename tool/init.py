"""
Initialization script: Import 129 papers from lamnew-markdown directory into mem.json
Use Agno Agent to extract and structure literature information
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


class LiteratureIngestionAgent:
    """Literature Ingestion Agent: Read, parse, structure papers and save to memory"""
    
    def __init__(self):
        # Initialize Agent
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
        
        self.mem_file = "mem.json"
        self.literature_dir = "lamnew-markdown"
    
    def extract_title_from_folder(self, folder_path: Path) -> str:
        """
        Extract paper title from folder name
        Folder format: "Paper Title.pdf-uuid"
        """
        folder_name = folder_path.name
        # Remove .pdf-uuid suffix
        title = re.sub(r'\.pdf-[a-f0-9\-]+$', '', folder_name)
        return title
    
    def structure_literature_with_llm(self, title: str, full_content: str) -> dict:
        """
        Use LLM to extract structured information from literature
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
- Stay faithful to the original text and accurately extract key informatio, don not miss any important information in the paper.
- Maintain logical coherence with natural transitions between ideas
- Deeply explain core mechanisms and theoretical foundations
- Use fluent academic language, avoiding simple enumeration
- DO NOT use real line breaks in JSON string values - use \\n\\n to separate paragraphs and \\n for line breaks
"""
        
        try:
            response = self.agent.run(prompt)
            result_text = response.content
            
            # Clean possible markdown code block markers
            result_text = result_text.strip()
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.startswith('```'):
                result_text = result_text[3:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            
            # Parse JSON
            structured_data = json.loads(result_text)
            structured_data["title"] = title  # Add title from folder name
            return structured_data
            
        except Exception as e:
            print(f"  ⚠️  LLM extraction failed: {e}")
            print(f"  Error details: {str(e)[:200]}")
            # Return basic information
            return {
                "title": title,
                "background": "N/A",
                "method": "N/A",
                "experiment": "N/A",
                "result": "N/A",
                "keywords": "machine_learning polymer molecular_property",
                "abstract": title[:30] if title else "N/A"
            }
    
    def load_memories(self) -> list:
        """Load existing memories"""
        if not os.path.exists(self.mem_file):
            return []
        
        with open(self.mem_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('memory', [])
    
    def save_memories(self, memories: list):
        """Save memories to mem.json"""
        data = {'memory': memories}
        with open(self.mem_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def add_memory(self, content: dict, local_path: str) -> str:
        """
        Add a literature memory entry
        """
        memories = self.load_memories()
        
        # Generate unique ID
        mem_id = f"mem_{len(memories)+1:03d}"
        
        # Build memory object
        memory = {
            "id": mem_id,
            "type": "knowledge",
            "abstract": content.get("abstract", ""),
            "keyword": content.get("keywords", ""),
            "content": {
                "background": content.get("background", ""),
                "method": content.get("method", ""),
                "experiment": content.get("experiment", ""),
                "result": content.get("result", ""),
                "local_path": local_path,
                "year_title": content.get("title", "")
            },
            "connections": {},
            "metadata": {
                "created": datetime.now().isoformat(),
                "last_changed": datetime.now().isoformat(),
                "source": "lamnew-markdown",
                "embedding": None  # Can add embedding later
            }
        }
        
        memories.append(memory)
        self.save_memories(memories)
        
        return mem_id
    
    def process_all_literatures(self):
        """
        Process all literatures and import into mem.json
        """
        # Find all full.md files
        literature_path = Path(self.literature_dir)
        md_files = list(literature_path.glob("*/full.md"))
        
        print(f"\n{'='*60}")
        print(f"📚 Literature Import Task")
        print(f"{'='*60}")
        print(f"Found {len(md_files)} papers")
        print(f"Target: mem.json")
        print(f"{'='*60}\n")
        
        success_count = 0
        failed_files = []
        
        for md_file in tqdm(md_files, desc="Processing papers"):
            try:
                # Read file
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract title from folder name
                title = self.extract_title_from_folder(md_file.parent)
                
                # Use LLM to extract structured information
                structured_data = self.structure_literature_with_llm(
                    title,
                    content
                )
                
                # Save to mem.json with full path
                mem_id = self.add_memory(
                    structured_data,
                    str(md_file)
                )
                
                success_count += 1
                
            except Exception as e:
                failed_files.append((str(md_file), str(e)))
                print(f"\n❌ Processing failed: {md_file.name}")
                print(f"   Error: {e}")
        
        # Output statistics
        print(f"\n{'='*60}")
        print(f"✅ Import complete!")
        print(f"{'='*60}")
        print(f"Success: {success_count}/{len(md_files)}")
        print(f"Failed: {len(failed_files)}")
        
        if failed_files:
            print(f"\nFailed files:")
            for file, error in failed_files[:10]:  # Show first 10
                print(f"  - {Path(file).name}: {error}")
        
        print(f"\n💾 Memories saved to: {self.mem_file}")
        print(f"{'='*60}\n")


def main():
    """Main function"""
    print("\n🚀 Initialize AI Scientist System - Literature Import Module")
    
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment")
        print("Please configure in .env file: OPENAI_API_KEY=your_key")
        return
    
    # Check literature directory
    if not os.path.exists("lamnew-markdown"):
        print("❌ Error: lamnew-markdown directory not found")
        return
    
    # Create agent and process
    agent = LiteratureIngestionAgent()
    
    agent.process_all_literatures()


if __name__ == "__main__":
    main()

