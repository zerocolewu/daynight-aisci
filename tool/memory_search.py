"""
记忆检索模块 - 基于FAISS的向量检索工具

作为Agent工具使用，提供语义检索功能

可以加入重排器，但现在记忆量较少，还不需要
"""

import json
import numpy as np
import faiss
from pathlib import Path
from openai import OpenAI
from typing import List, Dict, Optional, Literal, Union
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class MemorySearchTool:
    """记忆检索工具类"""
    
    def __init__(
        self,
        json_path: str = 'mem.json',
        embeddings_dir: str = 'embeddings',
        index_path: Optional[str] = None
    ):
        """
        初始化检索工具
        
        参数:
            json_path: mem.json文件路径
            embeddings_dir: embedding文件目录
            index_path: FAISS索引保存路径
        """
        self.json_path = json_path
        self.embeddings_dir = embeddings_dir
        self.index_path = index_path or f"{embeddings_dir}/faiss.index"
        self.mapping_path = self.index_path.replace('.index', '_mapping.npy')
        
        self.memories = None
        self.index = None
        self.valid_indices = None
        self.dimension = 1024

        # Lightweight change tracking for mem.json (append-only is assumed in this project)
        self._mem_json_mtime: Optional[float] = None
        self._last_seen_mem_count: int = 0
        # Since mem.json is append-only, we can scan only the tail on incremental updates
        self._scan_cursor: int = 0
        
        # 加载JSON
        self._load_json()
    
    def _load_json(self):
        """加载JSON数据"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.memories = data['memory']
        try:
            self._mem_json_mtime = Path(self.json_path).stat().st_mtime
        except Exception:
            self._mem_json_mtime = None
        self._last_seen_mem_count = len(self.memories) if isinstance(self.memories, list) else 0
        # If we haven't scanned anything yet, set cursor to current end.
        if self._scan_cursor <= 0:
            self._scan_cursor = self._last_seen_mem_count
        print(f"✓ 加载了 {len(self.memories)} 条记忆")

    def _refresh_json_if_changed(self) -> bool:
        """
        Reload mem.json if it changed on disk since last load.
        Returns True if reloaded, else False.
        """
        try:
            current_mtime = Path(self.json_path).stat().st_mtime
        except Exception:
            return False

        if self._mem_json_mtime is None or current_mtime > self._mem_json_mtime:
            # Reload from disk
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.memories = data.get("memory", [])
            self._mem_json_mtime = current_mtime
            # If mem.json ever shrinks/reorders (not expected here), fall back to safe full scan later.
            new_count = len(self.memories) if isinstance(self.memories, list) else 0
            if new_count < self._last_seen_mem_count:
                self._scan_cursor = 0
            self._last_seen_mem_count = new_count
            return True

        return False

    def _get_metadata(self, item: Dict) -> Optional[Dict]:
        """
        Return metadata dict if present; otherwise None.
        Policy: memories without metadata are silently skipped (e.g., mem_001/mem_002).
        """
        md = item.get("metadata")
        return md if isinstance(md, dict) else None

    def _persist_memories(self) -> None:
        """Persist current memories back to mem.json (best-effort)."""
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump({"memory": self.memories}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _text_for_embedding(self, item: Dict) -> str:
        """
        Build a compact text for embedding.
        Keep it stable and cheap: primarily abstract; fall back to keyword/content if needed.
        """
        abstract = (item.get("abstract") or "").strip()
        if abstract and abstract != "N/A" and len(abstract) >= 10:
            return abstract
        keyword = (item.get("keyword") or "").strip()
        if keyword:
            return f"{item.get('id','')} {keyword}".strip()
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()[:2000]
        if isinstance(content, (dict, list)):
            try:
                return json.dumps(content, ensure_ascii=False)[:2000]
            except Exception:
                return str(content)[:2000]
        return ""

    def _embed_text(self, text: str) -> np.ndarray:
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        response = client.embeddings.create(
            model="text-embedding-v4",
            input=text,
            dimensions=self.dimension,
            encoding_format="float"
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def _ensure_embedding(self, item: Dict) -> Optional[str]:
        """
        Ensure this memory has a usable embedding file path in metadata.embedding.
        If missing/None, generate embedding and save to embeddings/<id>_embedding.npy.
        Returns path string if available, else None.
        """
        md = self._get_metadata(item)
        if md is None:
            return None

        embedding_path = md.get("embedding")
        if embedding_path and embedding_path not in ("null", "None"):
            return str(embedding_path)

        mem_id = str(item.get("id") or "").strip()
        if not mem_id:
            return None

        text = self._text_for_embedding(item)
        if not text:
            md["embedding"] = None
            return None

        try:
            embeddings_path = Path(self.embeddings_dir)
            embeddings_path.mkdir(exist_ok=True)
            npy_filename = f"{mem_id}_embedding.npy"
            npy_path = embeddings_path / npy_filename

            emb = self._embed_text(text)
            np.save(npy_path, emb)

            # Store relative path for portability (matches embed_memories.py convention)
            md["embedding"] = f"{self.embeddings_dir}/{npy_filename}"
            return str(md["embedding"])
        except Exception as e:
            print(f"警告: 无法为 {mem_id} 生成embedding: {e}")
            md["embedding"] = None
            return None
    
    def build_index(self, use_gpu: bool = False):
        """
        构建FAISS索引
        
        参数:
            use_gpu: 是否使用GPU加速
        """
        print("正在构建FAISS索引...")
        
        # 收集所有embedding
        embeddings = []
        valid_indices = []
        
        for i, item in enumerate(self.memories):
            md = self._get_metadata(item)
            if md is None:
                # Skip memories without metadata (e.g., mem_001/mem_002)
                continue

            embedding_path = md.get('embedding')
            
            if embedding_path and embedding_path != 'null' and embedding_path != 'None':
                try:
                    # 加载.npy文件
                    emb = np.load(embedding_path)
                    embeddings.append(emb)
                    valid_indices.append(i)
                except Exception as e:
                    print(f"警告: 无法加载 {item['id']} 的embedding: {e}")
            else:
                # Missing embedding: try to generate and persist
                new_path = self._ensure_embedding(item)
                if new_path:
                    try:
                        emb = np.load(new_path)
                        embeddings.append(emb)
                        valid_indices.append(i)
                    except Exception as e:
                        print(f"警告: 无法加载 {item.get('id')} 的embedding: {e}")
        
        if not embeddings:
            raise ValueError(
                "没有找到任何有效的embedding！\n"
                "- 若你希望 search_memory 自动生成embedding：请确保 OPENAI_API_KEY 可用且可访问 embedding API。\n"
                "- 或者先运行 tool/embed_memories.py 生成 embeddings 并写回 metadata.embedding。"
            )
        
        # 转换为numpy数组
        embeddings_matrix = np.array(embeddings, dtype='float32')
        print(f"✓ 加载了 {len(embeddings)} 个embeddings，维度: {embeddings_matrix.shape}")
        
        # 归一化（用于余弦相似度）
        faiss.normalize_L2(embeddings_matrix)
        
        # 创建FAISS索引（内积 = 归一化后的余弦相似度）
        if use_gpu:
            # GPU版本
            res = faiss.StandardGpuResources()
            index_flat = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.index_cpu_to_gpu(res, 0, index_flat)
        else:
            # CPU版本
            self.index = faiss.IndexFlatIP(self.dimension)
        
        self.index.add(embeddings_matrix)
        self.valid_indices = valid_indices
        
        print(f"✓ 索引构建完成！共 {self.index.ntotal} 条记录")
        # Persist any newly generated metadata.embedding
        self._persist_memories()
    
    def save_index(self):
        """保存FAISS索引到文件"""
        if self.index is None:
            raise ValueError("请先调用 build_index() 构建索引")
        
        # 如果是GPU索引，先转到CPU
        if hasattr(self.index, 'index'):
            index_to_save = faiss.index_gpu_to_cpu(self.index)
        else:
            index_to_save = self.index
        
        # 确保目录存在
        Path(self.embeddings_dir).mkdir(exist_ok=True)
        
        # 保存索引
        faiss.write_index(index_to_save, self.index_path)
        
        # 保存映射关系
        np.save(self.mapping_path, np.array(self.valid_indices))
        
        print(f"✓ 索引已保存到: {self.index_path}")
    
    def load_index(self, use_gpu: bool = False):
        """
        从文件加载FAISS索引
        
        参数:
            use_gpu: 是否使用GPU加速
        """
        if not Path(self.index_path).exists():
            raise FileNotFoundError(
                f"索引文件不存在: {self.index_path}\n"
                f"请先运行 build_and_save_index() 构建索引"
            )
        
        # 加载索引
        self.index = faiss.read_index(self.index_path)
        
        # 如果需要GPU
        if use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # 加载映射关系
        self.valid_indices = np.load(self.mapping_path).tolist()
        
        print(f"✓ 索引已加载: {self.index.ntotal} 条记录")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        return_full_content: bool = False
    ) -> List[Dict]:
        """
        检索最相似的记录（作为Agent工具使用）
        
        参数:
            query: 查询文本
            top_k: 返回top-k结果
            return_full_content: 是否返回完整内容（True=返回所有字段，False=只返回id+摘要）
        
        返回:
            List[Dict]: 检索结果列表
        """
        if self.index is None:
            # 尝试加载索引
            try:
                self.load_index()
            except FileNotFoundError:
                # 如果没有索引，构建并保存
                print("未找到索引文件，正在构建...")
                self.build_index()
                self.save_index()
        
        # 生成查询embedding
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        try:
            response = client.embeddings.create(
                model="text-embedding-v4",
                input=query,
                dimensions=1024,
                encoding_format="float"
            )
        except Exception as e:
            return {
                "error": f"生成查询embedding失败: {str(e)}",
                "results": []
            }
        
        query_embedding = np.array([response.data[0].embedding], dtype='float32')
        faiss.normalize_L2(query_embedding)
        
        # FAISS检索
        similarities, indices = self.index.search(
            query_embedding, 
            min(top_k, self.index.ntotal)
        )
        
        # 构建返回结果
        results = []
        for score, idx in zip(similarities[0], indices[0]):
            if idx == -1:  # FAISS返回-1表示没有更多结果
                break
                
            original_idx = self.valid_indices[idx]
            memory = self.memories[original_idx]
            
            if return_full_content:
                # 返回完整内容
                result = {
                    'id': memory['id'],
                    'type': memory.get('type', ''),
                    'abstract': memory.get('abstract', ''),
                    'keyword': memory.get('keyword', ''),
                    'content': memory.get('content', {}),
                    'similarity_score': float(score),
                    'metadata': memory.get('metadata', {})
                }
            else:
                # 只返回轻量字段（但保留type，便于上层做memory_type过滤）
                result = {
                    'id': memory['id'],
                    'type': memory.get('type', ''),
                    'abstract': memory.get('abstract', ''),
                    'keyword': memory.get('keyword', ''),
                    'similarity_score': float(score)
                }
            
            results.append(result)
        
        return results
    
    def build_and_save_index(self, use_gpu: bool = False):
        """构建并保存索引（一次性操作）"""
        self.build_index(use_gpu=use_gpu)
        self.save_index()
    
    def incremental_update(self, use_gpu: bool = False):
        """
        增量更新索引 - 只为新记忆构建embedding和索引
        
        参数:
            use_gpu: 是否使用GPU加速
        
        返回:
            int: 新增的记忆数量
        """
        # Refresh mem.json if new memories were appended (common in long-running agent process)
        self._refresh_json_if_changed()

        # Fast path: nothing new since last seen (append-only assumption)
        if isinstance(self.memories, list) and len(self.memories) <= self._scan_cursor and self.index is not None:
            return 0

        # 如果索引不存在，直接全量构建
        if self.index is None:
            try:
                self.load_index(use_gpu=use_gpu)
            except FileNotFoundError:
                print("索引不存在，执行全量构建...")
                self.build_and_save_index(use_gpu=use_gpu)
                # After a full rebuild, we've effectively scanned everything.
                if isinstance(self.memories, list):
                    self._scan_cursor = len(self.memories)
                return len(self.valid_indices)
        
        # 获取已索引的记忆ID集合
        indexed_ids = {self.memories[i].get('id') for i in self.valid_indices if 0 <= i < len(self.memories)}
        
        # 找出新记忆
        new_embeddings = []
        new_indices = []

        start_i = max(int(self._scan_cursor or 0), 0)
        for i in range(start_i, len(self.memories)):
            item = self.memories[i]
            # 如果这个记忆还没被索引
            mem_id = item.get("id")
            if mem_id in indexed_ids:
                continue

            md = self._get_metadata(item)
            if md is None:
                continue

            embedding_path = md.get('embedding')
                
            if not (embedding_path and embedding_path != 'null' and embedding_path != 'None'):
                embedding_path = self._ensure_embedding(item)

            if embedding_path and embedding_path != 'null' and embedding_path != 'None':
                try:
                    # 加载新记忆的embedding
                    emb = np.load(embedding_path)
                    new_embeddings.append(emb)
                    new_indices.append(i)
                except Exception as e:
                    print(f"警告: 无法加载 {item.get('id')} 的embedding: {e}")
        
        # 如果没有新记忆
        if not new_embeddings:
            print("✓ 没有新记忆需要添加")
            if isinstance(self.memories, list):
                self._scan_cursor = len(self.memories)
            return 0
        
        # 转换为numpy数组并归一化
        new_embeddings_matrix = np.array(new_embeddings, dtype='float32')
        faiss.normalize_L2(new_embeddings_matrix)
        
        print(f"正在添加 {len(new_embeddings)} 条新记忆到索引...")
        
        # 增量添加到FAISS索引
        self.index.add(new_embeddings_matrix)
        
        # 更新映射关系
        self.valid_indices.extend(new_indices)
        
        # 保存更新后的索引
        self.save_index()
        # Persist any newly generated metadata.embedding
        self._persist_memories()

        if isinstance(self.memories, list):
            self._scan_cursor = len(self.memories)
        
        print(f"✓ 成功添加 {len(new_embeddings)} 条新记忆！当前总数: {self.index.ntotal}")
        
        return len(new_embeddings)


# ========== Agent工具函数接口 ==========

# 全局实例（避免重复加载）
_global_search_tool = None


def _auto_initialize():
    """
    自动初始化（内部使用）
    - 首次调用：构建索引
    - 后续调用：加载索引 + 增量更新
    """
    global _global_search_tool
    
    _global_search_tool = MemorySearchTool(
        json_path='mem.json',
        embeddings_dir='embeddings'
    )
    
    try:
        # 尝试加载已有索引
        _global_search_tool.load_index()
        # 自动增量更新新记忆
        _global_search_tool.incremental_update()
    except FileNotFoundError:
        # 首次使用，构建索引
        print("首次使用，正在构建索引...")
        _global_search_tool.build_and_save_index()


def search_memory(
    query: str,
    top_k: int = 10,
    return_full_content: bool = False,
    memory_type: Optional[Union[str, List[str]]] = None,
) -> Dict:
    """
    Search through memories using semantic similarity (vector search)
    
    Args:
        query: Search query in natural language
        top_k: Number of results to return (default: 10)
        return_full_content: False = only id/abstract/keyword, True = include full content (default: False)
        memory_type: Optional filter by memory type(s): "knowledge" | "idea" | "experience" | "resource"
    
    Returns:
        Dict with query, count, and results list (each with id, abstract, keyword, similarity_score)
    
    Examples:
        search_memory(query="polymer machine learning QSPR")
        search_memory(query="GNN molecular property", top_k=5, return_full_content=True)
    """
    global _global_search_tool
    
    # 自动初始化（如果未初始化）
    if _global_search_tool is None:
        _auto_initialize()
    else:
        # Long-running processes append to mem.json; keep index fresh.
        # This is designed to be cheap when nothing changed.
        try:
            _global_search_tool.incremental_update()
        except Exception:
            # Never fail retrieval due to background index maintenance issues
            pass
    
    # Normalize type filter
    allowed_types = {"knowledge", "idea", "experience", "resource", "integration"}
    type_filter: Optional[set] = None
    if memory_type is not None:
        if isinstance(memory_type, str):
            memory_type = [memory_type]
        # Keep only valid types to avoid accidental emptying
        type_filter = {t for t in memory_type if isinstance(t, str) and t in allowed_types}
        if not type_filter:
            type_filter = None

    # Execute search
    # If filtering by type, fetch more candidates then filter down to top_k.
    fetch_k = top_k
    if type_filter is not None:
        fetch_k = max(top_k * 8, top_k)
        fetch_k = min(fetch_k, 200)

    raw_results = _global_search_tool.search(
        query=query,
        top_k=fetch_k,
        return_full_content=return_full_content
    )

    results = raw_results
    if type_filter is not None and isinstance(raw_results, list):
        filtered: List[Dict] = []
        for r in raw_results:
            if not isinstance(r, dict):
                continue
            if r.get("type") in type_filter:
                filtered.append(r)
            if len(filtered) >= top_k:
                break
        results = filtered
    
    # 返回结构化结果
    return {
        "query": query,
        "count": len(results),
        "results": results
    }
