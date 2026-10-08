"""
嵌入记忆模块 - 为mem.json中的abstract生成embedding并保存为.npy文件

使用方法:
    python embed_memories.py --api_key YOUR_API_KEY
"""

import json
import numpy as np
from pathlib import Path
from openai import OpenAI
import argparse
import time
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def embed_abstracts_to_files(
    json_path='mem.json',
    api_key=None,
    embeddings_dir='embeddings',
    batch_size=10,
    delay=0.5
):
    """
    为mem.json中的每个abstract生成embedding，保存为单独的.npy文件
    
    参数:
        json_path: mem.json文件路径
        api_key: 阿里云API密钥
        embeddings_dir: embedding文件保存目录
        batch_size: 每批处理数量（用于显示进度）
        delay: 每次API调用间隔（秒），避免速率限制
    """
    # 创建embeddings目录
    embeddings_path = Path(embeddings_dir)
    embeddings_path.mkdir(exist_ok=True)
    
    # 初始化OpenAI客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    # 读取JSON
    print(f"正在读取: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    memories = data['memory']
    total = len(memories)
    processed = 0
    skipped = 0
    embedded = 0
    
    print(f"共有 {total} 条记录待处理\n")
    
    # 处理每条记录
    for i, item in enumerate(memories, 1):
        abstract = item.get('abstract', '')
        mem_id = item['id']
        
        # 允许部分记忆没有metadata（例如 mem_001/mem_002），此时跳过以免崩溃
        if 'metadata' not in item or not isinstance(item.get('metadata'), dict):
            print(f"[{i}/{total}] {mem_id}: 缺少metadata，跳过")
            skipped += 1
            continue
        
        # 检查是否已有embedding
        npy_filename = f"{mem_id}_embedding.npy"
        npy_path = embeddings_path / npy_filename
        
        if npy_path.exists():
            print(f"[{i}/{total}] {mem_id}: 已存在，跳过")
            item['metadata']['embedding'] = f"{embeddings_dir}/{npy_filename}"
            skipped += 1
            continue
        
        # 检查abstract是否有效
        if not abstract or abstract == 'N/A' or len(abstract.strip()) < 10:
            print(f"[{i}/{total}] {mem_id}: 无效abstract，跳过")
            item['metadata']['embedding'] = None
            skipped += 1
            continue
        
        try:
            # 显示进度
            preview = abstract[:60].replace('\n', ' ')
            print(f"[{i}/{total}] {mem_id}: 正在嵌入... ({preview}...)")
            
            # 生成embedding
            response = client.embeddings.create(
                model="text-embedding-v4",
                input=abstract,
                dimensions=1024,
                encoding_format="float"
            )
            
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            # 保存为.npy文件
            np.save(npy_path, embedding)
            
            # 更新JSON中的路径
            item['metadata']['embedding'] = f"{embeddings_dir}/{npy_filename}"
            
            embedded += 1
            processed += 1
            
            print(f"[{i}/{total}] {mem_id}: ✓ 完成")
            
            # 延迟，避免API速率限制
            if delay > 0:
                time.sleep(delay)
                
        except Exception as e:
            print(f"[{i}/{total}] {mem_id}: ✗ 错误 - {str(e)}")
            item['metadata']['embedding'] = None
            processed += 1
            continue
    
    # 保存更新后的JSON
    print(f"\n正在保存更新后的JSON...")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"  新嵌入: {embedded} 条")
    print(f"  跳过: {skipped} 条")
    print(f"  总计: {total} 条")
    print(f"  Embeddings保存在: {embeddings_dir}/")
    print(f"{'='*60}\n")


def check_embeddings_status(json_path='mem.json'):
    """
    检查当前embedding状态
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data['memory'])
    has_embedding = 0
    no_embedding = 0
    no_abstract = 0
    
    for item in data['memory']:
        embedding = item['metadata'].get('embedding')
        abstract = item.get('abstract', '')
        
        if embedding and embedding != 'null':
            has_embedding += 1
        elif not abstract or abstract == 'N/A' or len(abstract.strip()) < 10:
            no_abstract += 0
        else:
            no_embedding += 1
    
    print(f"\n{'='*60}")
    print(f"Embedding状态检查 - {json_path}")
    print(f"{'='*60}")
    print(f"  总记录数: {total}")
    print(f"  已有embedding: {has_embedding} ({has_embedding/total*100:.1f}%)")
    print(f"  待嵌入: {no_embedding} ({no_embedding/total*100:.1f}%)")
    print(f"  无效abstract: {no_abstract} ({no_abstract/total*100:.1f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='为mem.json生成embeddings')
    parser.add_argument('--json_path', type=str, default='mem.json', help='mem.json路径')
    parser.add_argument('--embeddings_dir', type=str, default='embeddings', help='embedding保存目录')
    parser.add_argument('--delay', type=float, default=0.5, help='API调用间隔（秒）')
    parser.add_argument('--check', action='store_true', help='仅检查状态，不执行嵌入')
    
    args = parser.parse_args()
    
    check_embeddings_status(args.json_path)
    
    embed_abstracts_to_files(
        json_path=args.json_path,
        api_key=api_key,
        embeddings_dir=args.embeddings_dir,
        delay=args.delay
    )

