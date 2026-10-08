"""
ArXiv Tool - Search and download papers from arXiv
"""

import arxiv
from pathlib import Path
from typing import List, Dict, Optional
import time


def search_arxiv(
    query: str,
    max_results: int = 5,
    sort_by: str = "relevance"
) -> Dict:
    """
    Search arXiv for papers
    
    Args:
        query: Search query (e.g., "polymer graph neural network")
        max_results: Maximum number of results to return
        sort_by: "relevance" or "submitted_date"
    
    Returns:
        Dict with search results
    """
    try:
        # Map sort_by to arxiv.SortCriterion
        sort_criterion = arxiv.SortCriterion.Relevance
        if sort_by == "submitted_date":
            sort_criterion = arxiv.SortCriterion.SubmittedDate
        
        # Search
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion
        )
        
        results = []
        for paper in search.results():
            results.append({
                "arxiv_id": paper.entry_id.split("/")[-1],
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary,
                "published": paper.published.strftime("%Y-%m-%d"),
                "pdf_url": paper.pdf_url,
                "categories": paper.categories
            })
        
        return {
            "status": "success",
            "query": query,
            "count": len(results),
            "papers": results
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Search failed: {str(e)}"
        }


def download_arxiv_paper(
    arxiv_id: str,
    download_dir: str = "papers/arxiv",
    filename: Optional[str] = None
) -> Dict:
    """
    Download a paper from arXiv by ID
    
    Args:
        arxiv_id: ArXiv ID (e.g., "2301.12345" or full URL)
        download_dir: Directory to save the PDF
        filename: Custom filename (optional, will use arxiv_id.pdf by default)
    
    Returns:
        Dict with download status and file path
    """
    try:
        # Clean arxiv_id if it's a full URL
        if "arxiv.org" in arxiv_id:
            arxiv_id = arxiv_id.split("/")[-1].replace(".pdf", "")
        
        # Create directory
        save_dir = Path(download_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine filename
        if filename is None:
            filename = f"{arxiv_id.replace('/', '_')}.pdf"
        
        save_path = save_dir / filename
        
        # Search for the paper
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        
        # Download
        paper.download_pdf(dirpath=str(save_dir), filename=filename)
        
        return {
            "status": "success",
            "arxiv_id": arxiv_id,
            "title": paper.title,
            "file_path": str(save_path),
            "message": f"Downloaded to {save_path}"
        }
    
    except StopIteration:
        return {
            "status": "error",
            "message": f"Paper {arxiv_id} not found on arXiv"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Download failed: {str(e)}"
        }


def search_and_download(
    query: str,
    download_dir: str = "papers/arxiv",
    max_results: int = 3
) -> Dict:
    """
    Search and download papers in one go
    
    Args:
        query: Search query
        download_dir: Directory to save PDFs
        max_results: Maximum number of papers to download
    
    Returns:
        Dict with search and download results
    """
    # Search
    search_result = search_arxiv(query, max_results)
    
    if search_result["status"] != "success":
        return search_result
    
    papers = search_result["papers"]
    
    if not papers:
        return {
            "status": "success",
            "message": "No papers found",
            "downloaded": []
        }
    
    # Download each paper
    downloaded = []
    failed = []
    
    for paper in papers:
        arxiv_id = paper["arxiv_id"]
        
        # Generate safe filename
        safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' 
                           for c in paper["title"])[:50]
        filename = f"{arxiv_id.replace('/', '_')}_{safe_title}.pdf"
        
        result = download_arxiv_paper(arxiv_id, download_dir, filename)
        
        if result["status"] == "success":
            downloaded.append({
                "arxiv_id": arxiv_id,
                "title": paper["title"],
                "file_path": result["file_path"]
            })
        else:
            failed.append({
                "arxiv_id": arxiv_id,
                "error": result["message"]
            })
        
        # Avoid rate limiting
        time.sleep(1)
    
    return {
        "status": "success",
        "query": query,
        "downloaded_count": len(downloaded),
        "failed_count": len(failed),
        "downloaded": downloaded,
        "failed": failed
    }


if __name__ == "__main__":
    # Test search
    print("Searching arXiv...")
    result = search_arxiv("polymer graph neural network", max_results=3)
    
    if result["status"] == "success":
        print(f"Found {result['count']} papers:")
        for paper in result["papers"]:
            print(f"  - {paper['title'][:60]}... ({paper['arxiv_id']})")
    
    # Test download
    # print("\nDownloading first paper...")
    # if result["papers"]:
    #     arxiv_id = result["papers"][0]["arxiv_id"]
    #     download_result = download_arxiv_paper(arxiv_id)
    #     print(download_result)

