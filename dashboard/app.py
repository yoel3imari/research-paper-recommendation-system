"""Streamlit dashboard for arXiv paper recommendation system.

Implements Phase 6 of the project (docs/architecture.md §6):
- Interest query mode: free-text research interest → /recommend/query
- Seed paper mode: title search → paper selection → /recommend/paper/{id}

UI per card: title (link), similarity score badge, category chips, authors,
publication date, abstract expander, "View on arXiv" button.

Uses @st.cache_data on API calls. API_BASE_URL from env (default localhost:8000).
"""

import os
from typing import Any

import requests
import streamlit as st

# Constants
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
DEFAULT_K = 10
MIN_YEAR = 1991  # arXiv started in 1991
MAX_YEAR = 2026

# Helper functions for API calls
@st.cache_data(ttl=300)
def search_papers(query: str, limit: int = 10) -> dict[str, Any]:
    """Search papers by title substring.
    
    Args:
        query: Search query string
        limit: Maximum number of results to return
        
    Returns:
        PaperSearchResponse dict with results and total
        
    Raises:
        requests.RequestException: If API request fails
    """
    url = f"{API_BASE_URL}/papers/search"
    params = {"q": query, "limit": limit}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error searching papers: {e}")
        return {"results": [], "total": 0}


@st.cache_data(ttl=300)
def recommend_query(
    query: str, 
    k: int = DEFAULT_K, 
    year_min: int | None = None, 
    categories: list[str] | None = None
) -> dict[str, Any]:
    """Get recommendations based on a free-text query.
    
    Args:
        query: Research interest query text
        k: Number of recommendations to return
        year_min: Minimum publication year (optional)
        categories: Category filters (optional)
        
    Returns:
        RecommendResponse dict with results and metadata
        
    Raises:
        requests.RequestException: If API request fails
    """
    url = f"{API_BASE_URL}/recommend/query"
    params: dict[str, str | int] = {"q": query, "k": k}

    if year_min:
        params["year_min"] = year_min
    if categories:
        params["categories"] = ",".join(categories)
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error getting recommendations: {e}")
        return {"query": query, "k": k, "took_ms": 0.0, "results": []}


@st.cache_data(ttl=300)
def recommend_paper(
    arxiv_id: str,
    k: int = DEFAULT_K,
    year_min: int | None = None,
    categories: list[str] | None = None
) -> dict[str, Any]:
    """Get recommendations based on a seed paper.
    
    Args:
        arxiv_id: arXiv identifier (e.g., "2006.00001")
        k: Number of recommendations to return
        year_min: Minimum publication year (optional)
        categories: Category filters (optional)
        
    Returns:
        RecommendResponse dict with results and metadata
        
    Raises:
        requests.RequestException: If API request fails
    """
    url = f"{API_BASE_URL}/recommend/paper/{arxiv_id}"
    params: dict[str, str | int] = {"k": k}

    if year_min:
        params["year_min"] = year_min
    if categories:
        params["categories"] = ",".join(categories)
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error getting recommendations for paper {arxiv_id}: {e}")
        return {"query": arxiv_id, "k": k, "took_ms": 0.0, "results": []}


def format_categories(categories: list[str]) -> str:
    """Format categories as comma-separated string."""
    return ", ".join(categories)


def render_paper_card(paper: dict[str, Any], score: float) -> None:
    """Render a single paper recommendation card.
    
    Args:
        paper: Paper dict from API response
        score: Similarity score (0-1)
    """
    # Format score as percentage with 2 decimal places
    score_pct = score * 100
    
    # Create card container
    with st.container(border=True):
        # Title with link
        st.markdown(
            f"**[{paper['title']}]({paper['arxiv_url']})**",
            unsafe_allow_html=True
        )
        
        # Score badge and metadata row
        col1, col2, col3 = st.columns([1, 2, 2])
        
        # Score badge
        st.markdown(
            f"<span style='background-color: #e3f2fd; color: #1565c0; "
            f"padding: 4px 8px; border-radius: 12px; font-weight: bold;'>"
            f"Similarity: {score_pct:.2f}%</span>",
            unsafe_allow_html=True
        )
        
        # Categories
        if paper.get("categories"):
            st.markdown(
                f"**Categories:** {format_categories(paper['categories'])}",
                help="arXiv categories"
            )
        
        # Authors
        if paper.get("authors"):
            authors_str = ", ".join(paper["authors"][:3])  # Show first 3 authors
            if len(paper["authors"]) > 3:
                authors_str += f" ... (+{len(paper['authors']) - 3} more)"
            st.markdown(f"**Authors:** {authors_str}")
        
        # Publication date
        if paper.get("update_date"):
            st.markdown(f"**Published:** {paper['update_date']}")
        
        # Abstract expander
        if paper.get("abstract"):
            with st.expander("Abstract"):
                st.write(paper["abstract"])
        
        # View on arXiv button
        st.markdown(
            f"[View on arXiv 🔗]({paper['arxiv_url']})",
            unsafe_allow_html=True
        )


def interest_query_mode() -> None:
    """Render the interest query mode interface."""
    st.header("🔍 Interest Query Mode")
    st.write("Enter a research interest or topic to find similar papers.")
    
    # Query input
    query = st.text_area(
        "Research Interest",
        height=100,
        placeholder=(
            "e.g., 'retrieval augmented generation for code', "
            "'efficient transformers inference'"
        )
    )
    
    # Sidebar controls for both modes
    with st.sidebar:
        st.subheader("Recommendation Settings")
        
        # Number of results
        k = st.slider("Number of recommendations (k)", min_value=1, max_value=50, value=DEFAULT_K)
        
        # Year filter
        year_min = st.slider(
            "Minimum publication year",
            min_value=MIN_YEAR,
            max_value=MAX_YEAR,
            value=None,
            help="Filter papers published in or after this year"
        )
        
        # Categories (optional)
        categories = st.text_area(
            "Category filters (comma-separated, e.g., 'cs.LG, cs.AI')",
            height=80,
            placeholder="Leave empty for all categories"
        )
        
        # Parse categories
        parsed_categories = None
        if categories.strip():
            parsed_categories = [cat.strip() for cat in categories.split(",") if cat.strip()]
    
    # Search button
    if st.button("Find Similar Papers", type="primary"):
        if not query.strip():
            st.error("Please enter a research interest.")
            return
        
        with st.spinner("Searching for recommendations..."):
            response = recommend_query(
                query=query.strip(),
                k=k,
                year_min=year_min,
                categories=parsed_categories
            )
            
            if response.get("results"):
                st.success(f"Found {len(response['results'])} recommendations!")
                
                # Display results
                for i, rec in enumerate(response["results"]):
                    st.markdown(f"### Result {i + 1}")
                    render_paper_card(rec["paper"], rec["score"])
                    st.markdown("---")
            else:
                st.info("No recommendations found. Try adjusting your query or filters.")


def seed_paper_mode() -> None:
    """Render the seed paper mode interface."""
    st.header("📄 Seed Paper Mode")
    st.write("Search for a paper by title, then get similar papers based on it.")
    
    # Search for seed paper
    with st.container(border=True):
        st.subheader("🔍 Find Seed Paper")
        
        # Sidebar controls (shared with interest mode)
        with st.sidebar:
            st.subheader("Recommendation Settings")
            
            # Number of results
            k = st.slider(
                "Number of recommendations (k)",
                min_value=1,
                max_value=50,
                value=DEFAULT_K
            )
            
            # Year filter
            year_min = st.slider(
                "Minimum publication year",
                min_value=MIN_YEAR,
                max_value=MAX_YEAR,
                value=MIN_YEAR,
                help="Filter papers published in or after this year (set to min to disable)"
            )
            # If user wants to disable year filter, set to min value
            if year_min == MIN_YEAR:
                year_min = None
            
            # Categories (optional)
            categories = st.text_area(
                "Category filters (comma-separated, e.g., 'cs.LG, cs.AI')",
                height=80,
                placeholder="Leave empty for all categories"
            )
            
            # Parse categories
            parsed_categories = None
            if categories.strip():
                parsed_categories = [cat.strip() for cat in categories.split(",") if cat.strip()]
        
        # Search input
        seed_query = st.text_input(
            "Search for a paper by title or keywords",
            placeholder="e.g., 'attention mechanism', 'transformer architecture'"
        )
        
        # Search button
        if st.button("Search for Papers", type="primary"):
            if not seed_query.strip():
                st.error("Please enter a search query.")
                return
            
            with st.spinner("Searching for papers..."):
                search_response = search_papers(seed_query.strip(), limit=10)
                
                if search_response.get("results"):
                    st.success(f"Found {search_response['total']} papers.")
                    
                    # Create options for dropdown
                    paper_options = {}
                    for paper in search_response["results"]:
                        option_text = f"{paper['title']} ({paper['id']})"
                        paper_options[option_text] = paper
                    
                    # Paper selector
                    selected_option = st.selectbox(
                        "Select a seed paper:",
                        options=list(paper_options.keys()),
                        help="Choose a paper to get similar recommendations"
                    )
                    
                    if selected_option:
                        selected_paper = paper_options[selected_option]
                        
                        # Display selected paper info
                        st.markdown("### Selected Seed Paper")
                        st.markdown(f"**Title:** {selected_paper['title']}")
                        st.markdown(f"**arXiv ID:** {selected_paper['id']}")
                        
                        if st.button("Get Similar Papers", type="secondary"):
                            with st.spinner("Getting recommendations..."):
                                response = recommend_paper(
                                    selected_paper["id"],
                                    k=k,
                                    year_min=year_min,
                                    categories=parsed_categories
                                )
                                
                                if response.get("results"):
                                    st.success(f"Found {len(response['results'])} similar papers!")
                                    
                                    # Display results
                                    for i, rec in enumerate(response["results"]):
                                        st.markdown(f"### Result {i + 1}")
                                        render_paper_card(rec["paper"], rec["score"])
                                        st.markdown("---")
                                else:
                                    st.info("No similar papers found. Try adjusting filters.")
                else:
                    st.error("No papers found. Try a different search query.")


def main() -> None:
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="arXiv Paper Recommender",
        page_icon="🔍",
        layout="wide"
    )
    
    # Header
    st.title("🔍 arXiv Paper Recommendation System")
    st.markdown(
        "Find similar research papers based on your interests or seed papers. "
        "[API Documentation](/docs)"
    )
    
    # Sidebar mode selection
    st.sidebar.title("Mode Selection")
    mode = st.sidebar.radio(
        "Choose recommendation mode:",
        ["Interest Query", "Seed Paper"],
        index=0
    )
    
    # Render selected mode
    if mode == "Interest Query":
        interest_query_mode()
    else:
        seed_paper_mode()


if __name__ == "__main__":
    main()