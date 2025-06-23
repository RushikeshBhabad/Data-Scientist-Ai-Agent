import os
from apikey import groq_api_key
import streamlit as st
import pandas as pd
import plotly.express as px
import sweetviz as sv
import io
import base64
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
import uuid
from datetime import datetime
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv
import requests
import json
from urllib.parse import quote, urlencode
import time
from typing import List, Dict, Any
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from apikey import GITHUB_TOKEN
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv(find_dotenv())

# Initialize LLM
llm = ChatGroq(
    temperature=0.2,
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192"
)

class DatasetFinder:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Dataset sources
        self.sources = {
            'kaggle': 'https://www.kaggle.com/api/v1/datasets/list',
            'github': 'https://api.github.com/search/repositories',
            'google_dataset_search': 'https://datasetsearch.research.google.com/search',
            'data_gov': 'https://catalog.data.gov/api/3/action/package_search',
            'uci_ml': 'https://archive.ics.uci.edu/ml/datasets',
            'world_bank': 'https://api.worldbank.org/v2/datacatalog',
            'fivethirtyeight': 'https://github.com/fivethirtyeight/data',
            'awesome_datasets': 'https://github.com/awesomedata/awesome-public-datasets'
        }
        
    def search_kaggle_datasets(self, query: str, max_results: int = 20) -> List[Dict]:
        """Simplified Kaggle search with essential fields only"""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            
            return [{
                'title': ds.title,
                'description': (ds.subtitle[:200] + "...") if hasattr(ds, 'subtitle') and ds.subtitle else "No description",
                'url': f"https://kaggle.com/{ds.ref}",
                'source': 'Kaggle',
                'type': 'Dataset',
                'downloads': getattr(ds, 'totalDownloads', 'N/A'),
                'size': getattr(ds, 'size', 'N/A')
            } for ds in api.dataset_list(search=query)[:max_results]]
            
        except Exception as e:
            st.error(f"Kaggle API error: {str(e)}")
            return []

    def search_github_datasets(self, query: str, max_results: int = 20) -> List[Dict]:
            """Search GitHub for datasets"""
            try:
                params = {
                    'q': f'{query} dataset filetype:csv OR filetype:json OR filetype:xlsx',
                    'sort': 'stars',
                    'order': 'desc',
                    'per_page': max_results
                }
                
                response = self.session.get(self.sources['github'], params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    datasets = []
                    
                    for repo in data.get('items', []):
                        datasets.append({
                            'title': repo['name'],
                            'description': repo.get('description', 'No description available'),
                            'url': repo['html_url'],
                            'source': 'GitHub',
                            'stars': repo.get('stargazers_count', 0),
                            'type': 'Repository/Dataset'
                        })
                    
                    return datasets
            except Exception as e:
                st.error(f"Error searching GitHub: {str(e)}")
                return []

    def search_data_gov(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search Data.gov datasets"""
        try:
            params = {
                'q': query,
                'rows': max_results,
                'sort': 'score desc'
            }
            
            response = self.session.get(self.sources['data_gov'], params=params)
            
            if response.status_code == 200:
                data = response.json()
                datasets = []
                
                for dataset in data.get('result', {}).get('results', []):
                    datasets.append({
                        'title': dataset.get('title', 'Untitled'),
                        'description': dataset.get('notes', 'No description available'),
                        'url': f"https://catalog.data.gov/dataset/{dataset.get('name', '')}",
                        'source': 'Data.gov',
                        'organization': dataset.get('organization', {}).get('title', 'Unknown'),
                        'type': 'Government Dataset'
                    })
                
                return datasets
            else:
                st.error(f"Data.gov API returned status code: {response.status_code}")
                return []
                
        except Exception as e:
            st.error(f"Error searching Data.gov: {str(e)}")
            return []  # Return empty list instead of None
    
    def search_uci_datasets(self, query: str) -> List[Dict]:
        """Search UCI ML Repository"""
        try:
            # UCI datasets list (static but comprehensive)
            uci_datasets = [
                {'name': 'Iris', 'url': 'https://archive.ics.uci.edu/ml/datasets/iris', 'description': 'Classic flower classification dataset'},
                {'name': 'Wine Quality', 'url': 'https://archive.ics.uci.edu/ml/datasets/wine+quality', 'description': 'Wine quality assessment data'},
                {'name': 'Titanic', 'url': 'https://archive.ics.uci.edu/ml/datasets/titanic', 'description': 'Titanic passenger survival data'},
                {'name': 'Heart Disease', 'url': 'https://archive.ics.uci.edu/ml/datasets/heart+disease', 'description': 'Heart disease diagnosis data'},
                {'name': 'Boston Housing', 'url': 'https://archive.ics.uci.edu/ml/datasets/housing', 'description': 'Boston housing price prediction data'},
                {'name': 'Adult Income', 'url': 'https://archive.ics.uci.edu/ml/datasets/adult', 'description': 'Adult income classification dataset'},
                {'name': 'Bank Marketing', 'url': 'https://archive.ics.uci.edu/ml/datasets/bank+marketing', 'description': 'Bank marketing campaign data'},
                {'name': 'Breast Cancer Wisconsin', 'url': 'https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)', 'description': 'Breast cancer diagnosis data'},
                {'name': 'Car Evaluation', 'url': 'https://archive.ics.uci.edu/ml/datasets/car+evaluation', 'description': 'Car evaluation dataset'},
                {'name': 'Glass Identification', 'url': 'https://archive.ics.uci.edu/ml/datasets/glass+identification', 'description': 'Glass type identification data'}
            ]
            
            # Filter based on query
            filtered_datasets = []
            query_lower = query.lower()
            
            for dataset in uci_datasets:
                if (query_lower in dataset['name'].lower() or 
                    query_lower in dataset['description'].lower()):
                    filtered_datasets.append({
                        'title': dataset['name'],
                        'description': dataset['description'],
                        'url': dataset['url'],
                        'source': 'UCI ML Repository',
                        'type': 'ML Dataset'
                    })
            
            return filtered_datasets
            
        except Exception as e:
            st.error(f"Error searching UCI datasets: {str(e)}")
            return []
    
    def enhance_search_with_llm(self, query: str) -> List[str]:
        """Use LLM to generate related search terms and improve query"""
        try:
            prompt = f"""
            Given the dataset search query: "{query}"
            
            Generate 5 related search terms and synonyms that would help find relevant datasets.
            Also suggest specific dataset names that might be related to this query.
            
            Return only the search terms, separated by commas, without any explanation.
            """
            
            message = HumanMessage(content=prompt)
            response = llm([message])
            
            # Parse the response to get search terms
            search_terms = [term.strip() for term in response.content.split(',')]
            return search_terms[:5]  # Limit to 5 terms
            
        except Exception as e:
            st.warning(f"LLM enhancement failed: {str(e)}")
            return [query]
    
    def search_all_sources(self, query: str, max_per_source: int = 10) -> Dict[str, List[Dict]]:
        """Search all dataset sources"""
        results = {}
        
        # Initialize all sources with empty lists
        results = {
            'Kaggle': [],
            'GitHub': [],
            'Data.gov': [],
            'UCI ML Repository': []
        }
        
        # Enhance query with LLM
        enhanced_queries = self.enhance_search_with_llm(query)
        all_queries = [query] + enhanced_queries
        
        with st.spinner('🔍 Searching across multiple sources...'):
            # Search each source
            progress_bar = st.progress(0)
            total_sources = 4
            
            # Kaggle
            st.write("Searching Kaggle...")
            kaggle_results = []
            for q in all_queries[:2]:  # Use first 2 queries
                kaggle_results.extend(self.search_kaggle_datasets(q, max_per_source//2))
            results['Kaggle'] = kaggle_results[:max_per_source]
            progress_bar.progress(1/total_sources)
            
            # GitHub
            st.write("Searching GitHub...")
            github_results = []
            for q in all_queries[:2]:
                res = self.search_github_datasets(q, max_per_source//2)
                github_results.extend(res if res is not None else [])  # Safely handle None
            results['GitHub'] = github_results[:max_per_source]
            progress_bar.progress(2/total_sources)
            
            # Data.gov
            st.write("Searching Data.gov...")
            datagov_results = []
            for q in all_queries[:2]:
                res = self.search_data_gov(q, max_per_source//2)
                datagov_results.extend(res if res is not None else [])  # Safely handle None
            results['Data.gov'] = datagov_results[:max_per_source]
            progress_bar.progress(3/total_sources)
            
            # UCI
            st.write("Searching UCI ML Repository...")
            uci_results = self.search_uci_datasets(query)
            results['UCI ML Repository'] = uci_results if uci_results is not None else []
            progress_bar.progress(4/total_sources)
            
            progress_bar.empty()
        
        return results
    
    def rank_results_with_llm(self, results: Dict[str, List[Dict]], original_query: str) -> List[Dict]:
        """Use LLM to rank and filter results based on relevance"""
        try:
            all_results = []
            for source, datasets in results.items():
                all_results.extend(datasets)
            
            if not all_results:
                return []
            
            # Create a summary of all results for LLM ranking
            results_summary = []
            for i, result in enumerate(all_results):
                results_summary.append(f"{i}: {result.get('title', 'Unknown')} - {result.get('description', 'No description')[:100]}")
            
            prompt = f"""
            Original search query: "{original_query}"
            
            Here are dataset search results:
            {chr(10).join(results_summary[:20])}  # Limit to avoid token limits
            
            Rank these results by relevance to the original query. Return only the indices (numbers) of the top 15 most relevant results, separated by commas.
            Consider factors like:
            - Direct relevance to the query topic
            - Quality indicators (GitHub stars, data source reputation)
            - Completeness of description
            
            Return only the numbers, no explanation.
            """
            
            message = HumanMessage(content=prompt)
            response = llm([message])
            
            # Parse ranked indices
            try:
                ranked_indices = [int(idx.strip()) for idx in response.content.split(',') if idx.strip().isdigit()]
                ranked_results = [all_results[i] for i in ranked_indices if i < len(all_results)]
                return ranked_results
            except:
                return all_results[:15]  # Fallback to first 15 results
                
        except Exception as e:
            st.warning(f"LLM ranking failed, using default order: {str(e)}")
            all_results = []
            for source, datasets in results.items():
                all_results.extend(datasets)
            return all_results[:15]

def main():
    st.set_page_config(
        page_title="🔍 Advanced Dataset Finder",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 Advanced Dataset Finder")
    st.markdown("### Find datasets from across the internet with AI-powered search")
    
    # Initialize dataset finder
    if 'dataset_finder' not in st.session_state:
        st.session_state.dataset_finder = DatasetFinder()
    
    # Sidebar
    st.sidebar.title("Search Options")
    
    # Search input
    query = st.text_input(
        "🔍 Enter your dataset search query:",
        placeholder="e.g., 'COVID-19 data', 'stock prices', 'climate change', 'customer behavior'",
        help="Enter keywords, dataset names, or describe what kind of data you're looking for"
    )
    
    max_results = st.sidebar.slider("Results per source", 5, 25, 10)
    
    # Search categories
    st.sidebar.markdown("### Quick Categories")
    categories = [
        "Finance & Economics", "Healthcare & Medicine", "Climate & Environment",
        "Social Media & Sentiment", "E-commerce & Marketing", "Transportation",
        "Education", "Sports", "Government", "Demographics"
    ]
    
    selected_category = st.sidebar.selectbox("Or select a category:", ["Custom Search"] + categories)
    
    if selected_category != "Custom Search":
        query = selected_category.lower().replace(" & ", " ").replace("&", "")
    
    # Advanced options
    with st.sidebar.expander("🛠️ Advanced Options"):
        file_types = st.multiselect(
            "Preferred file types:",
            ["CSV", "JSON", "Excel", "Parquet", "SQL", "API"],
            default=["CSV", "JSON"]
        )
        
        data_size = st.selectbox(
            "Dataset size preference:",
            ["Any size", "Small (<1MB)", "Medium (1MB-100MB)", "Large (>100MB)"]
        )
        
        recency = st.selectbox(
            "Data recency:",
            ["Any time", "Last year", "Last 5 years", "Last 10 years"]
        )
    
    # Search button
    if st.button("🚀 Search Datasets", type="primary") or query:
        if query.strip():
            # Perform search
            results = st.session_state.dataset_finder.search_all_sources(query, max_results)
            
            # Count total results
            total_results = sum(len(datasets) for datasets in results.values())
            
            if total_results > 0:
                st.success(f"✅ Found {total_results} datasets across multiple sources!")
                
                # Get AI-ranked results
                st.markdown("### 🤖 AI-Ranked Best Matches")
                ranked_results = st.session_state.dataset_finder.rank_results_with_llm(results, query)
                
                if ranked_results:
                    for i, dataset in enumerate(ranked_results[:10], 1):
                        with st.expander(f"#{i} {dataset.get('title', 'Unknown Dataset')} ({dataset.get('source', 'Unknown')})"):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**Description:** {dataset.get('description', 'No description available')}")
                                if 'organization' in dataset:
                                    st.markdown(f"**Organization:** {dataset['organization']}")
                                if 'stars' in dataset:
                                    st.markdown(f"**⭐ Stars:** {dataset['stars']}")
                                st.markdown(f"**Type:** {dataset.get('type', 'Dataset')}")
                            
                            with col2:
                                st.link_button("🔗 Open Dataset", dataset.get('url', '#'))
                

                # Export results
                st.markdown("### 📥 Export Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Prepare data for export
                    export_data = []
                    for source, datasets in results.items():
                        for dataset in datasets:
                            export_data.append({
                                'Title': dataset.get('title', 'Unknown'),
                                'Description': dataset.get('description', 'No description'),
                                'URL': dataset.get('url', ''),
                                'Source': source,
                                'Type': dataset.get('type', 'Dataset')
                            })
                    
                    if export_data:
                        df_export = pd.DataFrame(export_data)
                        csv = df_export.to_csv(index=False)
                        st.download_button(
                            "📄 Download as CSV",
                            csv,
                            f"dataset_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv"
                        )
                
                with col2:
                    # Generate search report
                    if st.button("📊 Generate Search Report"):
                        report = f"""
# Dataset Search Report
**Query:** {query}
**Search Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Results:** {total_results}

## Sources Searched:
{chr(10).join([f"- {source}: {len(datasets)} results" for source, datasets in results.items()])}

## Top Recommendations:
{chr(10).join([f"{i+1}. {dataset.get('title', 'Unknown')} ({dataset.get('source', 'Unknown')})" for i, dataset in enumerate(ranked_results[:5])])}
                        """
                        st.download_button(
                            "📋 Download Report",
                            report,
                            f"dataset_search_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            "text/markdown"
                        )
            
            else:
                st.warning("❌ No datasets found. Try different keywords or check your spelling.")
                
                # Suggest alternatives
                st.markdown("### 💡 Suggestions:")
                st.markdown("- Try broader search terms")
                st.markdown("- Use synonyms or related keywords")
                st.markdown("- Check popular categories in the sidebar")
                st.markdown("- Search for specific dataset names you know")
        
        else:
            st.warning("Please enter a search query.")
    
    # Footer
    st.markdown("---")
    st.markdown("### About This Tool")
    st.markdown("""
    This advanced dataset finder searches across multiple sources:
    - **Kaggle**: Machine learning and data science datasets
    - **GitHub**: Open source datasets and repositories
    - **Data.gov**: US government open data
    - **UCI ML Repository**: Classic machine learning datasets
    
    🤖 **AI-Enhanced**: Uses advanced language models to understand your queries and rank results by relevance.
    """)

if __name__ == "__main__":
    main()