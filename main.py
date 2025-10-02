import streamlit as st
import requests
import pandas as pd
import time
import xml.etree.ElementTree as ET
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from collections import Counter
import numpy as np
from googlesearch import search as google_search
import webbrowser
from urllib.parse import quote

# Configure the page
st.set_page_config(
    page_title="Paulo Monteiro - Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def scrape_arxiv(subjects, max_results_per_subject=20):
    """Scrape articles from arXiv API"""
    st.info("🔍 Scraping arXiv...")
    
    all_articles_data = []
    arxiv_base_url = "http://export.arxiv.org/api/query?"
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, subject in enumerate(subjects):
        status_text.text(f"Searching for: {subject}")
        
        search_query = f'all:"{subject}"'
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results_per_subject
        }

        try:
            response = requests.get(arxiv_base_url, params=params)
            response.raise_for_status()

            # Parse XML response
            root = ET.fromstring(response.text)
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', namespace):
                title = entry.find('atom:title', namespace).text
                title = re.sub(r'\s+', ' ', title).strip() if title else "No title"
                
                summary = entry.find('atom:summary', namespace)
                abstract = summary.text if summary is not None else "No abstract available"
                abstract = re.sub(r'\s+', ' ', abstract).strip()
                
                # Find the PDF link
                link = ""
                for link_elem in entry.findall('atom:link', namespace):
                    if link_elem.get('title') == 'pdf' or link_elem.get('type') == 'application/pdf':
                        link = link_elem.get('href', '')
                        break
                
                if not link:  # Fallback to any link
                    for link_elem in entry.findall('atom:link', namespace):
                        if link_elem.get('rel') == 'alternate':
                            link = link_elem.get('href', '')
                            break

                all_articles_data.append({
                    'Source': 'arXiv',
                    'Type': 'Scientific Paper',
                    'Subject': subject,
                    'Title': title,
                    'Abstract': abstract,
                    'Link': link
                })

            progress_bar.progress((i + 1) / len(subjects))
            time.sleep(1)  # Be polite to the API

        except Exception as e:
            st.error(f"Error scraping arXiv for subject '{subject}': {str(e)}")
    
    status_text.text("✅ arXiv scraping completed!")
    return all_articles_data

def search_google_web(subjects, max_results_per_subject=10):
    """Search Google for web results related to subjects"""
    st.info("🌐 Searching Google Web...")
    
    all_web_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, subject in enumerate(subjects):
        status_text.text(f"Google search for: {subject}")
        
        try:
            # Perform Google search
            search_results = list(google_search(
                f"{subject} research latest developments 2024",
                num_results=max_results_per_subject,
                lang="en"
            ))
            
            for j, url in enumerate(search_results):
                # Extract domain for display
                domain = re.findall(r'https?://([^/]+)', url)
                domain_name = domain[0] if domain else "Web Resource"
                
                all_web_data.append({
                    'Source': domain_name,
                    'Type': 'Web Resource',
                    'Subject': subject,
                    'Title': f"{subject} - Web Resource {j+1}",
                    'Description': f"Web resource about {subject} from {domain_name}",
                    'Link': url
                })
            
            progress_bar.progress((i + 1) / len(subjects))
            time.sleep(2)  # Be polite to Google
            
        except Exception as e:
            st.error(f"Error searching Google for '{subject}': {str(e)}")
            st.info("This might be due to rate limiting. Trying next subject...")
    
    status_text.text("✅ Google web search completed!")
    return all_web_data

def open_link(url):
    """Function to open links in new tab"""
    try:
        webbrowser.open_new_tab(url)
        return True
    except:
        return False

def analyze_topics(df):
    """Perform basic topic analysis on abstracts"""
    if df.empty:
        return None, None
    
    # Combine all abstracts/descriptions
    text_column = 'Abstract' if 'Abstract' in df.columns else 'Description'
    all_text = ' '.join(df[text_column].astype(str))
    
    # Clean text
    words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())
    
    # Remove common stop words
    stop_words = {'this', 'that', 'with', 'have', 'from', 'they', 'which', 'their', 
                 'what', 'will', 'would', 'there', 'were', 'been', 'have', 'when',
                 'then', 'them', 'such', 'some', 'these', 'than', 'were', 'about',
                 'also', 'more', 'most', 'other', 'only', 'just', 'like'}
    
    filtered_words = [word for word in words if word not in stop_words]
    
    # Get most common words
    word_freq = Counter(filtered_words).most_common(20)
    
    return word_freq, all_text

def create_wordcloud(text):
    """Create a word cloud from text"""
    if not text:
        return None
    
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def main():
    # Header with author attribution
    st.title("🔍 Paulo Monteiro - Research Assistant")
    st.markdown("""
    *Created by Paulo Monteiro - 1/10/2025*
    
    This app searches both scientific papers and web resources based on your research interests,
    analyzes the content, and provides insights through visualizations.
    """)
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Default subjects
    default_subjects = [
        "large language models", "NLP", "LLM", "Topic Modelling", 
        "Machine Learning", "Web Services", "Algebra", "Calculus matrix",
        "Reasoning", "Data Visualization", "Rendering Pipeline", 
        "Neural Rendering", "Shading Equations", "Vectorial Calculus", 
        "Random Forest", "Neural Networks"
    ]
    
    # Subject input
    st.sidebar.subheader("Research Subjects")
    subjects_input = st.sidebar.text_area(
        "Enter scientific subjects (one per line or comma-separated):",
        value="\n".join(default_subjects),
        height=150
    )
    
    # Parse subjects
    if "\n" in subjects_input:
        subjects = [s.strip() for s in subjects_input.split("\n") if s.strip()]
    else:
        subjects = [s.strip() for s in subjects_input.split(",") if s.strip()]
    
    # Scraping parameters
    st.sidebar.subheader("Search Parameters")
    max_scientific_results = st.sidebar.slider(
        "Max scientific papers per subject:",
        min_value=5,
        max_value=50,
        value=20
    )
    
    max_web_results = st.sidebar.slider(
        "Max web results per subject:",
        min_value=3,
        max_value=20,
        value=10
    )
    
    # Main content area with dual tabs for Web vs Scientific
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Overview", 
        "📚 Scientific Papers", 
        "🌐 Web Resources", 
        "📈 Analyze Topics", 
        "💾 Export Data"
    ])
    
    with tab1:
        st.header("Overview")
        st.markdown("""
        ### Search Capabilities:
        
        **📚 Scientific Papers Tab:**
        - **arXiv**: Open-access repository with permissive API access
        - Academic research papers and pre-prints
        - PDF downloads available
        
        **🌐 Web Resources Tab:**
        - **Google Search**: Latest web resources, articles, and news
        - Real-time information from various sources
        - Blog posts, tutorials, and industry updates
        
        ### Features:
        - Dual search approach (academic + web)
        - Interactive link opening
        - Topic modeling and word frequency analysis
        - Export results to CSV
        - Created by Paulo Monteiro (1/10/2025)
        """)
        
        if subjects:
            st.info(f"🎯 Ready to search for: {', '.join(subjects[:5])}{'...' if len(subjects) > 5 else ''}")
    
    with tab2:
        st.header("Scientific Papers Search")
        st.markdown("Search academic papers from arXiv and other scientific repositories")
        
        if st.button("🚀 Search Scientific Papers", type="primary", key="scientific_search"):
            if not subjects:
                st.error("Please enter at least one scientific subject.")
                return
            
            with st.spinner("Searching scientific papers from arXiv..."):
                articles_data = scrape_arxiv(subjects, max_scientific_results)
            
            if articles_data:
                df = pd.DataFrame(articles_data)
                st.session_state.scientific_df = df
                st.success(f"✅ Successfully found {len(df)} scientific papers!")
                
                # Display results
                st.subheader(f"Scientific Papers ({len(df)} total)")
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    selected_subjects = st.multiselect(
                        "Filter by subject:",
                        options=df['Subject'].unique(),
                        default=df['Subject'].unique(),
                        key="scientific_subjects"
                    )
                
                filtered_df = df[df['Subject'].isin(selected_subjects)]
                
                # Display articles with clickable links
                for idx, row in filtered_df.iterrows():
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"**{row['Title']}**")
                            st.write(f"*Subject: {row['Subject']}*")
                            st.write(f"{row['Abstract'][:200]}...")
                        with col2:
                            if st.button("📄 Open", key=f"sci_{idx}"):
                                if open_link(row['Link']):
                                    st.success("Opening paper...")
                                else:
                                    st.error("Could not open link")
                        st.markdown("---")
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Papers", len(df))
                with col2:
                    st.metric("Unique Subjects", df['Subject'].nunique())
                with col3:
                    st.metric("Source", "arXiv")
                
            else:
                st.error("No scientific papers were found. Please try different search terms.")
    
    with tab3:
        st.header("Web Resources Search")
        st.markdown("Search latest web resources, articles, and news from Google")
        
        if st.button("🌐 Search Web Resources", type="primary", key="web_search"):
            if not subjects:
                st.error("Please enter at least one scientific subject.")
                return
            
            with st.spinner("Searching web resources from Google..."):
                web_data = search_google_web(subjects, max_web_results)
            
            if web_data:
                df = pd.DataFrame(web_data)
                st.session_state.web_df = df
                st.success(f"✅ Successfully found {len(df)} web resources!")
                
                # Display results
                st.subheader(f"Web Resources ({len(df)} total)")
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    selected_subjects = st.multiselect(
                        "Filter by subject:",
                        options=df['Subject'].unique(),
                        default=df['Subject'].unique(),
                        key="web_subjects"
                    )
                
                filtered_df = df[df['Subject'].isin(selected_subjects)]
                
                # Display web resources with clickable links
                for idx, row in filtered_df.iterrows():
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"**{row['Title']}**")
                            st.write(f"*Source: {row['Source']} | Subject: {row['Subject']}*")
                            st.write(f"{row['Description']}")
                        with col2:
                            if st.button("🌐 Open", key=f"web_{idx}"):
                                if open_link(row['Link']):
                                    st.success("Opening web resource...")
                                else:
                                    st.error("Could not open link")
                        st.markdown("---")
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Resources", len(df))
                with col2:
                    st.metric("Unique Subjects", df['Subject'].nunique())
                with col3:
                    st.metric("Sources", df['Source'].nunique())
                
            else:
                st.error("No web resources were found. Please try different search terms.")
    
    with tab4:
        st.header("Topic Analysis")
        
        # Let user choose which data to analyze
        analysis_option = st.radio(
            "Choose data to analyze:",
            ["Scientific Papers", "Web Resources", "Combined Data"],
            horizontal=True
        )
        
        if analysis_option == "Scientific Papers" and 'scientific_df' in st.session_state:
            df = st.session_state.scientific_df
            data_type = "Scientific Papers"
        elif analysis_option == "Web Resources" and 'web_df' in st.session_state:
            df = st.session_state.web_df
            data_type = "Web Resources"
        elif analysis_option == "Combined Data" and ('scientific_df' in st.session_state or 'web_df' in st.session_state):
            # Combine both datasets if available
            dfs = []
            if 'scientific_df' in st.session_state:
                dfs.append(st.session_state.scientific_df)
            if 'web_df' in st.session_state:
                dfs.append(st.session_state.web_df)
            df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            data_type = "Combined Data"
        else:
            st.warning(f"Please search for {analysis_option.lower()} first using the appropriate tab.")
            return
        
        if df.empty:
            st.warning("No data available for analysis.")
            return
        
        # Perform topic analysis
        word_freq, all_text = analyze_topics(df)
        
        if word_freq:
            st.success(f"Analyzing {len(df)} {data_type.lower()}")
            
            # Create columns for different visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Word Frequency")
                words, counts = zip(*word_freq)
                fig = px.bar(
                    x=counts,
                    y=words,
                    orientation='h',
                    title=f"Most Frequent Words in {data_type}",
                    labels={'x': 'Frequency', 'y': 'Words'}
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Word Cloud")
                wordcloud_fig = create_wordcloud(all_text)
                st.pyplot(wordcloud_fig)
            
            # Subject distribution
            st.subheader(f"Content by Subject")
            subject_counts = df['Subject'].value_counts()
            fig_subjects = px.pie(
                values=subject_counts.values,
                names=subject_counts.index,
                title=f"Distribution of {data_type} by Subject"
            )
            st.plotly_chart(fig_subjects, use_container_width=True)
            
            # Source/Type distribution
            st.subheader(f"Content by Type/Source")
            if 'Type' in df.columns:
                type_counts = df['Type'].value_counts()
                fig_types = px.bar(
                    x=type_counts.index,
                    y=type_counts.values,
                    title=f"Number of Items by Type",
                    labels={'x': 'Type', 'y': 'Count'}
                )
                st.plotly_chart(fig_types, use_container_width=True)
    
    with tab5:
        st.header("Export Data")
        
        export_option = st.radio(
            "Choose data to export:",
            ["Scientific Papers", "Web Resources", "Combined Data"],
            horizontal=True
        )
        
        if export_option == "Scientific Papers" and 'scientific_df' in st.session_state:
            df = st.session_state.scientific_df
            filename_suffix = "scientific_papers"
        elif export_option == "Web Resources" and 'web_df' in st.session_state:
            df = st.session_state.web_df
            filename_suffix = "web_resources"
        elif export_option == "Combined Data" and ('scientific_df' in st.session_state or 'web_df' in st.session_state):
            dfs = []
            if 'scientific_df' in st.session_state:
                dfs.append(st.session_state.scientific_df)
            if 'web_df' in st.session_state:
                dfs.append(st.session_state.web_df)
            df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            filename_suffix = "combined_research"
        else:
            st.warning(f"No {export_option.lower()} data available to export.")
            return
        
        if df.empty:
            st.warning("No data to export.")
            return
        
        st.subheader("Download Data")
        st.dataframe(df, use_container_width=True, height=300)
        
        # Convert DataFrame to CSV
        csv = df.to_csv(index=False)
        
        st.download_button(
            label=f"📥 Download {export_option} as CSV",
            data=csv,
            file_name=f"paulo_monteiro_{filename_suffix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            type="primary"
        )
        
        # Show data statistics
        st.subheader("Data Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Info:**")
            st.write(f"- Total items: {len(df)}")
            if 'Type' in df.columns:
                st.write(f"- Types: {', '.join(df['Type'].unique())}")
            st.write(f"- Sources: {df['Source'].nunique()}")
            st.write(f"- Subjects: {df['Subject'].nunique()}")
            
        with col2:
            st.write("**Sample Content:**")
            text_column = 'Abstract' if 'Abstract' in df.columns else 'Description'
            for i, text in enumerate(df[text_column].head(3)):
                st.write(f"{i+1}. {text[:100]}...")

if __name__ == "__main__":
    main()
