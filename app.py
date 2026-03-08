import streamlit as st
import os
from pathlib import Path
import tempfile
import pandas as pd
from datetime import datetime
import json

from resume_search import SimpleResumeSearch

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
if 'resumedb' not in st.session_state:
    st.session_state.resumedb = None
if 'ranked_results' not in st.session_state:
    st.session_state.ranked_results = None

st.title("AI-Powered Resume Screening System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # API Key (try secrets first, then env, then input)
    default_api_key = ""
    try:
        default_api_key = st.secrets.get("OPENAI_API_KEY", "")
    except:
        default_api_key = os.environ.get('OPENAI_API_KEY', '')
    
    api_key = st.text_input(
        "OpenAI API Key", 
        type="password",
        value=default_api_key,
        help="Enter your OpenAI API key (or set in secrets)"
    )
    
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
    
    st.markdown("---")
    
    # Database info
    st.subheader("Database Status")
    if st.session_state.resumedb:
        count = st.session_state.resumedb.collection.count()
        st.success(f"{count} resumes loaded")
    else:
        st.info("No database loaded")
    
    # Clear database button
    if st.button("Clear Database"):
        if st.session_state.resumedb:
            st.session_state.resumedb.clear_database()
            st.success("Database cleared!")
            st.rerun()

# Main tabs
tab1, tab2, tab3 = st.tabs(["Upload Resumes", "Screen Candidates", "Results"])

# Tab 1: Upload Resumes
with tab1:
    st.header("Upload Resumes")
    
    uploaded_files = st.file_uploader(
        "Choose PDF or DOCX files",
        type=['pdf', 'docx'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"{len(uploaded_files)} files selected")
        
        if st.button("Process Resumes", type="primary"):
            if not api_key:
                st.error("Please enter OpenAI API key in sidebar")
            else:
                # Show immediate feedback
                with st.spinner("Starting resume processing..."):
                    # Initialize database
                    if not st.session_state.resumedb:
                        st.session_state.resumedb = SimpleResumeSearch(
                            storage_path="./chroma_db",
                            api_key=api_key
                        )
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = Path(temp_dir)
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, uploaded_file in enumerate(uploaded_files):
                            file_path = temp_path / uploaded_file.name
                            with open(file_path, 'wb') as f:
                                f.write(uploaded_file.getbuffer())
                            
                            status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                            try:
                                st.session_state.resumedb.add_resume(str(file_path))
                            except Exception as e:
                                st.warning(f"Error processing {uploaded_file.name}: {e}")
                            
                            progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        status_text.empty()
                        progress_bar.empty()
                    
                    st.success(f"Processed {len(uploaded_files)} resumes!")
                    st.balloons()

# Tab 2: Screen Candidates
with tab2:
    st.header("Screen Candidates")
    
    if not st.session_state.resumedb or st.session_state.resumedb.collection.count() == 0:
        st.warning("Please upload resumes first (Tab 1)")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            job_description = st.text_area(
                "Job Description",
                height=200,
                placeholder="""Role: Senior Software Engineer
Skills: Java, Spring Boot, AWS, Microservices
Experience: 5+ years
Location: Bangalore""",
                help="Describe the role, required skills, and experience"
            )
        
        with col2:
            st.subheader("Number of Candidates")
            top_n = st.slider("Show top N candidates", min_value=1, max_value=20, value=5)
        
        st.markdown("---")
        
        # Scoring weights
        with st.expander("Advanced: Adjust Scoring Weights"):
            st.write("Customize how candidates are scored:")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                skills_weight = st.slider("Skills", 0.0, 1.0, 0.4, 0.05)
            with col2:
                exp_weight = st.slider("Experience", 0.0, 1.0, 0.3, 0.05)
            with col3:
                cert_weight = st.slider("Certifications", 0.0, 1.0, 0.2, 0.05)
            with col4:
                title_weight = st.slider("Job Titles", 0.0, 1.0, 0.1, 0.05)
            
            total_weight = skills_weight + exp_weight + cert_weight + title_weight
            if abs(total_weight - 1.0) > 0.01:
                st.error(f"Weights must sum to 1.0 (current: {total_weight:.2f})")
        
        # Screen button
        if st.button("Screen Candidates", type="primary", use_container_width=True):
            if not job_description:
                st.error("Please enter job description")
            elif not api_key:
                st.error("Please enter OpenAI API key in sidebar")
            else:
                with st.spinner("Screening candidates... This may take 30-60 seconds"):
                    try:
                        weights = {
                            "skills": skills_weight,
                            "experience": exp_weight,
                            "certifications": cert_weight,
                            "job_titles": title_weight
                        }
                        
                        # Rank candidates
                        ranked = st.session_state.resumedb.rank_candidates_with_llm(
                            job_description=job_description,
                            requirements_weights=weights,
                            top_n=top_n
                        )
                        
                        # Extract contact info
                        enriched = st.session_state.resumedb.enrich_with_contact_info(ranked)
                        
                        st.session_state.ranked_results = enriched
                        st.success("Screening complete!")
                        st.info("View results in the 'Results' tab")
                        
                    except Exception as e:
                        st.error(f"Error: {e}")

# Tab 3: Results
with tab3:
    st.header("Screening Results")
    
    if not st.session_state.ranked_results:
        st.info("No results yet. Screen candidates in the previous tab.")
    else:
        results = st.session_state.ranked_results
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Candidates", len(results))
        with col2:
            avg_score = sum(r['final_score'] for r in results) / len(results)
            st.metric("Average Score", f"{avg_score:.2f}")
        with col3:
            top_score = results[0]['final_score'] if results else 0
            st.metric("Top Score", f"{top_score:.2f}")
        
        st.markdown("---")
        
        # Display each candidate
        for i, candidate in enumerate(results, 1):
            rank_label = "RANK 1" if i==1 else "RANK 2" if i==2 else "RANK 3" if i==3 else f"RANK {i}"
            
            with st.expander(
                f"{rank_label}: {candidate.get('name', candidate.get('filename', 'Unknown'))} - "
                f"Score: {candidate['final_score']:.2f}",
                expanded=(i <= 3)
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Details")
                    
                    # Show contact info
                    name = candidate.get('name', 'N/A')
                    email = candidate.get('email', 'N/A')
                    phone = candidate.get('phone', 'N/A')
                    
                    if name != 'N/A':
                        st.write(f"**Name:** {name}")
                    if email != 'N/A':
                        st.write(f"**Email:** {email}")
                    if phone != 'N/A':
                        st.write(f"**Phone:** {phone}")
                    
                    st.write(f"**Resume:** {candidate.get('filename', 'N/A')}")
                    
                    st.markdown("**Why this candidate?**")
                    justification = candidate.get('justification', 'Ranked based on resume match')
                    st.success(justification)
                
                with col2:
                    st.subheader("Scores")
                    
                    # Score breakdown
                    detailed = candidate.get('detailed_scores', {})
                    for criterion, score in detailed.items():
                        st.progress(score / 100, text=f"{criterion.title()}: {score}/100")
                    
                    st.markdown("---")
                    st.metric("Final Score", f"{candidate['final_score']:.2f}")
                
                # Download resume button
                filepath = candidate.get('filepath', '')
                if filepath and Path(filepath).exists():
                    with open(filepath, 'rb') as f:
                        st.download_button(
                            label="Download Resume",
                            data=f,
                            file_name=candidate.get('filename', 'resume.pdf'),
                            mime="application/pdf",
                            key=f"download_{i}"
                        )
        
        st.markdown("---")
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as CSV
            df = pd.DataFrame([{
                'Rank': i,
                'Name': r.get('name', 'N/A'),
                'Email': r.get('email', 'N/A'),
                'Phone': r.get('phone', 'N/A'),
                'Score': r['final_score'],
                'Skills Score': r.get('detailed_scores', {}).get('skills', 0),
                'Experience Score': r.get('detailed_scores', {}).get('experience', 0),
                'Filepath': r.get('filepath', '')
            } for i, r in enumerate(results, 1)])
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="Export as CSV",
                data=csv,
                file_name=f"screening_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Export as JSON
            json_data = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="Export as JSON",
                data=json_data,
                file_name=f"screening_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Built with Streamlit | AI-Powered Resume Screening
    </div>
    """,
    unsafe_allow_html=True
)