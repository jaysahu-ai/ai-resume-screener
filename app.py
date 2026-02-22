# ========== app.py ==========

import streamlit as st
import os
from pathlib import Path
import tempfile
import pandas as pd
from datetime import datetime
import json

# Your existing classes
from resume_search import SimpleResumeSearch #, extract_contact_info_from_resume

# Page config
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

# Title
st.title("🎯 AI-Powered Resume Screening System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key
    api_key = st.text_input(
        "OpenAI API Key", 
        type="password",
        value=os.environ.get('OPENAI_API_KEY', ''),
        help="Enter your OpenAI API key"
    )
    
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
    
    st.markdown("---")
    
    # Database info
    st.subheader("📊 Database Status")
    if st.session_state.resumedb:
        count = st.session_state.resumedb.collection.count()
        st.success(f"✓ {count} resumes loaded")
    else:
        st.info("No database loaded")
    
    # Clear database button
    if st.button("🗑️ Clear Database"):
        if st.session_state.resumedb:
            st.session_state.resumedb.clear_database()
            st.success("Database cleared!")
            st.rerun()

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload Resumes", "🔍 Screen Candidates", "📊 Results"])

# Tab 1: Upload Resumes
with tab1:
    st.header("Upload Resumes")
    
    uploaded_files = st.file_uploader(
        "Choose PDF or DOCX files",
        type=['pdf', 'docx'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} files selected")
        
        if st.button("🚀 Process Resumes", type="primary"):
            if not api_key:
                st.error("❌ Please enter OpenAI API key in sidebar")
            else:
                # Initialize database
                if not st.session_state.resumedb:
                    st.session_state.resumedb = SimpleResumeSearch(
                        storage_path="./chroma_db",
                        api_key=api_key
                    )
                
                # Create temp directory for uploads
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    # Save uploaded files
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Save file
                        file_path = temp_path / uploaded_file.name
                        with open(file_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process
                        status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                        try:
                            st.session_state.resumedb.add_resume(str(file_path))
                        except Exception as e:
                            st.warning(f"⚠️ Error processing {uploaded_file.name}: {e}")
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    status_text.empty()
                    progress_bar.empty()
                
                st.success(f"✅ Processed {len(uploaded_files)} resumes!")
                st.balloons()

# Tab 2: Screen Candidates
with tab2:
    st.header("Screen Candidates")
    
    if not st.session_state.resumedb or st.session_state.resumedb.collection.count() == 0:
        st.warning("⚠️ Please upload resumes first (Tab 1)")
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
            st.subheader("Budget Range (LPA)")
            min_ctc = st.number_input("Minimum CTC", min_value=0, value=15, step=1)
            max_ctc = st.number_input("Maximum CTC", min_value=0, value=28, step=1)
            
            st.subheader("Number of Candidates")
            top_n = st.slider("Show top N candidates", min_value=1, max_value=20, value=5)
        
        st.markdown("---")
        
        # Scoring weights
        with st.expander("⚙️ Advanced: Adjust Scoring Weights"):
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
                st.error(f"⚠️ Weights must sum to 1.0 (current: {total_weight:.2f})")
        
        # Screen button
        if st.button("🎯 Screen Candidates", type="primary", use_container_width=True):
            if not job_description:
                st.error("❌ Please enter job description")
            elif not api_key:
                st.error("❌ Please enter OpenAI API key in sidebar")
            else:
                with st.spinner("🔍 Screening candidates... This may take 30-60 seconds"):
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
                        # enriched = []
                        # for candidate in ranked:
                        #     enriched_candidate = extract_contact_info_from_resume(
                        #         st.session_state.resumedb, 
                        #         candidate
                        #     )
                        #     enriched.append(enriched_candidate)
                        
                        # st.session_state.ranked_results = enriched
                        st.session_state.ranked_results = ranked
                        st.success("✅ Screening complete!")
                        st.info("👉 View results in the 'Results' tab")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

# Tab 3: Results
with tab3:
    st.header("📊 Screening Results")
    
    if not st.session_state.ranked_results:
        st.info("ℹ️ No results yet. Screen candidates in the previous tab.")
    else:
        results = st.session_state.ranked_results
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Candidates", len(results))
        with col2:
            avg_score = sum(r['final_score'] for r in results) / len(results)
            st.metric("Average Score", f"{avg_score:.2f}")
        with col3:
            top_score = results[0]['final_score'] if results else 0
            st.metric("Top Score", f"{top_score:.2f}")
        with col4:
            st.metric("Budget Matches", "TBD")
        
        st.markdown("---")
        
        # Display each candidate
        for i, candidate in enumerate(results, 1):
            with st.expander(
                f"{'🥇' if i==1 else '🥈' if i==2 else '🥉' if i==3 else '👤'} "
                f"Rank #{i}: {candidate.get('name', 'Unknown')} - "
                f"Score: {candidate['final_score']:.2f}",
                expanded=(i <= 3)
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📋 Details")
                    st.write(f"**Name:** {candidate.get('name', 'N/A')}")
                    st.write(f"**Email:** {candidate.get('email', 'N/A')}")
                    st.write(f"**Phone:** {candidate.get('phone', 'N/A')}")
                    st.write(f"**Current Role:** {candidate.get('current_role', 'N/A')}")
                    
                    st.markdown("**Justification:**")
                    st.info(candidate.get('justification', 'No justification available'))
                
                with col2:
                    st.subheader("📊 Scores")
                    
                    # Score breakdown
                    detailed = candidate.get('detailed_scores', {})
                    for criterion, score in detailed.items():
                        st.progress(score / 100, text=f"{criterion.title()}: {score}/100")
                    
                    st.markdown("---")
                    st.metric("Final Score", f"{candidate['final_score']:.2f}")
                
                # Download resume button
                if Path(candidate['filepath']).exists():
                    with open(candidate['filepath'], 'rb') as f:
                        st.download_button(
                            label="📄 Download Resume",
                            data=f,
                            file_name=candidate['filename'],
                            mime="application/pdf"
                        )
        
        st.markdown("---")
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as CSV
            df = pd.DataFrame([{
                'Rank': i,
                'Name': r.get('name', 'Unknown'),
                'Email': r.get('email', 'N/A'),
                'Phone': r.get('phone', 'N/A'),
                'Score': r['final_score'],
                'Skills Score': r['detailed_scores'].get('skills', 0),
                'Experience Score': r['detailed_scores'].get('experience', 0),
                'Filepath': r['filepath']
            } for i, r in enumerate(results, 1)])
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Export as CSV",
                data=csv,
                file_name=f"screening_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Export as JSON
            json_data = json.dumps(results, indent=2)
            st.download_button(
                label="📥 Export as JSON",
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
