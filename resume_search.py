import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
import docx
from pathlib import Path
from openai import OpenAI
import os, dotenv, json
dotenv.load_dotenv()

class SimpleResumeSearch:
    def __init__(self,storage_path="./chroma_db", api_key=None):
        self.client = chromadb.PersistentClient(path=storage_path)
        self.api_key = api_key

        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = self.client.get_or_create_collection(
            name="resumes",
            embedding_function=sentence_transformer_ef,
            metadata={"description": "Resume database with LLM extracted key info"}
        )

        print(f"Database ready at {storage_path}")
        print(f"Currently storing {self.collection.count()} resumes")
    
    def extract_text_from_pdf(self,pdf_path):
        text = "" 
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()

    def extract_text_from_docx(self,docx_path):
        doc = docx.Document(docx_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    
    def extract_text(self, file_path):
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension == '.pdf':
            return self.extract_text_from_pdf(str(file_path))
        elif extension in ['.docx','.doc']:
            return self.extract_text_from_docx(str(file_path))
        else:
            with open(file_path,'r',encoding='utf-8') as f:
                return f.read()
    
    def extract_key_info_with_llm(self, resume_text):
                
        client = OpenAI(api_key=self.api_key)
        
        prompt = f"""Extract only the key searchable information from this resume. 
                Focus on: skills, technologies, job titles, companies, years of experience, education, certifications.

                Remove: addresses, references, objectives, filler words, hobbies, personal statements.

                Create a condensed summary (max 400 words) containing:
                - All technical skills, tools, and technologies
                - Job titles with companies and duration
                - Total years of experience
                - Education (degrees, institutions, graduation years)
                - Certifications and licenses
                - Key quantified achievements

                Format as natural text optimized for semantic search.

                Resume text:
                {resume_text[:3500]}

                Provide the extracted key information:"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0
            )
            
            extracted_info = response.choices[0].message.content.strip()
            return extracted_info
            
        except Exception as e:
            print(f"\n LLM extraction failed: {e}")
            print(f" Falling back to first 1500 chars of resume")
            # Fallback: use beginning of resume if LLM fails
            return resume_text[:1500]
            
    def add_resume(self, file_path):
        file_path = Path(file_path)
        print(f"Adding: {file_path.name}")
        
        resume_text = self.extract_text(file_path)
        searchable_text = self.extract_key_info_with_llm(resume_text)
        
        self.collection.add(
            documents=[searchable_text],  # The resume text
            ids=[file_path.stem],      # Unique ID (filename without extension)
            metadatas=[{               # Extra info for filtering
                "filename": file_path.name,
                "filepath": str(file_path.absolute()),
                "full_text": resume_text
            }]
        )
        
        print(f"Added: {file_path.name}")


    def add_all_resumes(self, resume_folder):
        resume_folder = Path(resume_folder)
        # Find all resume files
        resume_files = list(resume_folder.glob("*.pdf"))
        resume_files += list(resume_folder.glob("*.docx"))
        resume_files += list(resume_folder.glob("*.doc"))
        resume_files += list(resume_folder.glob("*.txt"))
        
        print(f"\nFound {len(resume_files)} resume files")
        print("Adding to database...\n")
        
        for i, resume_file in enumerate(resume_files, 1):
            print(f"[{i}/{len(resume_files)}] ", end="")
            try:
                self.add_resume(resume_file)
            except Exception as e:
                print(f" Error with {resume_file.name}: {e}")
        
        print(f"\n Database now contains {self.collection.count()} resumes")


    def search(self, job_description, top_n=5):
        print(f"\nSearching for top {top_n} candidates...")

        results = self.collection.query(
            query_texts=[job_description],
            n_results=top_n
        )
        
        return results
    
    def print_results(self, results):
        if not results['documents'][0]:
            print("No results found!")
            return
        
        print("\n" + "="*70)
        print("TOP CANDIDATES")
        print("="*70)
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            similarity_score = 1 - distance  # Convert distance to similarity
            
            print(f"\n#{i} - {metadata['filename']}")
            print(f"Similarity Score: {similarity_score:.2%}")
            print(f"File: {metadata['filepath']}")
            print(f"\nResume Preview (first 300 chars):")
            print("-" * 70)
            print(doc[:300] + "...")
            print("-" * 70)

    def clear_database(self):
        all_ids = self.collection.get()['ids']
    
        if not all_ids:
            print("Database is already empty!")
            return
    
        print(f"Deleting {len(all_ids)} resumes...")
        self.collection.delete(ids=all_ids)
        print("Database cleared!")
    
    def rank_candidates_with_llm(self, job_description, requirements_weights=None,top_n=5):
        print(f"Stage 1: Finding candidates with vector search...")

        initial_results = self.collection.query(
                query_texts=[job_description],
                n_results=min(top_n * 2, self.collection.count())
            )
        if not initial_results['documents'][0]:
            print("No candidates found!")
            return []
        
        num_candidates = len(initial_results['documents'][0])
        print(f"Found {num_candidates} candidates.")
    
        print(f"\nStage 2: Scoring with LLM using weighted criteria...\n")
    
        candidates_for_scoring=[]
    
        for i, (doc, metadata, resume_id) in enumerate(zip(
            initial_results['documents'][0],
            initial_results['metadatas'][0],
            initial_results['ids'][0]
            ),1):
            candidates_for_scoring.append({
                "candidate_id": resume_id,
                "candidate_number": i,
                "resume_content": doc[:1000]
            })
        
        # requirements_weights = {
        #         "skills": 0.4,
        #         "experience": 0.3,
        #         "certifications": 0.2,
        #         "job_titles": 0.1
        #     }
        if requirements_weights is None:
            requirements_weights = {
                "skills": 0.4,
                "experience": 0.3,
                "certifications": 0.2,
                "job_titles": 0.1
            }
    
        weights_text = "\n".join([f"- {k}: {v*100}%" for k, v in requirements_weights.items()])
    
        prompt = f"""You are a resume screening expert. Score each candidate against the job requirements.

    JOB DESCRIPTION:
    {job_description}   

    CANDIDATES TO SCORE:
    {json.dumps(candidates_for_scoring, indent=2)}  

    SCORING INSTRUCTIONS:   

    1. For EACH criterion, score the candidate from 0-100:
       - 0 = No match at all
       - 50 = Partial match
       - 100 = Perfect match    

    2. Apply these weights to calculate final score:
    {weights_text}  

    3. Formula: final_score = (skills_score * 0.4) + (experience_score * 0.3) + (certifications_score * 0.2) + (job_titles_score * 0.1)
       This gives a score from 0-100.   

    4. Then NORMALIZE to 0-1 scale by dividing by 100.  

    CRITICAL: Return ONLY valid JSON in this EXACT format (no markdown, no explanation):
    {{
      "ranked_candidates": [
        {{
          "candidate_id": "resume_id_here",
          "candidate_number": 1,
          "scores": {{
            "skills": 85,
            "experience": 70,
            "certifications": 60,
            "job_titles": 90
          }},
          "weighted_score": 0.76,
          "rank": 1,
          "brief_justification": "One sentence why this candidate ranked here"
        }}
      ]
    }}  

    Return candidates sorted by weighted_score (highest first)."""
        try:
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model = "gpt-4o-mini",
                messages = [{"role":"user","content":prompt}],
                max_tokens=1500,
                temperature=0    
            )   

            response_text = response.choices[0].message.content.strip() 

            result = json.loads(response_text)  

            ranked_candidates = result.get('ranked_candidates',[])  

            final_results = []
            for candidate in ranked_candidates[:top_n]:
                candidate_id = candidate['candidate_id']

                # Find the corresponding metadata
                idx = initial_results['ids'][0].index(candidate_id)
                metadata = initial_results['metadatas'][0][idx]

                final_results.append({
                    "resume_id": candidate_id,
                    "filename": metadata['filename'],
                    "filepath": metadata['filepath'],
                    "rank": candidate['rank'],
                    "final_score": candidate['weighted_score'],
                    "detailed_scores": candidate['scores'],
                    "justification": candidate['brief_justification']
                })

            print(f" Ranked {len(final_results)} candidates")
            return final_results
        
        except Exception as e:
            print(f"Error during LLM ranking: {e}")
            return []
    
    def print_ranked_results(self, ranked_candidates):
        """
        Pretty print LLM-ranked candidates

        Args:
            ranked_candidates: Output from rank_candidates_with_llm()
        """
        if not ranked_candidates:
            print("No ranked candidates to display!")
            return

        print("\n" + "="*80)
        print("TOP RANKED CANDIDATES (LLM-Scored with Weighted Criteria)")
        print("="*80)

        for candidate in ranked_candidates:
            print(f"\n{'#'*80}")
            print(f"RANK #{candidate['rank']}: {candidate['filename']}")
            print(f"{'#'*80}")
            print(f"Name: {candidate.get('name', 'N/A')}")
            print(f"Email: {candidate.get('email', 'N/A')}")
            print(f"Phone: {candidate.get('phone', 'N/A')}")
            print(f"Final Score: {candidate['final_score']:.2f} (out of 1.00)")
            print(f"Resume ID: {candidate['resume_id']}")
            print(f"File Path: {candidate['filepath']}")
            print(f"\nDetailed Scores (out of 100):")
            for criterion, score in candidate['detailed_scores'].items():
                bar_length = int(score / 5)  # Scale to 20 chars
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"  {criterion:15s} [{bar}] {score}/100")

            print(f"\nWhy this rank:")
            print(f"  {candidate['justification']}")
            print("-" * 80)

        print(f"\n{'='*80}")
        print(f"Total candidates ranked: {len(ranked_candidates)}")
        print(f"{'='*80}\n")


# ========== Simple version - just use filename ==========

    def enrich_with_contact_info(self, ranked_candidates):
        """
        Extract contact information for ranked candidates
        
        Args:
            ranked_candidates: List from rank_candidates_with_llm()
            
        Returns:
            List of candidates enriched with contact info (name, email, phone)
        """
        print(f"\nExtracting contact information for {len(ranked_candidates)} candidates...")
        
        enriched_candidates = []
        
        for i, candidate in enumerate(ranked_candidates, 1):
            print(f"[{i}/{len(ranked_candidates)}] Processing {candidate['filename']}...", end=" ")
            
            # Get full resume text
            filepath = candidate['filepath']
            try:
                full_text = self.extract_text(filepath)
            except Exception as e:
                print(f"Error reading file: {e}")
                candidate['name'] = 'N/A'
                candidate['email'] = 'N/A'
                candidate['phone'] = 'N/A'
                enriched_candidates.append(candidate)
                continue
            
            # Extract contact info with LLM
            prompt = f"""Extract contact information from this resume.
    
    Resume text:
    {full_text[:2000]}
    
    Return ONLY valid JSON:
    {{
      "name": "Full Name",
      "email": "email@example.com or null",
      "phone": "+919876543210 or null"
    }}
    """
            
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key)
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0
                )
                
                result_text = response.choices[0].message.content.strip()
                
                # Clean JSON
                import re
                if result_text.startswith('```'):
                    result_text = re.sub(r'^```json?\s*', '', result_text)
                    result_text = re.sub(r'\s*```$', '', result_text)
                
                import json
                contact_info = json.loads(result_text)
                
                # Add to candidate
                candidate['name'] = contact_info.get('name') or 'N/A'
                candidate['email'] = contact_info.get('email') or 'N/A'
                candidate['phone'] = contact_info.get('phone') or 'N/A'
                
                print(f"✓ {candidate['name']}")
                
            except Exception as e:
                print(f"✗ Extraction failed: {e}")
                candidate['name'] = 'N/A'
                candidate['email'] = 'N/A'
                candidate['phone'] = 'N/A'
            
            enriched_candidates.append(candidate)
        
        print(f"\n✓ Contact extraction complete\n")
        return enriched_candidates