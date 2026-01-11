import streamlit as st
import pandas as pd
import os
import time
import re
import sys
import gc
import mysql.connector
from mysql.connector import Error
import shutil # <--- REQUIRED for deleting DB folders
# --- Streamlit Context Imports (The Fix) ---
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
import threading # <--- REQUIRED
# --- LangChain & CrewAI Imports ---
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from crewai.tools import tool
from crewai import Agent, Task, Crew, Process
import stat #
import chromadb # Ensure this is imported
from chromadb.config import Settings # Import Settings
import tempfile # <--- ADDED for safe storage paths

# --- Page Configuration ---
st.set_page_config(page_title="AI Loan Processor", page_icon="🏦", layout="wide")
st.title("🧑‍💻 AI Agent Loan Assessment")

# --- 1. NEW: Custom CSS to Fix the "Big Text" Issue ---
def style_app():
    # This CSS targets the Streamlit code block to make it look like a real terminal
    st.markdown("""
    <style>
        /* Target the code block container */
        .stCodeBlock {
            background-color: #0e1117; /* Dark background matches Streamlit dark mode */
        }
        
        /* Target the code text itself */
        code {
            font-size: 10px !important; /* Smaller font */
            font-family: 'Courier New', Courier, monospace !important;
            line-height: 1.0 !important; /* Tighter spacing */
        }
        
        /* Optional: Hide the 'copy' button if it blocks text, or keep it */
        .stCopyButton {
            opacity: 0.5;
        }
    </style>
    """, unsafe_allow_html=True)

style_app() # <--- CALL THIS IMMEDIATELY

# st.title("🏦 AI Agent Loan Processing Unit")
st.markdown("""
This system uses a **Multi-Agent Crew** to orchestrate loan decisions. 
It retrieves mock data, consults PDF policy documents via RAG, and generates a final report.
""")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")
    openai_key = st.text_input("OpenAI API Key", type="password")
    st.info("Ensure you have Ollama running locally with `nomic-embed-text` for embeddings.")

    # Files Status Check (Dynamic)
    risk_exists = os.path.exists("Bank Loan Overall Risk Policy.pdf")
    interest_exists = os.path.exists("Bank Loan Interest Rate Policy.pdf")
    
    if risk_exists: st.success("✅ Risk Policy Loaded")
    else: st.error("❌ Risk Policy Missing")
        
    if interest_exists: st.success("✅ Interest Policy Loaded")
    else: st.error("❌ Interest Policy Missing")
    
# --- UTILITY: Context-Aware Output Redirector ---
class StreamToExpander:
    def __init__(self, expander):
        self.expander = expander
        self.buffer = []
        # 1. Capture the Streamlit Context from the main thread
        self.ctx = get_script_run_ctx()

    def write(self, data):
        # 2. If called from a background thread (CrewAI), attach the context
        if self.ctx:
            add_script_run_ctx(threading.current_thread(), self.ctx)
            
        # 3. Clean ANSI colors
        cleaned_data = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', data)
        
        # 4. Update the UI
        if cleaned_data.strip():
            self.buffer.append(cleaned_data)
            self.expander.code("".join(self.buffer), language='text')

    def flush(self):
        pass

@st.cache_resource
def get_chroma_client(db_dir):
    # This ensures only ONE client exists for this path across your app
    return chromadb.PersistentClient(path=db_dir)

# --- Helper: Save Uploaded File (Updates) ---
def handle_file_upload(uploaded_file, target_filename):
    """Saves file, deletes old DB, and triggers explicit rebuild."""
    try:
        # 1. Save PDF
        with open(target_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
            
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False


def remove_policy_file(target_filename, db_folder):
    # 1. Get the SAME client instance from the cache
    client = get_chroma_client(db_folder)
    
    # 2. Clear caches and reset (if allowed in your settings)
    client.clear_system_cache()
    
    # 3. Explicitly clear the Streamlit cache for this resource 
    # to prevent the app from trying to use a deleted folder
    st.cache_resource.clear()

    # 4. Force release of file handles
    del client
    gc.collect()

    # 5. Safe to delete now
    try:
        if os.path.exists(target_filename):
            os.remove(target_filename)
        if os.path.exists(db_folder):
            shutil.rmtree(db_folder)
        return True
    except PermissionError:
        st.error("File is still locked by another process. Try again in a few seconds.")
        return False

# --- Helper: Database Connection ---
def get_db_connection():
    """Establishes and returns a database connection."""
    try:
        db_host = os.getenv("DB_HOST", "localhost")
        # CONNECT TO YOUR DB
        conn = mysql.connector.connect(
            host=db_host,
            user="root",          # Your MySQL Username
            password="hoising123",# Your MySQL Password
            database="bank_loan_system"
        )
        return conn
    except Error as e:
        print(f"Connection failed: {e}")
        return None
    
# --- 1. Load Data from MySQL ---
def load_data_from_mysql():
    """Connects to MySQL and returns the 3 tables as DataFrames."""
    
    # FIX: Call the helper function to get the connection object
    conn = get_db_connection()
    
    # Check if connection was successful before proceeding
    if conn is None:
        return None, None, None, "Failed to connect to database."

    try:
        if conn.is_connected():
            # FETCH DATA
            # Updated queries to include Email
            q_credit = "SELECT customer_id as ID, name as Name, email as Email, credit_score as Credit_Score FROM credit_scores"
            q_account = "SELECT customer_id as ID, name as Name, nationality as Nationality, email as Email, account_status as Account_Status FROM account_statuses"
            q_pr = "SELECT customer_id as ID, name as Name, email as Email, pr_status as PR_Status FROM pr_statuses"
            
            # Read SQL directly into DataFrames
            c_df = pd.read_sql(q_credit, conn)
            a_df = pd.read_sql(q_account, conn)
            p_df = pd.read_sql(q_pr, conn)
            
            # Close connection after fetching data
            conn.close()
            
            # Return the 3 dataframes and a Success flag
            return c_df, a_df, p_df, True
            
    except Error as e:
        # If an error occurs during fetching, close connection if open
        if conn and conn.is_connected():
            conn.close()
        return None, None, None, str(e)
    
    # --- 2. Upload Data Function (NEW) ---
def upload_excel_to_mysql(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        
        # Standardize headers (strip whitespace, lowercase)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Check required columns
        required = {'customer_id', 'name', 'email', 'credit_score', 'nationality', 'account_status', 'pr_status'}
        if not required.issubset(df.columns):
            return False, f"Missing columns. Required: {required}"

        conn = get_db_connection()
        # --- SAFETY CHECK (ADD THIS) ---
        if conn is None:
            return False, "❌ Database Connection Failed. Check Docker logs."
        # -------------------------------
        cursor = conn.cursor()

        # 1. Update Credit Table
        # Syntax: INSERT ... ON DUPLICATE KEY UPDATE
        sql_credit = """
        INSERT INTO credit_scores (customer_id, name, email, credit_score)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE name=VALUES(name), email=VALUES(email), credit_score=VALUES(credit_score)
        """
        data_credit = df[['customer_id', 'name', 'email', 'credit_score']].values.tolist()
        cursor.executemany(sql_credit, data_credit)

        # 2. Update Account Table
        sql_account = """
        INSERT INTO account_statuses (customer_id, name, nationality, email, account_status)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE name=VALUES(name), nationality=VALUES(nationality), email=VALUES(email), account_status=VALUES(account_status)
        """
        data_account = df[['customer_id', 'name', 'nationality', 'email', 'account_status']].values.tolist()
        cursor.executemany(sql_account, data_account)

        # 3. Update PR Table
        sql_pr = """
        INSERT INTO pr_statuses (customer_id, name, email, pr_status)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE name=VALUES(name), email=VALUES(email), pr_status=VALUES(pr_status)
        """
        data_pr = df[['customer_id', 'name', 'email', 'pr_status']].values.tolist()
        cursor.executemany(sql_pr, data_pr)

        conn.commit()
        cursor.close()
        conn.close()
        return True, "Successfully uploaded and updated all tables."

    except Exception as e:
        return False, str(e)
    
    # --- 3. DELETE Customer Function (NEW) ---
def delete_customer(identifier):
    """
    Deletes a customer from ALL 3 tables.
    Accepts 'identifier' which can be a Customer ID OR a Name.
    """
    try:
        conn = get_db_connection()
        if conn is None: return False, "DB Connection Failed"
        cursor = conn.cursor()

        # --- STEP 1: Resolve the Identifier to a specific Customer ID ---
        target_id = None
        
        # A. First, check if the input matches a Customer ID exactly
        cursor.execute("SELECT customer_id FROM credit_scores WHERE customer_id = %s", (identifier,))
        id_match = cursor.fetchone()
        
        if id_match:
            # It was an ID
            target_id = id_match[0]
        else:
            # B. If not an ID, check if it matches a Name (Case-Insensitive)
            cursor.execute("SELECT customer_id FROM credit_scores WHERE LOWER(name) = LOWER(%s)", (identifier,))
            name_matches = cursor.fetchall()
            
            if len(name_matches) == 0:
                return False, f"No customer found with ID or Name: '{identifier}'"
            elif len(name_matches) > 1:
                # SAFETY LOCK: If multiple people have the same name, refuse to delete.
                return False, f"⚠️ Multiple customers found with the name '{identifier}'. Please delete using their unique Customer ID instead."
            else:
                # It was a unique Name
                target_id = name_matches[0][0]

        # --- STEP 2: Execute Deletion using the resolved Target ID ---
        # We now have the correct ID, so we can safely delete from all tables
        queries = [
            "DELETE FROM credit_scores WHERE customer_id = %s",
            "DELETE FROM account_statuses WHERE customer_id = %s",
            "DELETE FROM pr_statuses WHERE customer_id = %s"
        ]
        
        for q in queries:
            cursor.execute(q, (target_id,))
            
        conn.commit()
        cursor.close()
        conn.close()
        
        return True, f"Successfully deleted customer '{identifier}' (ID: {target_id})."

    except Exception as e:
        return False, str(e)

# Load data immediately so tools can access it in global scope
credit_df, account_df, pr_df, db_status = load_data_from_mysql()

# Display Databases in Expander
with st.expander("📊 View Mock Banking Systems (Live Database View)"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Credit System")
        st.dataframe(credit_df, hide_index=True)
    with col2:
        st.subheader("Account System")
        st.dataframe(account_df, hide_index=True)
    with col3:
        st.subheader("Govt PR System")
        st.dataframe(pr_df, hide_index=True)

# ==============================================================================
# 2. VECTOR DATABASE SETUP (FIXED FOR IMMEDIATE REBUILD)
# ==============================================================================
RISK_DB_DIR = "chroma_db_risk"
INTEREST_DB_DIR = "chroma_db_interest"

@st.cache_resource
def setup_vector_dbs():
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest", base_url="http://localhost:11434")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def get_or_create_vector_store(pdf_path, db_dir, collection_name):
        if not os.path.exists(pdf_path): return None
        
        # 1. Create Persistent Client
        client = chromadb.PersistentClient(path=db_dir)
        
        # 2. Check if collection exists and has data
        try:
            collection = client.get_collection(collection_name)
            if collection.count() > 0:
                print(f"Loading existing {collection_name}...")
                return Chroma(client=client, embedding_function=embeddings, collection_name=collection_name)
        except Exception:
            pass 

        # 3. Build if missing
        print(f"Building new vector store for {collection_name}...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splits = text_splitter.split_documents(docs)
        
        return Chroma.from_documents(
            client=client,
            documents=splits,
            embedding=embeddings,
            collection_name=collection_name,
        )

    risk_db = get_or_create_vector_store("Bank Loan Overall Risk Policy.pdf", RISK_DB_DIR, "risk_policy_collection")
    interest_db = get_or_create_vector_store("Bank Loan Interest Rate Policy.pdf", INTEREST_DB_DIR, "interest_rate_collection")
    
    return risk_db, interest_db

# --- 3. The EXPLICIT BUILDER (Call this on Upload) ---
def rebuild_vector_store(pdf_path, db_dir, collection_name, embeddings=None):
    """
    Forces a complete rebuild of the vector store.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest", base_url="http://localhost:11434")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    print(f"⚡ Starting manual build for {collection_name} in {db_dir}...")
    
    # 1. Force Clean Slate
    if os.path.exists(db_dir):
        try:
            shutil.rmtree(db_dir)
            time.sleep(1) 
        except Exception as e:
            st.error(f"Error deleting old DB: {e}")
            return None

    # 2. Load PDF
    if not os.path.exists(pdf_path):
        st.error(f"File {pdf_path} not found.")
        return None
        
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splits = text_splitter.split_documents(docs)
    
    # 3. Create Client & Build
    client = None
    try:
        client = chromadb.PersistentClient(path=db_dir) 
        db = Chroma.from_documents(
            client=client,
            documents=splits,
            embedding=embeddings,
            collection_name=collection_name,
        )
        print(f"✅ Build Complete for {collection_name}")
        return db
    except Exception as e:
        st.error(f"ChromaDB Build Error: {e}")
        return None
    finally:
        # CRITICAL: Release the client so setup_vector_dbs can pick it up on rerun
        if client:
            del client
            gc.collect()
    
# Initialize on app load
risk_db, interest_db = setup_vector_dbs()


# --- 3. Tool Definitions ---

@tool("Get Customer ID")
def get_customer_id(query: str):
    """
    Fetches the unique Customer ID and Name by searching for an ID, Name, or Email.
    """
    query_clean = query.strip().lower()
    
    # 1. Check ID (Exact match or partial match)
    mask_id = credit_df['ID'].astype(str).str.lower().str.contains(query_clean, na=False)
    
    # 2. Check Name
    mask_name = credit_df['Name'].str.lower().str.contains(query_clean, na=False)
    
    # 3. Check Email (if column exists)
    if 'Email' in credit_df.columns:
        mask_email = credit_df['Email'].str.lower().str.contains(query_clean, na=False)
        final_mask = mask_id | mask_name | mask_email
    else:
        final_mask = mask_id | mask_name

    results = credit_df[final_mask]

    if results.empty:
        return f"No customer found matching '{query}'."
    elif len(results) > 1:
        found_list = results['Name'].tolist()
        return f"Ambiguous search. Found multiple customers: {found_list}. Please be more specific."
    
    # Success
    found_row = results.iloc[0]
    # UPDATED RETURN: Returns both ID and Name
    return f"ID: {found_row['ID']}, Name: {found_row['Name']}"

@tool("Get Credit Score")
def get_credit_score(customer_id: str):
    """Fetches the credit score. Requires a Customer ID."""
    row = credit_df[credit_df['ID'] == str(customer_id)]
    if row.empty: return "Credit Score not found."
    return row.iloc[0]['Credit_Score']

@tool("Get Account Status")
def get_account_status(customer_id: str):
    """Fetches Account Status & Nationality. Requires a Customer ID."""
    row = account_df[account_df['ID'] == str(customer_id)]
    if row.empty: return "Account Status not found."
    return f"Status: {row.iloc[0]['Account_Status']}, Nationality: {row.iloc[0]['Nationality']}"

@tool("Get PR Status")
def get_pr_status(customer_id: str):
    """Fetches PR Status. Requires a Customer ID."""
    row = pr_df[pr_df['ID'] == str(customer_id)]
    if row.empty: return "PR Status not found."
    return row.iloc[0]['PR_Status']

@tool("Risk Policy Search")
def search_risk_policy(query: str):
    """Searches the 'Bank Loan Overall Risk Policy' PDF."""
    if risk_db is None: return "Error: Risk DB not initialized."
    results = risk_db.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in results])

@tool("Interest Rate Policy Search")
def search_interest_rate_policy(query: str):
    """Searches the 'Bank Loan Interest Rate Policy' PDF."""
    if interest_db is None: return "Error: Interest DB not initialized."
    results = interest_db.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in results])

# --- 4. Main Execution Logic ---

st.divider()
st.title("🏦 Banking System Dashboard")

# 1. Define the Tab Names
TABS = ["🚀 Loan Processor", "📂 Data Admin", "🧠 Knowledge Base (PDFs)"]

# 2. Initialize Session State for Tabs if not present
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = TABS[0]

# 3. Create a Custom Navigation Bar (Horizontal Radio)
# We use the session state to determine the 'index', ensuring it stays selected on rerun
selected_tab = st.radio(
    "Navigation", 
    TABS,
    index=TABS.index(st.session_state["active_tab"]),
    horizontal=True,
    label_visibility="collapsed",
    key="nav_radio"
)

# 4. Sync the Radio selection back to Session State
# (This handles when the user manually clicks a tab)
if st.session_state["active_tab"] != selected_tab:
    st.session_state["active_tab"] = selected_tab
    st.rerun() # Force a rerun to update the view immediately

# ==========================================
# VIEW 1: LOAN PROCESSOR
# ==========================================
if selected_tab == TABS[0]:
    col_input, col_btn = st.columns([4, 1], vertical_alignment="bottom")
    with col_input:
        input_name = st.text_input("Enter Applicant Name", placeholder="e.g., Loren, Matt, Andy")

    with col_btn:
        run_btn = st.button("🚀 Process Loan", type="primary", use_container_width=True)

    if run_btn:
        if not openai_key:
            st.error("Please enter your OpenAI API Key in the sidebar.")
        elif not (risk_exists and interest_exists): st.error("⚠️ One or more Policy PDFs are missing. Check 'Knowledge Base' tab.")
        else:
            os.environ["OPENAI_API_KEY"] = openai_key
            # Initialize LLM
            llm = ChatOpenAI(
                openai_api_key=openai_key,
                model_name="gpt-4",
                temperature=0.1
            )

            # Define Agents (Re-instantiated with the live LLM)
            router_agent = Agent(
                role='Loan Workflow Manager',
                goal='Orchestrate the end-to-end loan application process and ensure data integrity.',
                backstory=(
                    "You are the Lead Loan Officer."
                    "You manage the flow of information. You must validate the output of every specialist before proceeding."
                    "\n\n"
                    "YOUR PROCESS:"
                    "1. Ask Data Specialist for Profile."
                    "   -> **CIRCUIT BREAKER**: If Data Specialist returns 'STATUS: NOT_FOUND' or 'STATUS: DATA_MISSING', STOP. "
                    "      Reply: 'Error: Customer record could not be retrieved. Please verify the name/ID.'"
                    "\n\n"
                    "2. Check Eligibility (as per previous instructions)."
                    "   -> If 'INELIGIBLE':"
                    "     STOP the process. Do NOT contact Risk or Interest specialists."
                    "     **Draft a Final Rejection Report:**"
                    "     You must acknowledge the applicant's Credit Standing in your rejection."
                    "     Format: 'Although the applicant has a [Good/Bad] Credit Score of {insert_score}, we cannot recommend this loan because {insert_rejection_reason}.'"
                    "\n\n"
                    "3. Ask Risk Specialist for Risk Level."
                    "   -> **CIRCUIT BREAKER**: If Risk Specialist returns 'MANUAL_REVIEW_NEEDED', STOP."
                    "      Reply: 'Process Halted: Application requires manual underwriting review due to outlier credit score.'"
                    "\n\n"
                    "4. Ask Interest Specialist for Rate (Only if Risk Level is valid)."
                    "   -> **CIRCUIT BREAKER**: If Interest Specialist returns 'Undetermined', STOP."
                    "\n\n"
                    "5. Compile final report only if all steps succeeded."
                ),
                allow_delegation=True,
                llm=llm,
                verbose=True
            )

            data_agent = Agent(
                role='Data Retrieval Specialist',
                goal='Fetch and aggregate all necessary customer information using any available identifier.',
                backstory=(
                    "You are an expert Banking Database Analyst."
                    "You must execute the following workflow strictly in order:"
                    "\n\n"
                    "STEP 1: Use 'Get Customer ID' using the provided input."
                    "  - **CRITICAL FAILURE CHECK**: If the tool returns 'No customer found', STOP. Return 'STATUS: NOT_FOUND'."
                    "\n\n"
                    "STEP 2: Use the ID to fetch 'Credit Score' and 'Account Status' ONLY."
                    "  - If these fields are empty/null, Return 'STATUS: DATA_MISSING'."
                    " -If PR Status = 'nan', set PR Status as 'N/A'."
                    "\n\n"
                    "STEP 3: Check Nationality & PR Status (Only if Step 1 & 2 succeeded)."
                    "  - IF (Nationality == 'Singaporean': **STOP. Do NOT call 'Get PR Status'.**"
                    "  - IF (Nationality == 'Non-Singaporean' AND PR Status == False/0): Set Eligibility = 'INELIGIBLE'."
                    "  - OTHERWISE: Set Eligibility = 'ELIGIBLE'."
                    "\n\n"
                    "STEP 4: DETERMINE ELIGIBILITY & REASONING:"
                    "  - IF (Nationality == 'Non-Singaporean' AND PR Status == False):"
                    "      Status = 'INELIGIBLE'"
                    "      Reason = 'Applicant is Non-Singaporean and does not hold Permanent Residency.'"
                    "  - ELSE:"
                    "      Status = 'ELIGIBLE'"
                    "      Reason = 'N/A'"
                    "\n\n"
                    "OUTPUT REQUIREMENT:"
                    "You must return a structured summary including:"
                    "1. The Profile Data (Name, ID, Credit Score, Nationality, Account Status, PR Status(True/False))."
                    "2. The Eligibility Status (ELIGIBLE / INELIGIBLE)."
                    "3. The Rejection Reason (if any)."
                ),
                tools=[get_customer_id, get_credit_score, get_account_status, get_pr_status],
                llm=llm,
                verbose=True
            )

            # 3. Risk Assessment Specialist (Unchanged)
            risk_agent = Agent(
                role='Risk Assessment Specialist',
                goal='Determine the Risk Level (Low/Medium/High) based on the policy documents.',
                backstory=(
                    "You are a Risk Underwriter. You work with the 'Bank Loan Overall Risk Policy'."
                    "You will receive a Customer Profile containing a Credit Score and Account Status."
                    "Your sole responsibility is to search the Risk Policy to find which Risk Level corresponds to that specific Credit Score and Account Status."
                    "\n\n"
                    "FAILURE HANDLING:"
                    "If the search results do NOT contain a rule for the specific credit score provided (e.g., score is too low/high or policy is ambiguous), you MUST:"
                    "1. DO NOT GUESS."
                    "2. Return 'Risk Level: MANUAL_REVIEW_NEEDED'."
                    "3. State the reason (e.g., 'Credit score 350 is below policy threshold')."
                ),
                tools=[search_risk_policy],
                verbose=True,
                llm=llm
            )

            # 4. Interest Rate Specialist (Unchanged)
            interest_agent = Agent(
                role='Interest Rate Specialist',
                goal='Determine the applicable Interest Rate based on the Risk Level.',
                backstory=(
                    "You are a Pricing Analyst."
                    "You will receive a 'Risk Level' (e.g., Low, Medium, High) from the Risk Assessment Specialist."
                    "Search the Interest Rate Policy to find the specific interest rate percentage associated with that Risk Level."
                    "Return only the numerical rate and the justification."
                    "\n\n"
                    "FAILURE HANDLING:"
                    "If the search results do NOT contain a rule for the specific credit score provided (e.g., score is too low/high or policy is ambiguous), you MUST:"
                    "1. DO NOT GUESS."
                    "2. Return 'Risk Level: MANUAL_REVIEW_NEEDED'."
                    "3. State the reason (e.g., 'interest rate cannot be found for current credit score')."
                ),
                tools=[search_interest_rate_policy],
                verbose=True,
                llm=llm
            )

            # Define Task
            task = Task(
                description=(
                    f"Process the full loan application for: {input_name}. "
                    "\n\n"
                    "Execution Steps:"
                    "1. DATA: Retrieve profile and check Eligibility (Singaporean or PR). "
                    "2. DECISION: If Ineligible, terminate with a rejection notice. "
                    "3. RISK: ONLY if Eligible, proceed to determine Risk Level with credit score & account status. "
                    "4.PRICING: ONLY if Risk Level is valid, determine Interest Rate. "
                    "5. REPORT: Generate a final summary table. State the reason for the final decision."
                ),
                expected_output='A Markdown table containing: Customer ID, Name, Nationality, PR Status, Credit Score, Risk Level, Interest Rate, and Final Decision. Provide a sentence on the reason for the final decision.',
                agent=router_agent
            )

            # Define Crew
            loan_crew = Crew(
                agents=[router_agent, data_agent, risk_agent, interest_agent],
                tasks=[task],
                verbose=True
            )

            # --- PROCESS OUTPUT ---
            st.subheader("🤖 Agent Thinking Process")
            with st.status("Agents are working...", expanded=True) as status:
                log_container = st.empty()
                sys_out_redirector = StreamToExpander(log_container)
                
                # Redirect stdout
                original_stdout = sys.stdout
                sys.stdout = sys_out_redirector
                
                try:
                    result = loan_crew.kickoff()
                    status.update(label="✅ Complete!", state="complete", expanded=False)
                except Exception as e:
                    result = f"Error: {e}"
                    status.update(label="❌ Error", state="error")
                
                # Restore stdout
                sys.stdout = original_stdout

            st.divider()
            st.subheader("📝 Final Decision Report")
            st.markdown(result)
# ==========================================
# VIEW 2: DATA ADMIN
# ==========================================
elif selected_tab == TABS[1]:
    st.subheader("Bulk Upload Customer Data")
    st.markdown("Upload an Excel file to update the database. Columns required: `Customer ID`, `Name`, `Email`, `Credit Score`, `Nationality`, `Account Status`, `PR Status`.")
    
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])
    
    if uploaded_file is not None:
        if st.button("Upload to Database"):
            with st.spinner("Uploading and updating tables..."):
                success, message = upload_excel_to_mysql(uploaded_file)
                if success:
                    st.success(message)
                    time.sleep(1)
                    # Force stay on this tab
                    st.session_state["active_tab"] = TABS[1] 
                    st.rerun() 
                else:
                    st.error(f"Upload Failed: {message}")

    st.info("💡 Note: Duplicate Customer IDs will automatically UPDATE existing records (Upsert).")

    # 2. DELETE Section 
    st.subheader("🗑️ Delete Customer")
    st.markdown("⚠️ **Warning:** This action will remove the customer from **ALL** database tables.")
    
    col_del_input, col_del_btn = st.columns([1, 1], vertical_alignment="bottom")
    
    input_identifier = col_del_input.text_input("Enter Customer ID or Name", placeholder="e.g. 1111 or Loren")
    
    if col_del_btn.button("Delete Customer", type="primary"):
        if not input_identifier:
            st.warning("Please enter an ID or Name.")
        else:
            with st.spinner(f"Searching and Deleting '{input_identifier}'..."):
                success, msg = delete_customer(input_identifier)
                if success:
                    st.success(msg)
                    time.sleep(2)
                    # Force stay on this tab
                    st.session_state["active_tab"] = TABS[1]
                    st.rerun() 
                else:
                    st.error(f"Deletion Failed: {msg}")

# ==========================================
# VIEW 3: KNOWLEDGE BASE
# ==========================================
elif selected_tab == TABS[2]:
    st.subheader("📚 Policy Document Management")
    st.markdown("Vectors are stored in `chroma_db_risk` and `chroma_db_interest`.")
    col_risk, col_interest = st.columns(2)
    
    # RISK POLICY
    with col_risk:
        st.info("🛡️ **Risk Policy Database**")
        if risk_exists:
            st.success("✅ Active (Vectors Loaded)")
            if st.button("🗑️ Remove Risk Policy", type="secondary"):
                remove_policy_file("Bank Loan Overall Risk Policy.pdf", RISK_DB_DIR)
                # CRITICAL: Tell Streamlit to stay on this tab before reloading
                st.session_state["active_tab"] = TABS[2]
                st.rerun()
        else: st.error("❌ Missing")
        
        risk_up = st.file_uploader("Upload New Risk Policy", type=['pdf'], key="rk")
        
        if risk_up and st.button("💾 Save & Build Risk DB"):
            if handle_file_upload(risk_up, "Bank Loan Overall Risk Policy.pdf"):
                with st.spinner("Generating Embeddings & Building Vector Store..."):
                    rebuild_vector_store("Bank Loan Overall Risk Policy.pdf", RISK_DB_DIR, "risk_policy_collection") 
                
                st.success("Database Rebuilt!")
                st.cache_resource.clear()
                time.sleep(1)
                
                # CRITICAL: Tell Streamlit to stay on this tab before reloading
                st.session_state["active_tab"] = TABS[2]
                st.rerun()

    # INTEREST POLICY
    with col_interest:
        st.info("📈 **Interest Rate Policy Database**")
        if interest_exists:
            st.success("✅ Active (Vectors Loaded)")
            if st.button("🗑️ Remove Interest Policy", type="secondary"):
                remove_policy_file("Bank Loan Interest Rate Policy.pdf", INTEREST_DB_DIR)
                # CRITICAL: Tell Streamlit to stay on this tab before reloading
                st.session_state["active_tab"] = TABS[2]
                st.rerun()
        else: st.error("❌ Missing")
        
        int_up = st.file_uploader("Upload New Interest Policy", type=['pdf'], key="int")
        
        if int_up and st.button("💾 Save & Build Interest DB"):
            if handle_file_upload(int_up, "Bank Loan Interest Rate Policy.pdf"):
                with st.spinner("Generating Embeddings & Building Vector Store..."):
                    rebuild_vector_store("Bank Loan Interest Rate Policy.pdf", INTEREST_DB_DIR, "interest_rate_collection")
                
                st.success("Database Rebuilt!")
                st.cache_resource.clear()
                time.sleep(1)
                
                # CRITICAL: Tell Streamlit to stay on this tab before reloading
                st.session_state["active_tab"] = TABS[2]
                st.rerun()