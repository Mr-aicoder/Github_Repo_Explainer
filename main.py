import streamlit as st 
import os 
import shutil 
from git import Repo, GitCommandError 
from dotenv import load_dotenv 
import json 
import time 
import stat 
import gc 
from urllib.parse import urlparse, urlunparse
from typing import TypedDict, Annotated, List, Dict

# Langchain & Langgraph imports
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") # Optional, for private repos

if not GROQ_API_KEY:
    st.error("Groq API Key not found. Please set it in the .env file. Example: GROQ_API_KEY=gsk_YOUR_KEY_HERE")
    st.stop()

# Initialize the Groq LLM
llm = ChatGroq(temperature=0.7, groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")

# --- Graph State Definition ---
# This TypedDict defines the shared state that will be passed between nodes in our graph.
class GraphState(TypedDict):
    repo_url: str
    github_token: str
    repo_dir: str
    readme_content: str
    requirements_content: str
    env_content: str
    repo_structure: str
    overall_explanation: str
    requirements_explanation: str
    env_explanation: str
    run_instructions: str
    generated_readme: str
    error_message: str # To capture errors and pass them through the graph

# --- Helper Functions (retained from previous version) ---
def get_repo_structure(repo_path, max_depth=3):
    """
    Generates a string representation of the repository's directory structure.
    Limits traversal to max_depth to prevent overly long outputs for the LLM.
    """
    structure_str = []
    for root, dirs, files in os.walk(repo_path):
        relative_path = os.path.relpath(root, repo_path)
        if relative_path == '.':
            depth = 0
        else:
            depth = relative_path.count(os.sep) + 1

        if depth > max_depth:
            del dirs[:] 
            continue

        indent = '    ' * depth
        if depth > 0:
            dir_name = os.path.basename(root)
            structure_str.append(f"{indent}📁 {dir_name}/")

        files.sort()
        dirs.sort()

        for f in files:
            if f.startswith('.') or f.endswith((".pyc", ".log", ".DS_Store", ".iml", ".class", ".swp", ".git")):
                continue
            structure_str.append(f"{indent}    📄 {f}")
            
    return "\n".join(structure_str)

def onerror(func, path, exc_info):
    """
    Error handler for shutil.rmtree.
    If the error is a PermissionError, it tries to change the file permissions
    and then retries the operation.
    """
    if issubclass(exc_info[0], PermissionError):
        st.warning(f"Permission error for '{path}'. Attempting to change permissions and retry.")
        try:
            os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO) # Equivalent to 0o777
            time.sleep(0.2) 
            func(path)
        except Exception as e:
            st.error(f"Failed to resolve permission issue for '{path}' even after chmod: {e}")
            raise exc_info[0].with_traceback(exc_info[1])
    else:
        raise exc_info[0].with_traceback(exc_info[1])

# --- Agent Nodes ---

def clone_repo_agent(state: GraphState):
    """
    Agent to clone the GitHub repository.
    """
    st.subheader("1. Cloning Repository...")
    repo_url = state['repo_url']
    github_token = state['github_token']
    repo_dir = state['repo_dir']

    # Initial cleanup before cloning
    if os.path.exists(repo_dir):
        st.info(f"Cleaning up previous clone in '{repo_dir}'...")
        max_retries = 5
        for i in range(max_retries):
            try:
                shutil.rmtree(repo_dir, onerror=onerror)
                st.success("Previous clone cleaned up successfully!")
                break
            except PermissionError as pe:
                if i < max_retries - 1:
                    st.warning(f"Permission error during cleanup: {pe}. Retrying in 2 seconds... ({i+1}/{max_retries})")
                    time.sleep(2)
                else:
                    st.error(f"Failed to clean up previous clone after {max_retries} retries. Please manually delete the '{repo_dir}' folder. Error: {pe}")
                    return {"error_message": f"Failed initial cleanup: {pe}"}
            except Exception as e:
                st.error(f"An unexpected error occurred during cleanup: {e}")
                return {"error_message": f"Unexpected cleanup error: {e}"}
    
    with st.spinner(f"Cloning '{repo_url}'... This might take a moment."):
        try:
            gc.collect() # Explicitly run garbage collection
            
            if github_token:
                parsed_url = urlparse(repo_url)
                if parsed_url.scheme == 'https':
                    auth_netloc = f"oauth2:{github_token}@{parsed_url.netloc}"
                    authenticated_url = urlunparse(parsed_url._replace(netloc=auth_netloc))
                elif parsed_url.scheme == 'git':
                     st.warning("GitHub token authentication primarily works for HTTPS URLs. If this is an SSH URL (git@github.com), ensure your SSH agent is configured or use an HTTPS URL.")
                     authenticated_url = repo_url
                else:
                    authenticated_url = repo_url
                
                Repo.clone_from(authenticated_url, repo_dir)
            else:
                Repo.clone_from(repo_url, repo_dir)
            
            st.success("Repository cloned successfully!")
            return state # Return the updated state (repo_dir is implicitly set by cloning)
        except GitCommandError as gce:
            st.error(f"Git Command Error: {gce}. This often means the repository URL is incorrect, "
                     f"the repository is private and the token is missing/invalid, or network issues.")
            st.code(gce.stderr)
            return {"error_message": f"Git cloning failed: {gce}"}
        except Exception as e:
            st.error(f"An unexpected error occurred during cloning: {e}")
            return {"error_message": f"Unexpected cloning error: {e}"}

def file_discovery_reader_agent(state: GraphState):
    """
    Agent to discover and read content of key files.
    """
    st.subheader("2. Analyzing Repository Contents...")
    repo_dir = state['repo_dir']
    
    readme_content = ""
    requirements_content = ""
    env_content = ""
    
    readme_path = os.path.join(repo_dir, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8", errors='ignore') as f:
            readme_content = f.read()
        st.info("README.md found.")
    else:
        st.warning("README.md not found.")

    requirements_path = os.path.join(repo_dir, "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8", errors='ignore') as f:
            requirements_content = f.read()
        st.info("requirements.txt found.")
    else:
        st.info("requirements.txt not found.")

    env_example_path = os.path.join(repo_dir, ".env.example")
    env_path = os.path.join(repo_dir, ".env")
    if os.path.exists(env_example_path):
        with open(env_example_path, "r", encoding="utf-8", errors='ignore') as f:
            env_content = f.read()
        st.info(".env.example found.")
    elif os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8", errors='ignore') as f:
            env_content = f.read()
        st.info(".env found.")
    else:
        st.info(".env or .env.example not found.")

    # Pass max_depth=3 to get_repo_structure for the prompt to reduce token count for large repos
    repo_structure = get_repo_structure(repo_dir, max_depth=3) 
    
    return {
        "readme_content": readme_content,
        "requirements_content": requirements_content,
        "env_content": env_content,
        "repo_structure": repo_structure
    }

def overall_explainer_agent(state: GraphState):
    """
    Agent to generate the overall explanation of the repository.
    """
    st.subheader("3. Generating AI Explanation (Overall)...")
    
    # Truncate README content for LLM input to save tokens
    truncated_readme = state['readme_content'][:4000] + "..." if len(state['readme_content']) > 4000 else state['readme_content']

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert software architect and technical explainer. Provide a comprehensive, clear explanation."),
            ("user", """
            Analyze the following GitHub repository information and provide a comprehensive, clear explanation.
            Focus on:
            1.  **Purpose:** What does this repository do? What problem does it solve?
            2.  **Key Technologies:** What programming languages, frameworks, or libraries are likely used? (Infer from file names, README, and common configuration files like package.json, requirements.txt, pom.xml, Dockerfile, etc.).
            3.  **Repository Overview:** Briefly describe the main folders and their likely contents based on the provided structure.

            ---
            **Repository URL:** {repo_url}

            **README.md Content (if available, truncated to important parts):**
            {readme_content}

            ---
            **Detailed Repository File and Directory Structure (up to 3 levels deep):**
            ```
            {repo_structure}
            ```
            ---

            Please provide your explanation in Markdown format, using clear headings and bullet points.
            Start directly with the explanation, no conversational filler.
            """)
        ]
    )
    chain = prompt_template | llm
    
    with st.spinner("AI is generating overall repository explanation..."):
        try:
            response = chain.invoke({
                "repo_url": state['repo_url'],
                "readme_content": truncated_readme, # Use truncated README
                "repo_structure": state['repo_structure']
            })
            overall_explanation = response.content
            st.markdown("#### Overall Repository Explanation:")
            st.markdown(overall_explanation)
            return {"overall_explanation": overall_explanation}
        except Exception as e:
            st.error(f"Error generating overall explanation: {e}")
            return {"error_message": f"Overall explanation failed: {e}"}

def requirements_explainer_agent(state: GraphState):
    """
    Agent to explain the requirements.txt file.
    """
    if not state['requirements_content']:
        st.info("Skipping requirements.txt explanation: No requirements.txt found.")
        return {"requirements_explanation": "N/A"} # Return N/A if no content

    st.subheader("4. Explaining `requirements.txt`...")
    
    # Truncate requirements content for LLM input
    truncated_requirements = state['requirements_content'][:2000] + "..." if len(state['requirements_content']) > 2000 else state['requirements_content']

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a software dependency expert. Explain the purpose of the following `requirements.txt` file content."),
            ("user", """
            Explain the purpose of the following `requirements.txt` file content.
            List each major dependency and briefly explain what it's used for in a typical Python project.

            ---
            **requirements.txt content:**
            ```
            {requirements_content}
            ```
            ---
            Provide your explanation in Markdown format.
            """)
        ]
    )
    chain = prompt_template | llm
    
    with st.spinner("AI is explaining requirements.txt..."):
        try:
            response = chain.invoke({"requirements_content": truncated_requirements}) # Use truncated content
            req_explanation = response.content
            st.markdown("#### `requirements.txt` Explanation:")
            st.markdown(req_explanation)
            return {"requirements_explanation": req_explanation}
        except Exception as e:
            st.error(f"Error explaining requirements.txt: {e}")
            return {"error_message": f"Requirements explanation failed: {e}"}

def env_explainer_agent(state: GraphState):
    """
    Agent to explain the .env or .env.example file.
    """
    if not state['env_content']:
        st.info("Skipping .env explanation: No .env or .env.example found.")
        return {"env_explanation": "N/A"} # Return N/A if no content

    st.subheader("5. Explaining `.env` / `.env.example`...")
    
    # Truncate env content for LLM input
    truncated_env = state['env_content'][:1000] + "..." if len(state['env_content']) > 1000 else state['env_content']

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a configuration expert. Explain the purpose of the following `.env` or `.env.example` file content."),
            ("user", """
            Explain the purpose of the following `.env` or `.env.example` file content.
            List each environment variable and briefly explain its likely purpose (e.g., API keys, database URLs, port numbers).

            ---
            **.env/.env.example content:**
            ```
            {env_content}
            ```
            ---
            Provide your explanation in Markdown format.
            """)
        ]
    )
    chain = prompt_template | llm
    
    with st.spinner("AI is explaining .env variables..."):
        try:
            response = chain.invoke({"env_content": truncated_env}) # Use truncated content
            env_explanation = response.content
            st.markdown("#### `.env` / `.env.example` Explanation:")
            st.markdown(env_explanation)
            return {"env_explanation": env_explanation}
        except Exception as e:
            st.error(f"Error explaining .env: {e}")
            return {"error_message": f"Env explanation failed: {e}"}

def run_instructions_agent(state: GraphState):
    """
    Agent to generate step-by-step run instructions.
    """
    st.subheader("6. Generating 'How to Run' Instructions...")
    
    # Truncate README content for LLM input
    truncated_readme = state['readme_content'][:4000] + "..." if len(state['readme_content']) > 4000 else state['readme_content']

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert DevOps engineer and project setup guide. Provide clear, step-by-step instructions."),
            ("user", """
            Based on the following repository information, provide clear, step-by-step instructions on how someone might set up and run this project locally.
            Consider common tools and commands for inferred technologies (e.g., Python, Node.js, Java, Docker).
            Look for clues in the README, file structure (e.g., presence of `Makefile`, `Dockerfile`, `package.json` scripts, `requirements.txt`, `setup.py`, `main.py`, `app.js`, `pom.xml`, etc.).
            If specific commands are hinted at (e.g., `npm install`, `pip install -r requirements.txt`, `docker-compose up`), include them.
            If no explicit instructions are found, provide general guidance based on inferred technologies.
            Include steps for:
            1.  Prerequisites (e.g., Python, Node.js, Docker installation).
            2.  Cloning the repository (already done by the user, but mention it as context).
            3.  Installing dependencies.
            4.  Environment variable setup.
            5.  Running the main application or key components.
            6.  Any common build steps.

            ---
            **Repository URL:** {repo_url}

            **README.md Content (if available, truncated to important parts):**
            {readme_content}

            ---
            **Detailed Repository File and Directory Structure (up to 3 levels deep):**
            ```
            {repo_structure}
            ```
            ---
            **Overall Explanation:**
            {overall_explanation}
            **Requirements Explanation:**
            {requirements_explanation}
            **Environment Variables Explanation:**
            {env_explanation}

            Please provide your instructions in Markdown format, using numbered lists for steps and code blocks for commands.
            Start directly with the instructions, no conversational filler.
            """)
        ]
    )
    chain = prompt_template | llm
    
    with st.spinner("AI is generating 'How to Run' instructions..."):
        try:
            response = chain.invoke({
                "repo_url": state['repo_url'],
                "readme_content": truncated_readme, # Use truncated README
                "repo_structure": state['repo_structure'],
                "overall_explanation": state['overall_explanation'],
                "requirements_explanation": state['requirements_explanation'],
                "env_explanation": state['env_explanation']
            })
            run_instructions = response.content
            st.markdown("#### How to Run/Use This Repository:")
            st.markdown(run_instructions)
            return {"run_instructions": run_instructions}
        except Exception as e:
            st.error(f"Error generating run instructions: {e}")
            return {"error_message": f"Run instructions failed: {e}"}

def readme_generator_agent(state: GraphState):
    """
    Agent to generate a new README.md content.
    """
    st.subheader("7. Generating New `README.md`...")
    
    # Truncate original README content for LLM input
    truncated_original_readme = state['readme_content'][:4000] + "..." if len(state['readme_content']) > 4000 else state['readme_content']

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert technical writer. Generate a comprehensive and well-structured `README.md` file."),
            ("user", """
            Based on the following information about a GitHub repository,
            generate a comprehensive and well-structured `README.md` file.
            Include the following sections:
            -   **Project Title** (e.g., "My Awesome Project")
            -   **Description** (What it does, its purpose)
            -   **Features** (Key functionalities)
            -   **Technologies Used**
            -   **Setup** (Prerequisites, Installation)
            -   **Usage** (How to run/interact with the project)
            -   **Contributing** (Simple placeholder)
            -   **License** (Simple placeholder)

            Use clear Markdown headings, bullet points, and code blocks where appropriate.
            ---
            **Repository URL:** {repo_url}
            **Original README.md (for inspiration, truncated):**
            {original_readme_content}
            **Overall Explanation:**
            {overall_explanation}
            **Requirements Explanation:**
            {requirements_explanation}
            **Environment Variables Explanation:**
            {env_explanation}
            **Run Instructions:**
            {run_instructions}
            ---
            Generate the complete Markdown content for the new README.md file.
            Start directly with the Markdown, no conversational filler.
            """)
        ]
    )
    chain = prompt_template | llm
    
    with st.spinner("AI is generating a new README.md..."):
        try:
            response = chain.invoke({
                "repo_url": state['repo_url'],
                "original_readme_content": truncated_original_readme, # Use truncated content
                "overall_explanation": state['overall_explanation'],
                "requirements_explanation": state['requirements_explanation'],
                "env_explanation": state['env_explanation'],
                "run_instructions": state['run_instructions']
            })
            generated_readme = response.content
            st.markdown("#### Generated `README.md`:")
            st.code(generated_readme, language="markdown")
            
            st.download_button(
                label="Download Generated README.md",
                data=generated_readme,
                file_name="GENERATED_README.md",
                mime="text/markdown"
            )
            return {"generated_readme": generated_readme}
        except Exception as e:
            st.error(f"Error generating README.md: {e}")
            return {"error_message": f"README generation failed: {e}"}

def cleanup_agent(state: GraphState):
    """
    Agent to clean up the cloned repository directory.
    """
    repo_dir = state['repo_dir']
    if os.path.exists(repo_dir):
        st.info(f"Attempting final cleanup of cloned repository '{repo_dir}'...")
        max_retries = 5
        for i in range(max_retries):
            try:
                shutil.rmtree(repo_dir, onerror=onerror)
                st.success("Final cleanup complete.")
                return {"error_message": ""} # Clear any previous error if cleanup succeeds
            except PermissionError as pe:
                if i < max_retries - 1:
                    st.warning(f"Permission error during final cleanup: {pe}. Retrying in 2 seconds... ({i+1}/{max_retries})")
                    time.sleep(2)
                else:
                    st.error(f"Failed to perform final cleanup after {max_retries} retries. Please manually delete the '{repo_dir}' folder. Error: {pe}")
                    return {"error_message": f"Final cleanup failed: {pe}"}
            except Exception as e:
                st.error(f"An unexpected error occurred during final cleanup: {e}")
                return {"error_message": f"Unexpected final cleanup error: {e}"}
    return state # Return state even if dir doesn't exist

# --- Define the Langgraph Workflow ---
workflow = StateGraph(GraphState)

# Add nodes for each agent
workflow.add_node("clone_repo", clone_repo_agent)
workflow.add_node("file_discovery_reader", file_discovery_reader_agent)
workflow.add_node("overall_explainer", overall_explainer_agent)
workflow.add_node("requirements_explainer", requirements_explainer_agent)
workflow.add_node("env_explainer", env_explainer_agent)
workflow.add_node("run_instructions", run_instructions_agent)
workflow.add_node("readme_generator", readme_generator_agent)
workflow.add_node("cleanup", cleanup_agent)

# Set entry point
workflow.set_entry_point("clone_repo")

# Define edges (transitions)
workflow.add_edge("clone_repo", "file_discovery_reader")
workflow.add_edge("file_discovery_reader", "overall_explainer") 
workflow.add_edge("overall_explainer", "requirements_explainer")
workflow.add_edge("requirements_explainer", "env_explainer")
workflow.add_edge("env_explainer", "run_instructions")
workflow.add_edge("run_instructions", "readme_generator")
workflow.add_edge("readme_generator", "cleanup")
workflow.add_edge("cleanup", END)

# Compile the graph
app_graph = workflow.compile()

# --- Streamlit UI Execution ---
st.title("🤖 Advanced GitHub Repository Explainer AI")

st.markdown("""
Enter a GitHub repository URL below. For **private repositories**, you'll also need to provide a GitHub Personal Access Token with `repo` scope.
""")

repo_url_input = st.text_input("GitHub Repository URL", "https://github.com/streamlit/streamlit-example-app")
github_token_input = st.text_input("GitHub Personal Access Token (Optional, for private repos)", type="password", value=GITHUB_TOKEN)

if st.button("Explain Repository"):
    if not repo_url_input:
        st.error("Please enter a GitHub repository URL.")
    else:
        # Initial state for the graph
        initial_state = GraphState(
            repo_url=repo_url_input,
            github_token=github_token_input,
            repo_dir="temp_cloned_repo", # Fixed directory name
            readme_content="",
            requirements_content="",
            env_content="",
            repo_structure="",
            overall_explanation="",
            requirements_explanation="",
            env_explanation="",
            run_instructions="",
            generated_readme="",
            error_message=""
        )
        
        st.write("---")
        st.write("### AI Agent Workflow:")

        # Run the graph
        final_state = None
        try:
            # Iterating through the graph steps to show progress in Streamlit
            for s in app_graph.stream(initial_state):
                for key, value in s.items():
                    # st.write(f"Executing: {key}") # Optional: show which node is running
                    if "error_message" in value and value["error_message"]:
                        st.error(f"Workflow stopped due to error: {value['error_message']}")
                        final_state = value
                        raise Exception(value["error_message"]) # Break the loop on error
                final_state = value # Keep track of the latest state

            if final_state and final_state.get("error_message"):
                st.error(f"Workflow completed with errors: {final_state['error_message']}")
            elif final_state:
                st.success("Repository explanation workflow completed successfully!")
            else:
                st.error("Workflow did not produce a final state.")

        except Exception as e:
            st.error(f"An error occurred during the graph execution: {e}")
            if final_state and final_state.get("error_message"):
                st.error(f"Last known error in state: {final_state['error_message']}")
            # Ensure cleanup is attempted even if graph fails early
            if os.path.exists(initial_state['repo_dir']):
                st.info(f"Attempting emergency cleanup of '{initial_state['repo_dir']}'...")
                try:
                    shutil.rmtree(initial_state['repo_dir'], onerror=onerror)
                    st.success("Emergency cleanup complete.")
                except Exception as cleanup_e:
                    st.error(f"Emergency cleanup failed: {cleanup_e}. Please manually delete '{initial_state['repo_dir']}'.")


st.sidebar.info("Developed by Sachin Tupsamundar.")

