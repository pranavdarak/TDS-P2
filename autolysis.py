# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "seaborn",
#   "pandas",
#   "matplotlib",
#   "httpx",
#   "chardet",
#   "ipykernel",
#   "openai",
#   "numpy",
#   "scipy",
#   "python-dotenv",
# ]
# ///

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API configurations
API_BASE = "https://aiproxy.sanand.workers.dev/openai/v1"
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
MODEL = "gpt-4o-mini"

# Configure OpenAI API client
openai.api_base = API_BASE
openai.api_key = AIPROXY_TOKEN

# Validate OpenAI API Key
if not openai.api_key:
    print("Error: AIPROXY_TOKEN is not set. Ensure the token is loaded in the environment.")
    sys.exit(1)

# Function to call the LLM
def get_llm_response(prompt):
    try:
        print("Sending request to LLM...")
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an AI data analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800
        )
        print("LLM Raw Response:", response)
        if "choices" in response and len(response.choices) > 0:
            return response.choices[0].message["content"].strip()
        else:
            return "No analysis provided by LLM."
    except Exception as e:
        print(f"Error with LLM: {e}")
        return "LLM analysis failed."

# Validate command-line arguments
def validate_args():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

# Load the dataset
def load_dataset(csv_file):
    if not os.path.exists(csv_file):
        print(f"File {csv_file} not found.")
        sys.exit(1)

    try:
        data = pd.read_csv(csv_file, encoding='ISO-8859-1')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    return data

# Generate summary statistics
def generate_summary(data):
    summary_stats = data.describe(include="all").transpose()
    missing_values = data.isnull().sum()
    correlation_matrix = data.corr(numeric_only=True)
    return summary_stats, missing_values, correlation_matrix

# Save visualizations
def save_visualizations(data, correlation_matrix, output_dir):
    # Correlation matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.close()

    # Generate distribution plots for numerical columns
    for column in data.select_dtypes(include=[np.number]).columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[column].dropna(), kde=True, bins=30, color="blue")
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, f"{column}_distribution.png"))
        plt.close()

# Prepare prompt for LLM based on dataset analysis
def prepare_llm_prompt(data, missing_values):
    sample_data = data.head(5).to_dict(orient="records")
    llm_prompt = f"""
    You are an AI data analyst. Here's the dataset summary:
    - Number of Rows: {data.shape[0]}
    - Number of Columns: {data.shape[1]}
    - Columns: {list(data.columns)}
    - Data Types: {data.dtypes.to_dict()}
    - Missing Values: {missing_values.to_dict()}
    - Sample Data: {sample_data}

    Your task:
    1. Analyze the dataset.
    2. Highlight important trends, outliers, and correlations.
    3. Suggest potential applications or interpretations of the data.
    4. Provide actionable insights.
    """
    return llm_prompt

# Generate README.md content
def generate_readme(data, missing_values, llm_analysis, output_dir):
    markdown_content = f"""
    # Automated Analysis Report

    ## Dataset Overview
    - **Rows**: {data.shape[0]}
    - **Columns**: {data.shape[1]}
    - **Missing Values**:
    {missing_values.to_string()}

    ## Key Insights
    {llm_analysis}

    ## Visualizations
    ### Correlation Matrix
    ![Correlation Matrix](correlation_matrix.png)

    ### Distributions
    """
    for column in data.select_dtypes(include=[np.number]).columns:
        markdown_content += f"![{column} Distribution]({column}_distribution.png)\n"

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(markdown_content)

# Main function to orchestrate the tasks
def main():
    validate_args()
    csv_file = sys.argv[1]
    data = load_dataset(csv_file)
    summary_stats, missing_values, correlation_matrix = generate_summary(data)

    dataset_name = os.path.splitext(os.path.basename(csv_file))[0]
    output_dir = os.path.join(dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    save_visualizations(data, correlation_matrix, output_dir)

    llm_prompt = prepare_llm_prompt(data, missing_values)
    llm_analysis = get_llm_response(llm_prompt)

    generate_readme(data, missing_values, llm_analysis, output_dir)

    print(f"Analysis complete. Outputs saved in {output_dir}/README.md and PNG files.")

if __name__ == "__main__":
    main()
