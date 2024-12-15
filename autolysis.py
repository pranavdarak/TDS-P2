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
# ]
# ///

import os
import sys
from wsgiref import headers
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
import chardet
from pathlib import Path
import asyncio
import scipy.stats as stats
from PIL import Image
import numpy as np

# Ensure UTF-8 output for compatibility
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Constants
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# Helper Functions

def get_token():
    """Retrieve API token from environment variables."""
    try:
        return os.environ["AIPROXY_TOKEN"]
    except KeyError as e:
        print(f"Error: Environment variable '{e.args[0]}' not set.")
        raise

async def load_data(file_path):
    """Load CSV data with encoding detection."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Error: File '{file_path}' not found.")

    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    print(f"Detected file encoding: {encoding}")
    return pd.read_csv(file_path, encoding=encoding or 'utf-8')

async def async_post_request(headers, data):
    """Make asynchronous HTTP POST requests."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(API_URL, headers=headers, json=data, timeout=30.0)
            response.raise_for_status()
            response.encoding = 'utf-8'
            return response.json()['choices'][0]['message']['content']
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e}")
            raise
        except Exception as e:
            print(f"Error during request: {e}")
            raise

async def generate_narrative(analysis, token, file_path):
    """Generate a detailed narrative based on analysis results."""
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

    prompt = (
        f"You are an experienced data analyst. Analyze the following data for the file '{file_path.name}':\n\n"
        f"Column Names: {list(analysis['summary'].keys())}\n"
        f"Summary Statistics: {analysis['summary']}\n"
        f"Missing Values: {analysis['missing_values']}\n"
        f"Correlation Matrix: {analysis['correlation']}\n"
        f"Numeric Trends: {analysis['numeric_trends']}\n"
        f"Skewness and Kurtosis: {analysis['skewness_kurtosis']}\n"
        f"Outliers: {analysis['outliers']}\n"
        f"Hypothesis Test Results (if available): {analysis.get('hypothesis_test', 'None')}\n\n"
        "Provide a detailed narrative with explicit statistical insights, "
        "explaining correlations, trends, anomalies, and outliers. Suggest further analyses if necessary."
    )

    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]}
    try:
        return await async_post_request(headers, data)
    except Exception as e:
        print(f"Error during narrative generation: {e}")
        return "Narrative generation failed due to an error."

async def generate_refined_narrative(analysis, token, file_path):
    """Generate a refined narrative through multiple iterations with the LLM."""
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    refined_narrative = ""

    # First query for general insights
    prompt1 = (
        f"Analyze the following data from file '{file_path.name}' and summarize it:\n"
        f"Summary Statistics: {analysis['summary']}\n"
        f"Missing Values: {analysis['missing_values']}\n"
        f"Correlation Matrix: {analysis['correlation']}\n\n"
        "Provide general trends and areas needing further analysis."
    )
    data1 = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt1}]}
    response1 = await async_post_request(headers, data1)
    refined_narrative += response1

    # Second query for specific insights on correlations
    prompt2 = (
        f"Based on the correlation matrix and numeric trends:\n"
        f"Correlation Matrix: {analysis['correlation']}\n"
        "Identify key variables with significant correlations and suggest possible causal relationships."
    )
    data2 = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt2}]}
    response2 = await async_post_request(headers, data2)
    refined_narrative += "\n\n" + response2

    return refined_narrative


async def analyze_data(df, token):
    """Perform detailed data analysis and request insights from LLM."""
    if df.empty:
        raise ValueError("Error: Dataset is empty.")

    numeric_df = df.select_dtypes(include=['number'])
    categorical_df = df.select_dtypes(include=['object', 'category'])

    # Generate insights for numerical data
    summary_stats = df.describe(include='all').to_dict()
    missing_values = df.isnull().sum().to_dict()
    correlation = numeric_df.corr().to_dict() if not numeric_df.empty else {}

    # Extract specific trends for analysis
    numeric_trends = {
        column: {
            "mean": df[column].mean(),
            "std_dev": df[column].std(),
            "max": df[column].max(),
            "min": df[column].min()
        }
        for column in numeric_df.columns
    }

    # Calculate skewness and kurtosis
    skewness_kurtosis = {
        column: {
            "skewness": stats.skew(df[column].dropna()),
            "kurtosis": stats.kurtosis(df[column].dropna())
        }
        for column in numeric_df.columns
    }

    # Outlier detection
    outliers = {
        column: {
            "outliers": list(df[column][
                (df[column] > df[column].mean() + 3 * df[column].std()) |
                (df[column] < df[column].mean() - 3 * df[column].std())
            ])
        }
        for column in numeric_df.columns
    }

    # Hypothesis testing example
    hypothesis_test = {}
    if 'average_rating' in df.columns and 'num_pages' in df.columns:
        t_stat, p_value = stats.ttest_ind(df['average_rating'].dropna(), df['num_pages'].dropna())
        hypothesis_test = {'t_stat': t_stat, 'p_value': p_value}

    analysis = {
        'summary': summary_stats,
        'missing_values': missing_values,
        'correlation': correlation,
        'numeric_trends': numeric_trends,
        'skewness_kurtosis': skewness_kurtosis,
        'outliers': outliers,
        'hypothesis_test': hypothesis_test
    }

    # Request suggestions for further insights
    try:
        prompt = (
            f"Data analysis results for '{df.columns}':\n\n"
            f"Summary Statistics: {summary_stats}\n"
            f"Missing Values: {missing_values}\n"
            f"Correlation Matrix: {correlation}\n"
            f"Numeric Trends: {numeric_trends}\n"
            f"Skewness and Kurtosis: {skewness_kurtosis}\n"
            f"Outliers: {outliers}\n\n"
            "Based on these results, provide detailed insights into the dataset, "
            "highlight patterns, outliers, and possible future steps such as predictive modeling or clustering."
        )
        headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
        data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]}
        suggestions = await async_post_request(headers, data)
    except Exception as e:
        suggestions = f"Error fetching suggestions: {e}"

    print("Data analysis complete.")
    return analysis, suggestions
    
async def visualize_data(df, output_dir):
    """Generate and save visualizations with detailed commentary on their relevance."""
    sns.set(style="whitegrid")
    numeric_columns = df.select_dtypes(include=['number']).columns

    # Select columns for visualization
    selected_columns = numeric_columns[:3] if len(numeric_columns) >= 3 else numeric_columns

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save visualizations with commentary
    visualization_comments = []

    for column in selected_columns:
        plt.figure(figsize=(8, 6))  # Adjusted figure size for better readability
        sns.histplot(df[column].dropna(), kde=True, color='cornflowerblue')  # Updated color
        plt.title(f'Distribution of {column}', fontsize=14)  # Clearer title
        plt.xlabel(column, fontsize=12)  # Added label with increased font size
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)  # Improved gridlines for clarity
        file_name = output_dir / f'{column}_distribution.png'
        plt.savefig(file_name, dpi=100)
        plt.close()

        # Add commentary on the visualization's relevance
        comment = (
            f"The distribution plot of '{column}' is useful for understanding the "
            "spread and skewness of the data. The inclusion of a kernel density estimate (KDE) "
            "provides a smoother representation of the data distribution, highlighting key trends like "
            "central tendency and potential outliers. This visualization helps in detecting patterns such "
            "as bimodal distributions, skewness, and unusual peaks in the data."
        )
        visualization_comments.append(comment)

    if len(numeric_columns) > 1:
        plt.figure(figsize=(10, 8))  # Larger heatmap size
        corr = df[numeric_columns].corr()
        sns.heatmap(
            corr, annot=True, cmap='viridis', square=True,  # Updated color palette
            cbar_kws={"shrink": 0.8}, annot_kws={"fontsize": 10}
        )
        plt.title('Correlation Heatmap', fontsize=16)  # Clearer title
        plt.xticks(fontsize=12, rotation=45)  # Adjusted font size and rotation for clarity
        plt.yticks(fontsize=12)
        file_name = output_dir / 'correlation_heatmap.png'
        plt.savefig(file_name, dpi=100)
        plt.close()

        # Add commentary on the correlation heatmap
        comment = (
            "The correlation heatmap provides an overview of the relationships between numeric variables. "
            "It helps in identifying strong correlations and potential multicollinearity issues. "
            "By visualizing this matrix, key pairs of variables with high correlations can be spotted, "
            "which may reveal underlying patterns or redundancies in the data."
        )
        visualization_comments.append(comment)

    # Return visualizations and their associated comments
    return visualization_comments

async def analyze_image_quality_and_content(output_dir, token):
    """Analyze generated images using an image analysis agent."""
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    image_analysis_results = {}
    
    # Iterate through images in the output directory
    for img_path in output_dir.glob("*.png"):
        try:
            with Image.open(img_path) as img:
                img_array = np.array(img)
                # Example analysis: Check image properties
                brightness = np.mean(img_array)
                size = img.size
                mode = img.mode
                image_analysis_results[img_path.name] = {
                    "brightness": brightness,
                    "size": size,
                    "mode": mode
                }
                
                # Sending image quality insights to an agent for further interpretation
                analysis_prompt = (
                    f"Analyze the following image characteristics:\n"
                    f"Image Name: {img_path.name}\n"
                    f"Brightness: {brightness}\n"
                    f"Size: {size}\n"
                    f"Mode: {mode}\n\n"
                    "Provide feedback on how these characteristics affect the overall "
                    "quality and interpretability of the visualizations for the given dataset."
                )
                data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": analysis_prompt}]}
                image_quality_feedback = await async_post_request(headers, data)
                image_analysis_results[img_path.name]["feedback"] = image_quality_feedback
                
        except Exception as e:
            print(f"Error analyzing image {img_path.name}: {e}")
    
    return image_analysis_results

async def agentic_workflow_for_visualizations_and_analysis(df, output_dir, token):
    """Manage the multi-agent workflow for visualizations and image analysis."""
    print("Starting multi-agent workflow...")

    # Step 1: Generate visualizations and return commentary
    print("Generating visualizations and gathering insights...")
    visualization_comments = await visualize_data(df, output_dir)

    # Step 2: Analyze the visualizations with an image analysis agent
    print("Analyzing image quality and content...")
    image_analysis = await analyze_image_quality_and_content(output_dir, token)
    
    # Step 3: Combine insights and generate a comprehensive analysis report
    print("Generating report with multi-agent feedback...")
    comprehensive_report = ""
    comprehensive_report += "\n\n### Visualization Insights\n" + "\n".join(visualization_comments)
    comprehensive_report += "\n\n### Image Analysis Feedback\n"
    
    for img, analysis in image_analysis.items():
        comprehensive_report += (
            f"\n- **Image: {img}**\n"
            f"  - Brightness: {analysis['brightness']}\n"
            f"  - Size: {analysis['size']}\n"
            f"  - Mode: {analysis['mode']}\n"
            f"  - Feedback: {analysis['feedback']}\n"
        )
    
    # Step 4: Generate further insights using another agent (optional)
    headers= {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    additional_analysis_prompt = (
        f"Given the following report, generate further insights about the data:\n\n"
        f"Report: {comprehensive_report}\n\n"
        "Provide suggestions on further analyses or actions that should be taken based on this data."
    )
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": additional_analysis_prompt}]}
    additional_feedback = await async_post_request(headers, data)
    comprehensive_report += "\n\n### Further Insights\n" + additional_feedback

    return comprehensive_report


async def save_narrative_with_images(narrative, output_dir):
    """Save narrative to README.md and embed image links."""
    readme_path = output_dir / 'README.md'
    image_links = "\n".join(
        [f"![{img.name}]({img.name})" for img in output_dir.glob('*.png')]
    )
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(narrative + "\n\n" + image_links)
    print(f"Narrative successfully written to {readme_path}")

async def main(file_path):
    """Main function to orchestrate the data analysis workflow."""
    print("Starting autolysis process...")

    file_path = Path(file_path)
    if not file_path.is_file():
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)

    try:
        token = get_token()
    except Exception as e:
        print(e)
        sys.exit(1)

    try:
        df = await load_data(file_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    print("Dataset loaded successfully.")

    print("Analyzing data...")
    try:
        analysis, suggestions = await analyze_data(df, token)
    except ValueError as e:
        print(e)
        sys.exit(1)

    output_dir = Path(file_path.stem)
    output_dir.mkdir(exist_ok=True)

    print("Generating visualizations...")
    await visualize_data(df, output_dir)

    analysis_report = await agentic_workflow_for_visualizations_and_analysis(df, output_dir, token)

    # Step 2: Save the final analysis report with insights
    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(analysis_report)


    print("Generating refined narrative using iterative LLM calls...")
    narrative = await generate_refined_narrative(analysis, token, file_path)


    if narrative != "Narrative generation failed due to an error.":
        await save_narrative_with_images(narrative, output_dir)
    else:
        print("Narrative generation failed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))