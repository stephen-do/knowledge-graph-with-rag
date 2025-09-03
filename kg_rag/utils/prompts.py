"""Prompt templates for different RAG approaches with standard, Chain-of-Thought (CoT), and numerical answer variations."""

from datetime import datetime
from typing import Any, Literal


# Base prompts for different RAG systems
BASELINE_SYSTEM_PROMPT = """You are a helpful assistant that answers queries about SEC 10-Q filings using the blocks of context provided.
Just follow the instructions from the question exactly and use the context to provide accurate information.
The current date is {date}."""

ENTITY_SYSTEM_PROMPT = """You are a helpful assistant that answers queries about SEC 10-Q filings using knowledge graph data mapped to context chunks.
Base your answer ONLY on the provided context and the mapped entities and relationships.
The current date is {date}."""

CYHER_RESULT_INTERPRETATION = """\
Result semantics:
- The provided context is the POST-QUERY result for the user's question (already filtered by the appropriate Cypher).
- Treat every row as ALREADY linked to the target entities/relationships in the question unless a field explicitly contradicts it.
- Do NOT claim that linkage is missing just because the row does not redundantly repeat the relationship; the filtering has been applied upstream.
- All results are linked with the object inputs.

Aliases:
- p:Patient ; s:Specialization ; hp:HealthcareProvider ; l:Location

Large results policy:
- First return the total count of unique matches (deduplicate by stable id like `uri`).
- Then return a sample subset (e.g., 5 records).

Answering policy:
- If `p` objects are present, answer with those patients (e.g., list names and optional fields such as age, gender, condition).
- Only say “no results” when the context array is truly empty after deduplication.
- Do not ask for more information; base the answer solely on the provided context.
"""


GRAPHRAG_SYSTEM_PROMPT = """You are a helpful assistant that answers queries about SEC 10-Q filings using community-aware graph information.
Provide your answers based solely on the information in the provided graph communities.
The current date is {date}."""

# Chain-of-Thought extension
COT_EXTENSION = """
Follow these steps carefully:
1. PLAN: Break down what information you need to find in the context
2. SEARCH: Locate the relevant information in the provided context only
3. CALCULATE: If needed, perform any calculations step by step
4. VERIFY: Double-check your work and ensure your answer matches the question
5. FORMAT: Format your answer appropriately based on the question

Response should be formatted as valid JSON with the following structure:
{
    "reasoning": "Your detailed step-by-step analysis showing:
        1. What specific information you're looking for
        2. Where you found it in the context
        3. Any calculations performed
        4. How you verified the answer",
    "answer": "your final answer"
}
"""
# Numerical answer extension
NUMERICAL_EXTENSION = """

Important Rules:
- Base your answer ONLY on the provided context
- Do not make assumptions or use external knowledge besides the context provided
- Numbers must be whole integers without comma separators, unless specified
- Percentages must be whole numbers without % sign
- The answer field must contain ONLY the numerical value, no text or units
"""


def get_prompt(
    system_type: Literal["baseline", "entity", "cypher", "graphrag"],
    use_cot: bool = False,
    numerical_answer: bool = False,
) -> str:
    """
    Get the appropriate prompt for a given system type with optional extensions.

    Args:
        system_type: Type of RAG system ('baseline', 'entity', 'cypher', or 'graphrag')
        use_cot: Whether to use Chain-of-Thought prompting
        numerical_answer: Whether to format the answer as a numerical value only

    Returns
    -------
        The appropriate prompt template as a string
    """
    date = datetime.now().strftime("%B %d, %Y")

    # Select base prompt based on system type
    base_prompt_map = {
        "baseline": BASELINE_SYSTEM_PROMPT,
        "entity": ENTITY_SYSTEM_PROMPT,
        "cypher": CYHER_RESULT_INTERPRETATION,
        "graphrag": GRAPHRAG_SYSTEM_PROMPT,
    }

    base_prompt = base_prompt_map.get(system_type, BASELINE_SYSTEM_PROMPT).format(
        date=date
    )

    # Add extensions as needed
    extensions = []

    if use_cot:
        extensions.append(COT_EXTENSION)

    if numerical_answer:
        extensions.append(NUMERICAL_EXTENSION)

    # Combine base prompt with extensions
    full_prompt = base_prompt
    for extension in extensions:
        full_prompt += extension

    return full_prompt


def create_query_prompt(
    question: str,
    context: str,
    system_type: Literal["baseline", "entity", "cypher", "graphrag"],
    use_cot: bool = False,
    numerical_answer: bool = False,
) -> list[dict[str, Any]]:
    """
    Create a complete query prompt for a given question and context.

    Args:
        question: The user question
        context: The retrieved context
        system_type: Type of RAG system
        use_cot: Whether to use Chain-of-Thought prompting
        numerical_answer: Whether to format the answer as a numerical value only

    Returns
    -------
        A dictionary with prompt messages ready for the LLM
    """
    system_prompt = get_prompt(system_type, use_cot, numerical_answer)

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Using the following context, answer this question: {question}\n\nContext: {context}",
        },
    ]


# For GraphRAG compatibility, provide a function to create a prompt suffix
def get_graphrag_prompt_suffix(numerical_answer: bool = False) -> str:
    """
    Get a prompt suffix for GraphRAG that can be appended to LOCAL_SEARCH_SYSTEM_PROMPT.

    Args:
        numerical_answer: Whether to format the answer as a numerical value only

    Returns
    -------
        Prompt suffix as a string
    """
    current_date = datetime.now().strftime("%B %d, %Y")

    if not numerical_answer:
        return f"\nThe current date is {current_date}."

    return f"""
            Important Rules:
            - Base your answer ONLY on the provided context
            - Do not make assumptions or use external knowledge besides the context provided
            - Numbers must be whole integers without comma separators, unless specified
            - Percentages must be whole numbers without % sign
            - The answer field must contain ONLY the numerical value, no text or units
            - Your entire response must be valid JSON

            The current date is {current_date}.
            """
