# LLM_FILTER
from __future__ import annotations
from typing import Any, Dict, List, Tuple  # Must import these
from typing_extensions import Annotated
import json
import time
import openai
import pandas as pd
from pydantic import BaseModel, Field, RootModel

AI_API_KEY = "YOUR_KEY"
AI_ENDPOINT = "https://openrouter.ai/api/v1"

AI_MODEL = "arcee-ai/trinity-large-preview"

def get_client():
    global AI_API_KEY 
    if not AI_API_KEY or "sk-proj" in AI_API_KEY: # Note: OpenRouter keys usually start with sk-or-v1
        raise ValueError("Please provide a valid OpenRouter API Key.")

    return openai.OpenAI(
        base_url=AI_ENDPOINT,
        api_key=AI_API_KEY,
        default_headers={
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Data Analyst Script",
        }
    )

FILTERING_PROMPT_FORMAT = """
You are a data analyst. Evaluate the synthetic data against the real sample.

For EVERY synthetic record, decide if it is 'good' (true) or 'bad' (false).

IMPORTANT: You MUST return a JSON OBJECT with a key named "flags". 
The value of "flags" MUST be a list of booleans of length {num_items}.

Example Output:
{{
  "flags": [true, false, true, true, false]
}}

Dataset Description: 
{data_desc}

Columns:
{column_descs}

Schema:
{schema}
""".strip()


class DataDescription(BaseModel):
    dataset_description: str = Field(
        description="A high-level description of what the dataset represents, its purpose, and key characteristics"
    )
    column_descriptions: Dict[str, str] = Field(
        description="A mapping of each column name to a description of what that column represents"
    )


DATA_DESCRIPTION_PROMPT = (
    """
You are a data analyst. Given a sample of a dataset in JSON format, 
provide a high-level description of the dataset and a description for each column. Be concise but informative.

Your output schould conform to the following JSON schema:
""".strip()
    + "\n"
    + json.dumps(DataDescription.model_json_schema(), indent=2)
)


def describe_data(
    df: pd.DataFrame, sample_size: int = 100
) -> Tuple[str, Dict[str, str]]:
    sample = df.sample(n=min(sample_size, len(df)), replace=False)
    sample_records = sample.to_dict(orient="records")
    sample_records_json = json.dumps(sample_records, indent=2)

    client = get_client()

    response = client.beta.chat.completions.parse(
        model=AI_MODEL,
        messages=[
            {
                "role": "system",
                "content": DATA_DESCRIPTION_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Here is a sample of the dataset:\n\n{sample_records_json}\n\n"
                    "Describe the overall dataset and each column."
                ),
            },
        ],
        response_format=DataDescription,
        max_completion_tokens=500, # Added this to prevent truncation
    )

    result = response.choices[0].message.parsed

    assert set(result.column_descriptions.keys()) == set(df.columns), (
        f"Not all columns were described. "
        f"Missing: {set(df.columns) - set(result.column_descriptions.keys())}, "
        f"Extra: {set(result.column_descriptions.keys()) - set(df.columns)}"
    )

    return result.dataset_description, result.column_descriptions


def generate_output_model(m: int) -> type[BaseModel]:
    class FilterOutput(BaseModel):
        flags: Annotated[
            List[bool],
            Field(
                min_length=m,
                max_length=m,
                description=f"A list of exactly {m} booleans",
            ),
        ]
    return FilterOutput


def sample_rows(df, n):
    return df.sample(n, replace=False)


def apply_filter(
    chunk: List[Dict[str, Any]],
    orig_sample: List[Dict[str, Any]],
    data_desc: str,
    column_descs: Dict[str, str],
) -> List[bool]:
    num_items = len(chunk)
    output_model = generate_output_model(num_items)

    column_descs_str = "\n".join(f"- {col}: {desc}" for col, desc in column_descs.items())
    
    # Pass num_items to the prompt so the LLM knows the expected length
    system_prompt = FILTERING_PROMPT_FORMAT.format(
        num_items=num_items,
        data_desc=data_desc,
        column_descs=column_descs_str,
        schema=json.dumps(output_model.model_json_schema(), indent=2),
    )

    client = get_client()
    response = client.beta.chat.completions.parse(
        model=AI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Real Sample:\n{json.dumps(orig_sample)}\n\nSynthetic Chunk:\n{json.dumps(chunk)}"
            },
        ],
        response_format=output_model,
    )

    # Return the 'flags' property from the object
    return response.choices[0].message.parsed.flags


def run_filter_chunked(
    df, orig_df, data_desc, column_descs, filtering_chunk_size=5, original_sample_size=5
):
    saved_rows = []

    for i in range(0, len(df), filtering_chunk_size):
        chunk = df.iloc[i : i + filtering_chunk_size]
        orig_sample = sample_rows(orig_df, original_sample_size)

        chunk_records = chunk.to_dict(orient="records")
        orig_sample_records = orig_sample.to_dict(orient="records")

        try:
            keep_flags = apply_filter(
                chunk_records, orig_sample_records, data_desc, column_descs
            )
        except Exception as e:
            print(f"Error in LLM Filter: {e}. Skipping this chunk.")
            keep_flags = [False] * len(chunk) # Default to rejecting if the API fails
        
        assert len(keep_flags) == len(chunk), (
            f"apply_filter returned {len(keep_flags)} flags for a chunk of size {len(chunk)}"
        )

        for row, keep in zip(chunk.itertuples(index=True), keep_flags):
            if keep:
                saved_rows.append(row.Index)

    return saved_rows