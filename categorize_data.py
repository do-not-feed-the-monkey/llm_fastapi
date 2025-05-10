import json
import httpx
import asyncio
from typing import List, Dict

# Define the Pydantic models (as provided)
from pydantic import BaseModel, Field

class CategoryOutput(BaseModel):
    category: str = Field(description="The determined category of the event")
    confidence: float = Field(description="Confidence score (0-1) for the assigned category")

class CategoryInput(BaseModel):
    title: str
    description: str

async def categorize_event(endpoint_url: str, event_data: Dict) -> Dict:
    """
    Calls the categorization endpoint for a single event.

    Args:
        endpoint_url: The URL of the /categorize_event endpoint.
        event_data: A dictionary containing the 'title' and 'description' of the event.

    Returns:
        A dictionary containing the 'category' and 'confidence' fields,
        or None if the categorization fails.
    """
    category_input = CategoryInput(title=event_data['title'], description=event_data['description'])
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint_url, json=category_input.model_dump())
            response.raise_for_status()  # Raise an exception for bad status codes
            category_output = CategoryOutput(**response.json())
            return {
                "category": category_output.category,
            }
    except httpx.HTTPError as e:
        print(f"Error categorizing event '{event_data['title']}': {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while categorizing event '{event_data['title']}': {e}")
        return None

async def process_and_update_events(endpoint_url: str, filepath: str):
    """
    Processes a list of events by calling the categorization endpoint for each
    and updates the events in the JSON file with the category information.

    Args:
        endpoint_url: The URL of the /categorize_event endpoint.
        filepath: The path to the JSON file (e.g., "data.json").
    """
    try:
        with open(filepath, "r+", encoding="utf-8") as f:
            data = json.load(f)
            events = data.get("events", [])
            updated_events = []
            for event in events:
                category_info = await categorize_event(endpoint_url, event)
                if category_info:
                    updated_event = {**event, **category_info}
                    updated_events.append(updated_event)
                else:
                    updated_events.append(event) # Keep the original event if categorization failed

            data["events"] = updated_events
            f.seek(0)  # Go to the beginning of the file
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.truncate() # Remove remaining part if the new content is shorter

    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}.")
        return

async def main():
    endpoint_url = "http://127.0.0.1:8000/categorize_event"  # Replace with your actual endpoint URL
    json_filepath = "data.json"

    await process_and_update_events(endpoint_url, json_filepath)
    print(f"Successfully added 'category' field to {json_filepath}.")

if __name__ == "__main__":
    asyncio.run(main())