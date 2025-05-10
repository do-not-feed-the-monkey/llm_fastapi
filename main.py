from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Body
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
import json

app = FastAPI()

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
embeddings = OpenAIEmbeddings()
json_output_parser = JsonOutputParser()

class SentimentDetail(BaseModel):
    strength: int = Field(ge=0, le=100, description="Sentiment strength between 0 and 100")
    detectedEmotions: List[str]


class Sentiment(BaseModel):
    agitation: SentimentDetail
    neutral: SentimentDetail
    positive: SentimentDetail


class SentimentInput(BaseModel):
    title: str
    description: str
    last_news: str


prompt_sentiment = ChatPromptTemplate.from_template("""
    Analyze the sentiment expressed in these recent tweets about an event in Lublin.

    Event: {title}

    Context: {description}

    Recent Tweets:
    {last_news}

    Provide a sentiment analysis with the following structure:
    {{
        "agitation": {{
            "strength": <0-100>,
            "detectedEmotions": ["emotion1", "emotion2", ...]
        }},
        "neutral": {{
            "strength": <0-100>,
            "detectedEmotions": ["emotion1", "emotion2", ...]
        }},
        "positive": {{
            "strength": <0-100>,
            "detectedEmotions": ["emotion1", "emotion2", ...]
        }}
    }}

    Ensure the JSON is valid and the strengths across all categories sum to 100.
    """
)
sentiment_parser = PydanticOutputParser(pydantic_object=Sentiment)
sentiment_chain = prompt_sentiment | llm | sentiment_parser


class NewEventInput(BaseModel):
    news_content: str


class NewEventOutput(BaseModel):
    title: str
    description: str


prompt_new_event = ChatPromptTemplate.from_template(
    """
You are an expert in creating concise titles and descriptions for news articles to represent new events.

Here is the unrelated News article:
{news_content}

Based on this news article, create a short, descriptive title and a concise description that captures the essence of the event.

Return your response as a JSON object with the following format in Polish language:
{{
    "title": "Short summary of the news event",
    "description": "Concise description of the news event."
}}

Strictly follow the JSON format in your response. Do not include any additional text or explanations.
"""
)
new_event_chain = prompt_new_event | llm | json_output_parser


class VerificationInput(BaseModel):
    event_info: Dict[str, Any]
    authority_info: str


class VerificationOutput(BaseModel):
    comparison: str
    correctness: str
    justification: str
    public_response: str


prompt_verification = ChatPromptTemplate.from_template(
    """
You are an expert in verifying information related to events. You will be given information about an event and official information from authorities. Your task is to:

1. Differentiate between the event information and the authority information.
2. Check if the authority information confirms, contradicts, or provides additional context to the event information.
3. Determine the correctness of the event information based on the authority information.
4. Provide a message in Polish language that suggests a public response to the event information, including a summary of the event, a summary of the authority information, and an assessment of the correctness of the event information.

Here is the Event information:
{event_info}

Here is the Official information from authorities:
{authority_info}

Provide your analysis in a JSON object with the following structure in Polish language:
{{
    "comparison": "Detailed comparison of the event and authority information.",
    "correctness": "Assessment of the event information's correctness (e.g., \\"Correct\\", \\"Incorrect\\", \\"Partially Correct\\", \\"Cannot Determine\\").",
    "justification": "Explanation for the correctness assessment, referencing specific details from both sources.",
    "public_response": "Suggested public response to the event information, including a summary of the event, a summary of the authority information, and an assessment of the correctness of the event information."
}}

Strictly follow the JSON format. Do not include any extra text.
"""
)
verification_chain = prompt_verification | llm | json_output_parser


class CategoryOutput(BaseModel):
    category: str = Field(description="The determined category of the event")
    confidence: float = Field(description="Confidence score (0-1) for the assigned category")


class CategoryInput(BaseModel):
    title: str
    description: str


prompt_category = ChatPromptTemplate.from_template(
    """
You are an expert in categorizing events based on their title and description.
Given the following event title and description, determine the most appropriate category for this event in Polish Language.
Also, provide a confidence score between 0 and 1 for your categorization.

Title: {title}
Description: {description}

Return your response as a JSON object with the following format:
{{
    "category": "The determined category",
    "confidence": <0.0-1.0>
}}

Strictly follow the JSON format. Do not include any additional text or explanations.
"""
)
category_parser = PydanticOutputParser(pydantic_object=CategoryOutput)
category_chain = prompt_category | llm | category_parser


@app.post("/categorize_event", response_model=CategoryOutput)
async def categorize_event(category_input: CategoryInput = Body(...)):
    """
    Categorizes an event based on its title and description.
    """
    try:
        result = category_chain.invoke(category_input.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_sentiment", response_model=Sentiment)
async def analyze_sentiment(sentiment_input: SentimentInput = Body(...)):
    """
    Analyzes the sentiment of recent news about an event in Lublin.
    """
    try:
        result = sentiment_chain.invoke(sentiment_input.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/create_new_event", response_model=NewEventOutput)
async def create_new_event_endpoint(new_event_input: NewEventInput = Body(...)):
    """
    Creates a concise title and description for a news article.
    """
    try:
        result = new_event_chain.invoke({"news_content": new_event_input.news_content})
        return NewEventOutput(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify_event_information", response_model=VerificationOutput)
async def verify_event_information_endpoint(verification_input: VerificationInput = Body(...)):
    """
    Verifies event information against official information from authorities.
    """
    try:
        result = verification_chain.invoke({
            "event_info": json.dumps(verification_input.event_info, ensure_ascii=False),
            "authority_info": verification_input.authority_info
        })
        return VerificationOutput(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))