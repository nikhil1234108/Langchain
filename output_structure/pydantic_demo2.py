from pydantic import BaseModel, Field
from typing import Optional,List, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
class Review(BaseModel):
    Key_Points: list[str]=Field(description = "list the all key points in the review")
    summary: str=Field(description = "A Breif summary of the review")
    Rating: Optional [float]=Field(default = 0.0, gt = 0.0, lt = 5.0, description="Return the rating From the Review")
    sentiment: Literal["pos","neg"] = Field(description="Return sentiment of the review either positive, negitive or neutral")
    pros: Optional[list[str]] = Field(description="list the all pros")
    cons: Optional[list[str]] = Field(description="list the all cons")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
structured_model = model.with_structured_output(Review)
result = structured_model.invoke("Rating: 2/5 I had a disappointing stay. The room looked tired and wasn’t properly cleaned—there were stains on the bedding and dust in corners. The bathroom had a bad smell and the hot water was inconsistent. Noise from the hallway and nearby rooms made it hard to sleep. Check-in was slow, and when I reported issues, the response felt unhelpful and delayed. The location is okay, but overall it didn’t feel worth the price. I wouldn’t choose this hotel again unless they improve cleanliness and service.")
print(result)
print(result.Rating)