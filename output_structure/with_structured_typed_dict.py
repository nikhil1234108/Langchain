from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()
class Review(TypedDict):
    Key_Points: Annotated[list[str], "list the All key points in the review"]
    summary: Annotated[str, "A Breif summary of the review"]
    Rating: Annotated[int, "the rating of the review"]
    sentiment: Annotated[str, "Return sentiment of the review either positive, negitive or neureal"]
    pros: Annotated[Optional[list[str]], "list the All pros in the review"]
    cons: Annotated[Optional[list[str]], "list the All cons in the review"]
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
structured_model = model.with_structured_output(Review)
result = structured_model.invoke("Rating: 2/5 I had a disappointing stay. The room looked tired and wasn’t properly cleaned—there were stains on the bedding and dust in corners. The bathroom had a bad smell and the hot water was inconsistent. Noise from the hallway and nearby rooms made it hard to sleep. Check-in was slow, and when I reported issues, the response felt unhelpful and delayed. The location is okay, but overall it didn’t feel worth the price. I wouldn’t choose this hotel again unless they improve cleanliness and service.")
print(result)
print(result["Rating"])