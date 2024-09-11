from langchain.pydantic_v1 import BaseModel, Field


class MFInput(BaseModel):
    monthly_amount: int = Field(description="Monthly investment amount")
    interest_rate: float = Field(description="Yearly Interest rate")
    period: int = Field(description="Total Investment Period")
