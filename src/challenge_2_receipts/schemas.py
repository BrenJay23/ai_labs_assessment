from pydantic import BaseModel, Field
from typing import Optional


class ReceiptEntities(BaseModel):
    company: Optional[str] = Field(None, description="Business or vendor name")
    date: Optional[str] = Field(
        None, description="Transaction date only, no time or timezone"
    )
    address: Optional[str] = Field(None, description="Business address")
    total: Optional[str] = Field(None, description="Final total amount")
