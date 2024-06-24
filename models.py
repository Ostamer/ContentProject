from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Content(Base):
    __tablename__ = 'content'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    content_type = Column(String, index=True)
    content_url = Column(Text)
    content_features = Column(Text)
