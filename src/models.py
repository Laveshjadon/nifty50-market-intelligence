from sqlalchemy import Column, String, Float, DateTime, Integer, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class NiftyTicks(Base):
    __tablename__ = 'nifty_ticks'
    ticker = Column(String(20), primary_key=True)
    timestamp = Column(DateTime, primary_key=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

class Min5(Base):
    __tablename__ = 'min_5'
    ticker = Column(String(20), primary_key=True)
    bucket_time = Column(DateTime, primary_key=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

class Nifty50Index5m(Base):
    __tablename__ = 'nifty50_index_5m'
    index_name = Column(String(50), primary_key=True)
    bucket_time = Column(DateTime, primary_key=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

class RefinedNews(Base):
    __tablename__ = 'refined_news'
    news_id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, index=True)
    title = Column(Text)
    url = Column(Text)
    source_file = Column(Text)
    categories = Column(Text)
    relevance_score = Column(Integer)
    has_negation = Column(Boolean)
    impact_tier = Column(String)

class Nifty50Constituents(Base):
    __tablename__ = 'nifty50_constituents'
    ticker = Column(String(50), primary_key=True)
    weight = Column(Float, nullable=False)
    last_updated = Column(DateTime, nullable=False)
