from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    BASE_URL: str = "https://tai.tikdat.com"  # Default value, can be overridden by environment variable

settings = Settings() 