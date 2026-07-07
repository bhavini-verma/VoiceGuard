import os
from fastapi import Header, HTTPException

API_KEY = os.getenv("VOICEGUARD_API_KEY", "voiceguard123")

def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid x-api-key header value"
        )
