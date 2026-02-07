"""
API Key Manager - Secure handling of API keys from environment or config
"""

import os
import logging
from typing import Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class APIKeyManager:
    """Manage API keys from environment variables, .env file, or config"""
    
    _instance = None
    _keys: Dict[str, str] = {}
    
    # Service to environment variable mapping
    SERVICE_ENV_MAP = {
        'google_vision': 'GOOGLE_VISION_API_KEY',
        'roboflow': 'ROBOFLOW_API_KEY',
        'clarifai': 'CLARIFAI_API_KEY',
        'huggingface': 'HUGGINGFACE_API_KEY'
    }
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_dotenv()
        return cls._instance
    
    def _load_dotenv(self):
        """Load environment variables from .env file if it exists"""
        try:
            from dotenv import load_dotenv
            
            # Check for .env in project directory
            env_paths = [
                Path('.env'),
                Path(__file__).parent.parent / '.env',
                Path.home() / '.image_recognition.env'
            ]
            
            for env_path in env_paths:
                if env_path.exists():
                    load_dotenv(env_path)
                    logger.info(f"Loaded environment from: {env_path}")
                    break
                    
        except ImportError:
            logger.debug("python-dotenv not installed, using system environment only")
    
    @classmethod
    def get_key(cls, service: str) -> Optional[str]:
        """
        Get API key for a service
        
        Args:
            service: Service name ('google_vision', 'roboflow', 'clarifai', 'huggingface')
            
        Returns:
            API key string or None if not found
        """
        instance = cls()
        
        # Check cache first
        if service in instance._keys:
            return instance._keys[service]
        
        # Get environment variable name
        env_var = instance.SERVICE_ENV_MAP.get(service, f"{service.upper()}_API_KEY")
        
        # Try to get from environment
        key = os.environ.get(env_var)
        
        if key:
            instance._keys[service] = key
            logger.debug(f"Found API key for {service}")
            return key
        
        logger.warning(f"No API key found for {service}. Set {env_var} environment variable.")
        return None
    
    @classmethod
    def set_key(cls, service: str, key: str):
        """Manually set an API key (for programmatic configuration)"""
        instance = cls()
        instance._keys[service] = key
        logger.debug(f"API key set for {service}")
    
    @classmethod
    def has_key(cls, service: str) -> bool:
        """Check if an API key is available for a service"""
        return cls.get_key(service) is not None
    
    @classmethod
    def get_available_services(cls) -> list:
        """Get list of services with available API keys"""
        instance = cls()
        available = []
        for service in instance.SERVICE_ENV_MAP:
            if cls.has_key(service):
                available.append(service)
        return available


def validate_api_keys():
    """Validate and report on available API keys"""
    print("\n=== API Key Status ===")
    for service, env_var in APIKeyManager.SERVICE_ENV_MAP.items():
        if APIKeyManager.has_key(service):
            print(f"✓ {service}: Configured ({env_var})")
        else:
            print(f"✗ {service}: Not configured (set {env_var})")
    print()


if __name__ == "__main__":
    validate_api_keys()
