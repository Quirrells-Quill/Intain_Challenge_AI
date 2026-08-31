import yaml
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, Any, List

class DataPathsConfig(BaseModel):
    """Pydantic model for data path configurations."""
    train_data: str = Field(..., description="Path to training data parquet file")
    test_data: str = Field(..., description="Path to testing data parquet file")
    static_data: str = Field(..., description="Path to static pool data parquet file")
    servicer_updates: str = Field(..., description="Directory containing ongoing servicer updates")

class ModelParamsConfig(BaseModel):
    """Pydantic model for model configurations and hyperparameter grids."""
    random_seed: int = Field(42, description="Global random seed for reproducibility")
    cv_folds: int = Field(5, description="Number of cross-validation folds")
    hyperparameter_grids: Dict[str, Dict[str, List[Any]]] = Field(
        ..., description="Model-specific hyperparameter search spaces"
    )

class MLFlowConfig(BaseModel):
    """Pydantic model for MLflow experiment tracking configurations."""
    tracking_uri: str = Field(..., description="URI for the MLflow tracking server")
    experiment_name: str = Field(..., description="Name of the MLflow experiment")

class LLMSettingsConfig(BaseModel):
    """Pydantic model for Large Language Model configurations."""
    model_name: str = Field(..., description="Name of the LLM model used")
    temperature: float = Field(0.1, description="Sampling temperature for the LLM")
    structured_schema_paths: str = Field(..., description="Path to JSON schemas for structured output")

class AppConfig(BaseModel):
    """Root configuration model encompassing all application settings."""
    data_paths: DataPathsConfig
    model_params: ModelParamsConfig
    mlflow: MLFlowConfig
    llm_settings: LLMSettingsConfig

def load_config(config_path: str = "configs/config.yaml") -> AppConfig:
    """
    Loads YAML configuration and validates it against Pydantic models.
    
    Ensures that the Intain-Sight engine crashes early if critical configurations
    are missing, adhering to robust production engineering principles.
    
    Args:
        config_path (str): Relative or absolute path to the YAML configuration file.
        
    Returns:
        AppConfig: Fully validated and strictly typed application configuration.
        
    Raises:
        FileNotFoundError: If the config file does not exist.
        ValidationError: If the parsed YAML fails Pydantic type validation.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
        
    # Validates and instantiates the Pydantic configuration hierarchy
    return AppConfig(**config_dict)

if __name__ == "__main__":
    # Smoke test for configuration loading
    try:
        config = load_config()
        print(f"✅ Configuration loaded and validated successfully.")
        print(f"MLflow URI: {config.mlflow.tracking_uri}")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
