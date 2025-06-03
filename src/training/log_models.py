import mlflow
import os
from src.utils.logging_utils import setup_logger
from src.utils.config_loader import load_config

logger = setup_logger(__name__)
config = load_config()
mlflow_cfg = config.get('mlflow_config', {})

def log_generic_model(model, model_name, artifact_path, signature=None, registered_model_name=None):
    """
    A generic function to log a model to MLflow.
    This can be expanded based on specific model types (PyTorch, etc.).
    """
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", mlflow_cfg.get("tracking_uri")))
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "DefaultExperiment"))

        with mlflow.start_run(run_name=f"{model_name}_LoggingRun") as run:
            logger.info(f"Logging model '{model_name}' with artifact path '{artifact_path}'.")

            # Example for a generic PyFunc model, adapt as needed
            # mlflow.pyfunc.log_model(
            #     python_model=model, # This would be a class instance that implements pyfunc interface
            #     artifact_path=artifact_path,
            #     signature=signature, # mlflow.models.infer_signature(input_data, output_data)
            #     registered_model_name=registered_model_name
            # )

            # For other model types like sklearn, pytorch, tensorflow, use their specific log_model functions
            # e.g., mlflow.sklearn.log_model(model, artifact_path, registered_model_name=registered_model_name)

            logger.info(f"Model '{model_name}' logged to run ID: {run.info.run_id}")
            if registered_model_name:
                logger.info(f"Model registered as '{registered_model_name}'.")

        return run.info.run_id
    except Exception as e:
        logger.error(f"Error logging model {model_name}: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    # This is an example, you'd call this from your actual training scripts
    logger.info("This script is intended to be used as a module for logging models.")
    logger.info("Example: from src.training.log_models import log_generic_model")
    # Example:
    # class MyModel(mlflow.pyfunc.PythonModel):
    #     def predict(self, context, model_input):
    #         return model_input * 2
    # log_generic_model(MyModel(), "MySimpleModel", "my_simple_model_pyfunc", registered_model_name="MySimplePyFuncModel")
    pass
