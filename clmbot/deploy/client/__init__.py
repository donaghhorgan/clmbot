from .cli import CLIClient
from .gradio import GradioClient
from .client import Client

gradio = Gradio = GradioClient
cli = CLI = CLIClient

__all__ = ["Client", "cli", "CLI", "CLIClient", "gradio", "Gradio", "GradioClient"]
