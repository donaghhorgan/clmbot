from .cli import CLIClient
from .client import Client
from .gradio import GradioClient

gradio = Gradio = GradioClient
cli = CLI = CLIClient

__all__ = ["Client", "cli", "CLI", "CLIClient", "gradio", "Gradio", "GradioClient"]
