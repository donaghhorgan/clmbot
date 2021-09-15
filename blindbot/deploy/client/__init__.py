from .cli import CLIClient
from .client import Client

cli = CLI = CLIClient

__all__ = ["Client", "cli", "CLI", "CLIClient"]
