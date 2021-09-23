from dataclasses import dataclass, field
from typing import Any, Dict

import gradio as gr
import gradio.inputs
import gradio.outputs

from .client import Client


@dataclass(frozen=True)
class GradioClient(Client):
    interface_args: Dict[str, Any] = field(default_factory=dict)
    launch_args: Dict[str, Any] = field(default_factory=dict)

    def __call__(self):
        interface = gr.Interface(
            fn=lambda prompt: self.generate(prompt),
            inputs=[
                gradio.inputs.Textbox(
                    label="prompt",
                    lines=10,
                    placeholder="Write a prompt for the model to process",
                )
            ],
            outputs=[gradio.outputs.Textbox(label="response", type="str")],
            **self.interface_args
        )
        interface.launch(**self.launch_args)
