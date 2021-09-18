from dataclasses import dataclass

from .client import Client


@dataclass(frozen=True)
class CLIClient(Client):

    separator: str = "-" * 79
    input_prefix: str = "Input: "
    output_prefix: str = "\nOutput:\n\n"

    def __call__(self):
        while True:
            print(self.separator)
            prompt = input(self.input_prefix)
            response = self.pipeline(prompt, **self.generation_args)[0][
                "generated_text"
            ]
            print(self.output_prefix + response)
