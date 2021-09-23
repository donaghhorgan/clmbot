from dataclasses import dataclass

from .client import Client


@dataclass(frozen=True)
class CLIClient(Client):

    separator: str = "-" * 79
    input_prefix: str = "Input: "
    output_prefix: str = "Output:\n"
    multiline: bool = True

    def __call__(self):
        input_func = multiline_input if self.multiline else input

        while True:
            try:
                print(self.separator)
                prompt = input_func(self.input_prefix)
                print(self.separator)
                print(self.output_prefix)
                response = self.generate(prompt)
                print(response)
            except KeyboardInterrupt:
                break


def multiline_input(prompt: str) -> str:
    prompt = prompt.rstrip()
    prompt += " (Press Ctrl+D to finish.)\n"
    print(prompt)

    lines = []
    while True:
        try:
            lines.append(input())
        except EOFError:
            break
    return "\n".join(lines)
