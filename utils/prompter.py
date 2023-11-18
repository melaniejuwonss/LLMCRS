import json
from typing import Union
import os


class Prompter(object):
    __slots__ = ("template", "_verbose", "args")

    def __init__(self, args, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        self.args = args
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            if args.stage == "crs":
                template_name = "withoutCoT"
            elif args.stage == "quiz":
                template_name = "alpaca_legacy"
        file_name = os.path.join(args.home, "templates", f"{template_name}.json")
        # if not osp.exists(file_name):
        #     raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
            self,
            instruction: str,
            input: Union[None, str] = None,
            label: Union[None, str] = None,
            isNew: bool = False,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label and self.args.isNew is True:
            if isNew is False:
                res = f"{res}\n1)Chat about the item mentioned in a given conversation.\n2){label}" # \nChat about the item mentioned in a given dialog.\n
            elif isNew is True:
                res = f"{res}\n1)Recommend new item that did not appear in a given conversation.\n2){label}" # \nRecommend the item (Do not recommend the items already mentioned in a given dialog).\n
        elif label and self.args.isNew is False:
            if isNew is False:
                res = f"{res}{label}"
            elif isNew is True:
                res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
