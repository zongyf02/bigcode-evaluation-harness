import json
from functools import partial

from evaluate import load
from datasets import load_dataset
from bigcode_eval.base import Task

LANGUAGES = ["python", "cpp", "js", "java", "go", "rust"]

URF_LANGUAGE_TO_NAME = {
    "python": "python",
    "cpp": "cpp",
    "js": "javascript",
    "java": "java",
    "go": "go",
    "rust": "rust",
}

LANGUAGE_TO_NAME = {
    "python": "Python",
    "cpp": "C++",
    "js": "JavaScript",
    "java": "Java",
    "go": "Go",
    "rust": "Rust",
}

# Taken from https://huggingface.co/datasets/nuprl/MultiPL-E/ & https://github.com/THUDM/CodeGeeX
LANGUAGE_TO_STOP_WORDS = {
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L164
    "python": ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\nassert"],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L185
    "cpp": [],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L188
    "js": [],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L177
    "go": ["\n//", "\nfunc main(", "struct", "\nfunc"],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L169
    "java": [],
    "rust": [],
}

LANGUAGE_TO_TIMEOUT = {
    "python": 10,
    "cpp": 60,
    "js": 10,
    "java": 10,
    "go": 20,
    "rust": 300, # Necessary for first-time compilation of cargo
}

# Java sometimes fails with more workers; For JS it's twice as fast with 4 workers
LANGUAGE_TO_NUM_WORKERS = {
    "python": 4,
    "cpp": 4,
    "js": 4,
    "java": 1,
    "go": 4,
    "rust": 1,
}

# https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L6
IMPORT_HELPER = {
    "python": [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import statistics",
        "import functools",
        "import hashlib",
        "import numpy",
        "import numpy as np",
        "import string",
        "from typing import *",
        "from collections import *",
    ],
    "go": [
        "math",
        "strings",
        "fmt",
        "strconv",
        "time",
        "bytes",
        "regexp",
        "sort",
        "math/rand",
        "crypto/md5",
    ],
    "cpp": [
        "using namespace std;",      
        "#include<stdlib.h>",
        "#include<algorithm>",
        "#include<cmath>",
        "#include<math.h>",
        "#include<numeric>",
        "#include<stdio.h>",
        "#include<vector>",
        "#include<set>",
        "#include<map>",
        "#include<queue>",
        "#include<stack>",
        "#include<list>",
        "#include<deque>",
        "#include<boost/any.hpp>",
        "#include<string>",
        "#include<climits>",
        "#include<cstring>",
        "#include<iostream>",
        "#include<sstream>",
        "#include<fstream>",
    ],
}


def create_all_tasks():
    tasks = {}
    for i in range(len(LANGUAGES)):
        for j in range(i + 1, len(LANGUAGES)):
            tasks[f"humanevaltranslate-{LANGUAGES[i]}-{LANGUAGES[j]}"] = create_task(LANGUAGES[i], LANGUAGES[j])
            tasks[f"humanevaltranslate-{LANGUAGES[j]}-{LANGUAGES[i]}"] = create_task(LANGUAGES[j], LANGUAGES[i])
    return tasks


def create_task(src_language, target_language):
    class HumanevalTranslate(HumanEvalTranslateBase):
        def __init__(self, src_language=src_language, target_language=target_language, prompt="instruct"):
            super().__init__(src_language=src_language, target_language=target_language, prompt=prompt)
    return HumanevalTranslate


class HumanEvalTranslateBase(Task):
    DATASET_PATH = "bigcode/humanevalpack"
    DATASET_NAME = None
    SRC_DATASET_NAME = None


    def __init__(self, src_language, target_language, prompt="instruct"):
        self.SRC_DATASET_NAME = src_language
        self.DATASET_NAME = target_language
    
        self.prompt = prompt
        stop_words = LANGUAGE_TO_STOP_WORDS[target_language]
        if self.prompt == "urf":
            stop_words.append("```")
        # stop_words.append("<|endoftext|>")
        super().__init__(stop_words=stop_words, requires_execution=True)
        self.src_dataset = load_dataset(path=self.DATASET_PATH, name=self.SRC_DATASET_NAME)


    def get_dataset(self):
        return list({"src": src, "target": target}
                    for src, target in
                    zip(self.src_dataset["test"], self.dataset["test"])
        )


    def get_prompt(self, doc):
        prompt_base = doc["target"]["declaration"]
        instruction = f"Translate the code snippet from {self.SRC_DATASET_NAME} to {self.DATASET_NAME}.\n"
        if self.prompt == "instruct":
            prompt = (instruction + doc["src"]["prompt"] + doc["src"]["canonical_solution"]).rstrip() + "\n\n" + prompt_base
        elif self.prompt == "deepseek":
            inp = instruction + doc["src"]["prompt"] + doc["src"]["canonical_solution"]
            prompt = f"You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n### Instruction:\n{inp.strip()}\n### Response:\n{prompt_base}"
        elif self.prompt == "urf":
            code_block_start_template = f"```{URF_LANGUAGE_TO_NAME[self.DATASET_NAME]}\n" + "{code}"
            inp = instruction + code_block_start_template.format(
                code = doc["src"]["prompt"] + doc["src"]["canonical_solution"]
            ) + "```"
            prompt_base = code_block_start_template.format(code=prompt_base)
            prompt = f"You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n@@ Instruction\n{inp}\n\n@@ Response\n{prompt_base}"
        else:
            raise ValueError(f"The --prompt argument {self.prompt} wasn't provided or isn't supported")
        return prompt.strip()


    def get_reference(self, doc):
        return doc["target"]["prompt"] + doc["target"]["canonical_solution"]


    def remove_last_block(self, code):
        """
        Adapted from https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L151
        """
        for w in self.stop_words:
            if w in code:
                code = code[:code.find(w)]

        ### Find the first occassion where a chain of { } is closed
        if self.DATASET_NAME == "python":
            for i, line in enumerate(code.split("\n")):
                if len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t':
                    return "\n".join(code.split("\n")[:i])
        elif self.DATASET_NAME in ["java", "js", "go", "cpp", "rust"]:
            open_brackets = 2 if self.DATASET_NAME == "java" else 1
            cut = False
            for i, c in enumerate(code):
                if c == '{':
                    open_brackets += 1
                elif c == '}':
                    open_brackets -= 1
                if open_brackets == 0:
                    code = code[:i+1]
                    cut = True
                    break
            if not cut:
                if self.DATASET_NAME == "java":
                    main_pos = code.find("public static void main")
                    if main_pos != -1:
                        code = code[:main_pos] + '}'
                    if '}' in code:
                        code = code[:code.rfind('}')] + '}'
                    if code.count('{') - 1 == code.count('}'):
                        code += "\n}"
                elif '}' in code:
                    code = code[:code.rfind('}')] + '}'
        return code


    def postprocess_generation(self, generation, idx):
        doc = self.get_dataset()[idx]
        prompt = self.get_prompt(doc)
        gen = self.remove_last_block(generation[len(prompt):].rstrip())
        # Strip to maintain same behavior as with get_prompt
        return doc["target"]["declaration"].rstrip() + gen


    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references.

        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        code_metric = load("Muennighoff/code_eval_octopack")
        timeout = LANGUAGE_TO_TIMEOUT[self.DATASET_NAME]
        num_workers = LANGUAGE_TO_NUM_WORKERS[self.DATASET_NAME]
        language = self.DATASET_NAME if self.DATASET_NAME != "js" else "javascript"

        ### CUSTOM PROG LANGUAGE CHANGES ###
        # Inspiration: https://github.com/THUDM/CodeGeeX/blob/ebeb850f227a90c79de39f7e26b1302f374f3240/codegeex/benchmark/evaluate_humaneval_x.py
        if language == "python":
            python_imports = "\n".join(IMPORT_HELPER["python"])
            generations = [
                [(python_imports + "\n" + g).strip() for g in gen] for gen in generations
            ]
        elif language == "cpp":
            cpp_imports = "\n".join(IMPORT_HELPER["cpp"])
            # Remove main in case present
            generations = [
                [(cpp_imports + "\n" + g.split("int main")[0]).strip() for g in gen] for gen in generations
            ]
        elif language == "java":
            generations = [
                [g.replace("public class Main {\n    }", "").strip() for g in gen] for gen in generations
            ]
        elif language == "go":
            ds = self.get_dataset().select(range(len(generations)))
            for gen, ref, doc in zip(generations, references, ds):
                for line in doc["import"].split("\n"):
                    line = line.replace("import", "").replace("(", "").replace(")", "").replace('"', "").strip()
                    if line: assert line in IMPORT_HELPER["go"], doc["import"] # Will be added later
                test_setup_str = doc["test_setup"] + "\n"
                for i, g in enumerate(gen):
                    for line in test_setup_str.split("\n"):
                        line = line.replace("import", "").replace("(", "").replace(")", "").strip()
                        if line.startswith('"') and line in g:
                            test_setup_str = test_setup_str.replace(line, "")
                    g = test_setup_str + g + "\n" + ref
                    other_pkgs = set()
                    for pkg in IMPORT_HELPER["go"]:
                        if ('"' + pkg + '"' not in g):
                            p = pkg.split("/")[-1]
                            # Check if the package is used
                            if (p + "." in g):
                                # The problem is that it could appear in a comment
                                # E.g. in problem 158, the docstring is:
                                # // ... a list of strings.
                                # but the "strings" pkg is never used
                                # Golang throws an error if the pkg is not used
                                # Thus search for the package & make sure it's not in a commented line
                                lines = g.split("\n")
                                for line in lines:
                                    if (p + "." in line) and not(line.strip().startswith("//")):
                                        other_pkgs.add('"' + p + '"')
                                        break
                    other_pkgs_str = ""
                    if other_pkgs:
                        other_pkgs_str = "import (\n" + "\n".join(["    " + p for p in other_pkgs]) + "\n)\n"
                    if ("package main" in gen[i]) and ("package main" in test_setup_str):
                        gen[i] = gen[i].replace("package main", "")
                    gen[i] = test_setup_str + other_pkgs_str + gen[i]
        elif language == "rust":
            ds = self.get_dataset().select(range(len(generations)))
            main = "fn main(){}\n"
            for gen, doc in zip(generations, ds):
                declaration = doc["declaration"]
                for i, g in enumerate(gen):
                    new_gen = ""
                    if "fn main()" not in g:
                        new_gen += main
                    for line in declaration.split("\n"):
                        if line.strip() not in g:
                            # Skip if the function is already present
                            if line.strip().startswith("fn") and (line.strip().split("(")[0]) in g:
                                continue
                            new_gen += line.strip() + "\n"
                    # If fn main() is present twice, cut off before the second one
                    g = "fn main()".join(g.split("fn main()")[0:2])
                    new_gen += g
                    gen[i] = new_gen

        ### EVALUATION ###
        results, logs = code_metric.compute(
            references=references,
            predictions=generations,
            language=language,
            timeout=timeout,
            num_workers=num_workers,
        )
        # Write logs to json
        with open("logs.json", "w") as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)

        """Debugging help
        for i, (gen, ref) in enumerate(zip(generations, references)):
            import time
            starttime = time.time()            
            results, log = code_metric.compute(
                references=[ref],
                predictions=[gen],
                language=language,
                timeout=timeout,
            )
            print("Took: ", time.time() - starttime)
            with open("errors.txt", "a") as f:
                f.write(log[0][0][1]["result"] + "\n")
            if ("compilation error" in log[0][0][1]["result"]):
                print("Result")
                print(results)
                print("Log")
                print(log)
                print("Gen")
                print(gen[0])
                print("Ref")
                print(ref)
        """
        return results
