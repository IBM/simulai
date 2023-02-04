import glob

HEADER = """# (C) Copyright IBM Corp. 2019, 2020, 2021, 2022.
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#           http://www.apache.org/licenses/LICENSE-2.0
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
"""

HEADER_LINES = [line + "\n" for line in HEADER.split("\n")]

MODULE_NAMES = ["simulai", "examples"]

for module_name in MODULE_NAMES:
    print(f"Entering directory {module_name}")

    py_files = glob.glob(f"{module_name}/**/*.py", recursive=True)

    for py_file in py_files:
        print(f"Updating header for the file {py_file}.")

        with open(py_file, "r") as fp:
            content = fp.readlines()

        if HEADER in "".join(content):
            print("This file already has the header.")
        else:
            with open(py_file, "w") as fp:
                fp.writelines(HEADER_LINES + content)
