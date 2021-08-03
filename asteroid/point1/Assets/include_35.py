#!/usr/bin/python
import re
import os
import sys

script_path = os.path.dirname(__file__) + "/"
output_path = script_path + "out/";

if not os.path.exists(output_path):
    os.makedirs(output_path)

for filename in os.listdir(script_path):
    _, file_extension = os.path.splitext(filename)

    if file_extension in (".vs", ".fs", ".gs", ".cs"):
        out_lines = []
        with open(script_path + filename) as in_file:
            for line in in_file.readlines():
                result = re.match(r'#include "(.*)"', line)
                if result:
                    include_name = result.group(1)
                    _, include_extension = os.path.splitext(include_name)
                    assert include_extension == ".h"
                    with open(script_path + include_name) as include_file:
                        if include_name == "noise.h":
                            lines = include_file.readlines()
                            lines = [line.replace("A0","0.390402") for line in lines]
                            lines = [line.replace("B0","0.29451") for line in lines]
                            lines = [line.replace("C0","1105.5") for line in lines]
                            out_lines += lines
                        else:
                            out_lines += include_file.readlines()
                else:
                    out_lines.append(line)
        with open(output_path + filename, "w") as out_file:
            out_file.write("".join(out_lines))
        print("Created " + output_path + filename)
