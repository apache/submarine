#!/usr/bin/env python
import sys
from xml.etree import ElementTree

def run(files):
    first = None
    for filename in files:
        data = ElementTree.parse(filename).getroot()
        if first is None:
            first = data
        else:
            first.extend(data)
    if first is not None:
        print ElementTree.tostring(first)

if __name__ == "__main__":
    run(sys.argv[1:])
