#!/usr/bin/env python3
"""
Scripting the loop
"""

while True:
    question = input("Q: ").lower().strip()

    exit_words = ['exit', 'quit', 'goodbye', 'bye']

    if question in exit_words:
        print("A: Goodbye")
        break

    else:
        print("A: ")
