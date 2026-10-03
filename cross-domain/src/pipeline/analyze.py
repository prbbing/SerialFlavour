"""Summarize completed evaluation artifacts."""


def run(context):
    return context.module("analysis").analyze(context)
