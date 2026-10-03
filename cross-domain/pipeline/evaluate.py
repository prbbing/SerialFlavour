"""Evaluate selected checkpoints on the locked test set."""


def run(context):
    return context.module("evaluate").evaluate(context)
