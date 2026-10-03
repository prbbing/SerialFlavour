"""Prepare domain inputs and isolated A/B/Y splits."""


def run(context):
    return context.module("data").prepare(context)
