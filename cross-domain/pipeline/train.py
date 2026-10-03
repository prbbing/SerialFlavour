"""Train and select upstream checkpoints on A only."""


def run(context):
    return context.module("training").train(context)
