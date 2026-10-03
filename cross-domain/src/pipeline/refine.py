"""Fit downstream readouts using B main-task supervision."""


def run(context):
    return context.module("refine").train(context)
