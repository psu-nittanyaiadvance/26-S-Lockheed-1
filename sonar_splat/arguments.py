from argparse import ArgumentParser

class ModelParams:
    def __init__(self, parser, sentinel=False):
        pass

class PipelineParams:
    def __init__(self, parser):
        pass

def get_combined_args(parser):
    return parser.parse_args()
