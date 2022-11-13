import random
import warnings
from io import BytesIO
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from stability_sdk import client
from PIL import Image
from IPython.display import display

import argparse
import subprocess
import sys
import time
import os


def setup():
    install_cmds = [
        ['pip', 'torch', 'install', 'ftfy', 'gradio', 'regex', 'tqdm', 'stability-sdk',
            'transformers==4.21.2', 'timm', 'fairscale', 'requests'],
        ['pip', 'install', '-e', 'git+https://github.com/openai/CLIP.git@main#egg=clip'],
        ['pip', 'install', '-e',
            'git+https://github.com/pharmapsychotic/BLIP.git@lib#egg=blip'],
        ['git', 'clone', 'https://github.com/pharmapsychotic/clip-interrogator.git']
    ]
    for cmd in install_cmds:
        print(subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode('utf-8'))

setup()

sys.path.append('src/blip')
sys.path.append('src/clip')
sys.path.append('clip-interrogator')


import clip
import gradio as gr
from clip_interrogator import Interrogator, Config

ci = Interrogator(Config())

stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'],
    verbose=True
)


def inferAndRebuild(image, mode):
    image = image.convert('RGB')
    output = ''
    if (mode == 'best'):
        output = ci.interrogate(image)
    elif (mode == 'classic'):
        output = ci.interrogate_classic(image)
    else:
        output = ci.interrogate_fast(image)

    answers = stability_api.generate(
        prompt=str(output),
        seed=34567,
        steps=30,
        samples=5
    )

    imglist = []
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generate.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed. Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(BytesIO(artifact.binary))
                imglist.append(img)
    return [imglist, output]


inputs = [
    gr.inputs.Image(type='pil'),
    gr.Radio(['best', 'classic', 'fast'], label='Models', value='fast')
]

outputs = [
    gr.Gallery(),
    gr.outputs.Textbox(label='Prompt')
]

io = gr.Interface(
    inferAndRebuild,
    inputs,
    outputs,
    allow_flagging=False,
)

io.launch(share=True, debug=True)
