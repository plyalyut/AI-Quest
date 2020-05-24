# AI-Quest

## Overview

Fantasy text adventure games have been primarily hardcoded for a fixed set of action. We developed a text adventure game that uses deep learning NLP architectures like GPT2 and BERT to generate an infinite set of playable fantasy worlds. The course of the game changes based on the players responses.

## Prerequisites

* [ParlAI](https://github.com/facebookresearch/ParlAI)
* Pytorch
* [Huggingface Transformers Library](https://github.com/huggingface/transformers)

## Installation Steps and Data Formatting
1. Run the following command inside the ParlAI directory to download the LIGHT model.
'''python examples/eval_model.py -t light_dialog -mf models:light/biranker_dialogue/model'''
2. Locate the data used for the LIGHT training and testing.
3. 


Using BERT and GPT2 to model an agent in a text based fantasy adventure game.

You can find our final report in writeup.pdf. 
