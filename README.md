# AI-Quest

## Overview

Fantasy text adventure games have been primarily hardcoded for a fixed set of action. We developed a text adventure game that uses deep learning NLP architectures like GPT2 and BERT to generate an infinite set of playable fantasy worlds. The course of the game changes based on the players responses.

## Prerequisites

* [ParlAI](https://github.com/facebookresearch/ParlAI)
* Pytorch
* [Huggingface Transformers Library](https://github.com/huggingface/transformers)
* CometML Account

## Installation Steps and Data Formatting
1. Run the following command inside the ParlAI directory to download the LIGHT model. `python examples/eval_model.py -t light_dialog -mf models:light/biranker_dialogue/model`
2. Locate the data used for the LIGHT training and testing.
3. Setup CometML in your project directory.
4. Run `python light.py --traing_file [put_train_file] --test_file [put_test_file] -T -m gpt2`

## Findings
You can find our final report in writeup pdf.  
