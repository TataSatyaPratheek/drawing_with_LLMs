# Overview
Given a text prompt describing an image, your task is to generate Scalable Vector Graphics (SVG) code that renders it as an image as closely as possible. Your submissions will be built with [Kaggle Packages](https://kaggle.com/docs/packages?_gl=1*1nkvwdi*_ga*MTk2ODA5ODY4Mi4xNzQxNDIwMTAy*_ga_T7QHS60L4Q*MTc0MTQyNDMxNi4yLjEuMTc0MTQyNTY0Mi4wLjAuMA..), a new feature for building reusable models as Python libraries. For more information on [Scalable Vector Graphics (SVG)](https://en.wikipedia.org/wiki/SVG), please visit the Wikipedia page.

We have two Starter Notebooks with more details:

1. Official Starter Notebook that outlines our new Kaggle Packages feature and demos a trivial model.
2. Getting Started with Gemma 2 Notebook that demos a more advanced model but assumes familiarity with Kaggle Packages.

## Description
Specialized solutions can significantly outperform even the impressive capabilities of generative models like ChatGPT and Gemini while providing greater transparency into how they’re built. And while LLMs may demonstrate “sparks of AGI”, their capacity to generate image-rendering code is one area that needs improvement.

This competition challenges you to build practical, reusable solutions for image generation that follow robust software engineering patterns. Given a text description of an image, your task is to generate SVG code which renders it as closely as possible. Scalable Vector Graphics (SVG) is a vector image format that uses XML to describe two-dimensional graphics which can be scaled in size without quality loss.

Your submission, created using the new Kaggle Packages feature, will be a class Model with a predict() function which generates SVG code for a given prompt. The end results will be a set of deployable model packages evaluated on their ability to deeply reason about abstract descriptions and translate them into precise, executable code.

If you have feedback or questions, please let us know in this competition's Discussion forum. We appreciate your input as we improve and continue to develop Kaggle Packages as a new way to run Kaggle Competitions.

## Package Requirements

### This is a Package Competition
This is the first competition that requires submissions to be Kaggle Packages. This new feature expands on experimental functionality introduced in a few recent competitions and allows you to:

- Submit models defined with a predict() function for scoring
- Create simpler solutions as Kaggle’s scoring infrastructure will handle the test set iteration
- Share and discover solutions that are reusable as installable inferenceable models with defined dependencies using kagglehub.

When you submit your Package to this competition, we will install your Notebook's Package in a Container with your Notebook's Accelerator and Environment settings, with Internet disabled. Our scoring system will call your Model to run inference over the hidden test set and determine your score.

To learn how to create a Kaggle Package to submit to this competition, fork one of our two starter notebooks (basic or advanced) and check out our documentation.

### Package Competitions are Code Competitions

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

- CPU Notebook <= 9 hours run-time
- GPU Notebook <= 9 hours run-time
- Internet access disabled
- Freely & publicly available external data is allowed, including pre-trained models. You are free to use any tools, libraries, and pre-trained models to build your solution. You are also free to use any data that is freely & publicly available.

Please see the Code Competition FAQ for more information on how to submit. And review the code debugging doc if you are encountering submission errors.

## Evaluation

Submissions are evaluated by the mean CLIP similarity between the given descriptions and the submitted SVG code.

For each given description in the test set, your model must produce an SVG image depicting the scene described. We convert each SVG to a PNG image with the cairosvg Python library and append "SVG illustration of " to each description. We then compute the similarity between the text description and the PNG image as the cosine similarity of their feature embeddings as produced by a SigLIP SoViT-400m model. The final score is the average of these similarity scores.

In order to ensure the generated images adhere to the spirit of the competition, the SVG code must satisfy a number of constraints:

1. No SVG may more than 10,000 bytes long.
2. Each SVG may only include elements and attributes from an allowlist. Note that CSS style elements are not allowed.
3. No SVG may include any rasterized image data or data from external sources.

In addition, the evaluation system requires that:

1. Your model returns an SVG within 5 minutes of being passed a description.
2. All SVGs are generated in under 9 hours.

Submissions with SVGs violating any of these constraints are invalid.

You may review the implementation of the metric here: SVG Image Fidelity. This metric imports the svg_constraints package defined here: SVG Constraints. You may wish to use this constraints package to help ensure your submissions are valid.