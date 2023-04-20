This repo contains the instructions for the ML for Good bootcamp, for students and researchers interested in AI safety.

The program is aimed at beginners in machine learning, but is quite ambitious, and we hope that even advanced students will enjoy participating in this program.

# Curriculum ML4Good

First week: Classes, tutorials, peer coding:

- w1d1 - pytorch + CNN
- w1d2 - Interpretability - optimization
- w1d3 - transformer architecture
- w1d4 - experiments on GPT-2 - huggingface
- w1d5 - RL
- w1d6 - theoretical intepretability
- w1d7 - AI safety review

Second week: Starting the projects

- w2d1 - Presentation of possible projects and choice of project
- w2d2 - Literature review on the issue
- w2d3 - Beginning of the projets in groups.

We draw inspiration from the redwood mlab, which focuses mainly on the ML engineering part. In comparison, we will spend more time on conceptual aspects.

### Installation

To set up your machine, the simplest way is to install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and then create a virtual environment containing the correct versions of the dependencies.

```
git clone --depth 1 https://github.com/EffiSciencesResearch/ML4G.git
cd ML4G
conda create --name ML4G python=3.9 -y
conda activate ML4G
conda install pytorch=1.11.0 torchtext torchdata torchvision cudatoolkit=11.3 -c pytorch -y
pip install -r requirements.txt
```
