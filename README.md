# CAIFRanking Stability

This fork is dedicated to studying the stability of fair rankings.


### Prerequisite for language

Python, Java, R
1. Python 3 

[Install Tutorial](https://installpython3.com/)

2. Java

[Install Tutorial](https://docs.oracle.com/en/java/javase/14/install/overview-jdk-installation.html#GUID-8677A77F-231A-40F7-98B9-1FD0B48C346A)

3. R

[Install Tutorial](https://www.r-project.org/)


### Assumptions for Input Dataset

1. The input dataset is a table that is stored in .CSV file.
2. We use two real datasets: [COMPAS](https://github.com/IBM/AIF360/blob/master/aif360/data/raw/compas/README.md), [MEPS](https://github.com/IBM/AIF360/blob/master/aif360/data/raw/meps/README.md), and synthetic data that is generated by ourselves. Both real datasets are preprocessed as specified in [IBM AIF360](https://github.com/IBM/AIF360).

### Install CAIFRanking

Step 1 Download CAIFRanking.

Step 2 Unzip the downloaded source file and initiate the python environment.

```bash
cd CAIFRank  # go to the CAIFRank repository that is just downloaded
python -m venv CAIFRanking
source CAIFRanking/bin/activate  # activate the environment for CAIFRanking
pip install -r requirements.txt
```

### Run CAIFRanking

All the experiments can be executed by bash script. For MacOS or Linux, run the following script under ROOT ACCESS.

For the experiments on synthetic data,
```bash
bash run_exp_syn.sh mv m2 data/mv_m2.json G
bash run_exp_syn.sh mvr m2 data/mvr_m2.json GR
bash run_exp_syn.sh mvp m2 data/mvp_m2.json G
```

For the experiments on real data,
```bash
bash run_exp_real_cm.sh
bash run_exp_real_mp.sh
bash run_exp_real_cs.sh
```

### Methodology 

Details can be found in [Causal intersectionality for fair ranking](https://arxiv.org/abs/2006.08688).


### License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
