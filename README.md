# Script of "Bata-distribution-rescored Neighborhood Regularized Logistic Matrix Factorization for Improving Performance of Drug–Target Interaction Prediction"

Techniques for predicting interactions between a drug and a target (protein) are useful for strategic drug repositioning. Neighborhood regularized logistic matrix factorization (NRLMF) is known as one of the state-of-the-art drug--target interaction prediction method, which is based on a statistical model using the Bernoulli distribution. However, our survey revealed a problem that the prediction does not work well when drug--target interaction pairs have less active information (e.g. we supposed a sum of the number of ligands for a target and the number of target proteins for a drug). In this study, in order to solve this problem, we proposed neighborhood regularized logistic matrix factorization with beta distribution rescoring (NRLMFb), which is an algorithm to correct the score of NRLMF. The Beta distribution is known as a conjugative prior distribution of the Bernoulli distribution and it can reflect an amount of active information to its shape. Therefore, in NRLMFb, the Beta distribution was used for rescoring the NRLMF score. In the evaluation experiment, three types of 10-fold cross validation were performed five times for each of the four datasets (i.e. Nuclear receptor, GPCR, Ion channel, Enzyme) and we measured the average values of AUC and AUPR and 95¥% confidence intervals. As a result, the performance of NRLMFb was shown to exceed that of NRLMF. For this reason, we concluded that NRLMFb improved prediction accuracy as compared to NRLMF.

Requirements
------------
### Python
You need to use Python 3.x for executing this scripts. We recommends that you use Anaconda 2.4.0 to set up python environment. This script was created by using Python 3.5.4. For Python 3.5.4 please refer to the following URL.<br>
https://www.python.org/downloads/release/python-352/<br>

### Python packages
In addition, we use Numpy, scikit-learn (ver. 0.18.1 and above), scipy, pymatbridge (required only when using KBMF 2K) as Python package. For each package please refer to the following URL.<br>
−　Numpy: http://www.numpy.org/<br>
−　scikit-learn: http://scikit-learn.org/stable/<br>
−　scipy: http://www.scipy.org/<br>
−　pymatbridge: http://arokem.github.io/python-matlab-bridge/<br>

### Datasets
The drug-target interaction dataset created by Yamanishi et al. can be downloaded from the following URL.<br>
http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/<br>

Usage
-----

### Examples
Command to execute this script
```shell
$ python PyDTI.py --method="nrlmfb" --dataset="nr" --cvs="1" --data-dir="." --gpmi="delta=1e-100 max_iter=2688 n_init=1" --scoring="auc" --specify-arg=0 --params="params.txt" --log="job1.log"
```


Acknowledgement
---------------
This script was created based on PyDTI developed by Liu et al. PyDTI can be accessed from the following URL. https://github.com/stephenliu0423/PyDTI.git

We also created this script based on the BO-DTI script developed py Ban et al. BO-DTI can be accessed from the following URL. https://github.com/akiyamalab/BO-DTI.git

Contact
-------
These scripts was implemented by Tomohiro Ban.
E-mail: ban@bi.c.titech.ac.jp

Department of Computer Science, School of Computing, Tokyo Institute of Technology, Japan
http://www.bi.cs.titech.ac.jp/

If you have any questions, please feel free to contact the author.

References
----------
In preparetion ...

Copyright © 2018 Akiyama_Laboratory, Tokyo Institute of Technology, All Rights Reserved.

