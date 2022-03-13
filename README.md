OREA: Domination-Based Ordinal Regression Evolutionary Algorithm

Publication: X. Yu, X. Yao, Y. Wang, L. Zhu, and D. Filev, “Domination-based ordinal regression for expensive multi-objective optimization,” in Proceedings of the 2019 IEEE Symposium Series on Computational Intelligence (SSCI’19), 2019, pp. 2058–2065.


This project implements a surrogate-assisted evolutionary algorithm (SAEA), called OREA, for expensive multi-obejctive optimziation. 
Different from existing SAEAs that using fitness regression or classification surrogates, OREA employs one Kriging model to do ordinal regression. Clearly, the domination-based ordinal relations between evaluated solutions (in an archive) and the current non-dominated solutions are quantified and learned. 


Some Features:

1. One surrogate for multiple objectives (vs fitness regression based surrogates).
2. Most candidate solutions are comparable (vs classification based surrogates).
3. Use a hybird surrogate management strategy.
4. Assisted by reference vectors.


Execution:

Open 'Tester_OREA.py' or 'Tester_OREA.ipynb', set the benchmark testing function you want to run, and choose the number of runs (1-30). The results will be saved in the format: 'results/DTLZ1/Total(10,3)/'. Currently, a set of initial datasets has been uploaded for 10 variable, 3 objective DTLZ testing functions. 
If you want to generate your own initial datasets, set 'init_path == None'.


Contact:

For questions and feedback, please contact yuxunzhao@gmail.com
