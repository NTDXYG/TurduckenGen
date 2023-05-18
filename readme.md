### About Dataset

- The original dataset in Lyra is stored in Lyra-python-xxx.csv in the 'dataset/Python' directory.
- The original dataset in Pisces is stored in Pisces-java-xxx.csv in the 'dataset/Java' directory.

All data in Pisces can be compiled, but since no test cases were provided in Lyra, there are also no test cases in Pisces.

Therefore, human study is still needed to evaluate the correctness of the code.

### About Compiler

- We use the pylint library for compilation checking in Lyra. Our version is python3.9, and other dependent packages can be found at https://github.com/LIANGQINGYUAN/Lyra.
- We use Maven for compilation checking in Pisces (because we use Springboot). Our version is jdk1.8, and the relevant pom.xml file is provided.

### About SAT Algorithm

We implemented the SAT algorithm using the open-source [SPT-Code](https://github.com/NougatCA/SPT-Code), with the demo provided in 'SAT-demo.py'.

### HOW TO RUN

Modify the parameters and run 'run.py'.
