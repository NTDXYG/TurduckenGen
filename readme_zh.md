### 关于数据集

- Lyra中的原始数据集在'dataset/Python'目录的Lyra-python-xxx.csv中
- Pisces中的原始数据集在'dataset/Java'目录的Pisces-java-xxx.csv中

Pisces中的所有数据均通过编译，但是由于Lyra中未提供测试用例，Pisces中也是没有测试用例的。

因此，在评估代码的功能正确时依然要依赖human study来完成。

### 关于Compiler

- Lyra我们使用pylint库进行编译检测，我们的版本是python3.9，其余的依赖包参考https://github.com/LIANGQINGYUAN/Lyra
- Pisces我们使用的Maven进行编译检测（因为使用了Springboot），我们的版本是jdk1.8，相干的pom.xml文件已提供。

### 关于SAT算法

借助SAT-Code库开源代码进行实现，demo在SAT-demo.py中。

### HOW TU RUN

修改参数并运行run.py