##Pandas数据处理

### 一 概述

#### 1.1 业务建模流程

- 将业务抽象为分类or回归问题
- 定义标签，得到y
- 选取合适的样本，并匹配出全部的信息作为特征的来源
- 特征工程 + 模型训练 + 模型评价与调优（相互之间可能会有交互）
- 输出模型报告
- 上线与监控

#### 1.2什么是特征

在机器学习的背景下，特征是用来解释现象发生的单个特性或一组特性。 当这些特性转换为某种可度量的形式时，它们被称为特征。

举个例子，假设你有一个学生列表，这个列表里包含每个学生的姓名、学习小时数、IQ和之前考试的总分数。现在，有一个新学生，你知道他/她的学习小时数和IQ，但他/她的考试分数缺失，你需要估算他/她可能获得的考试分数。

在这里，你需要用IQ和study_hours构建一个估算分数缺失值的预测模型。所以，IQ和study_hours就成了这个模型的特征。

#### 1.3 特征工程可能包含的内容

- 基础特征构造
- 数据预处理
- 特征衍生
- 特征筛选

这是一个完整的特征工程流程，但不是唯一的流程，每个过程都有可能会交换顺序，随着学习的加深，大家会慢慢体会到。

### 二 数据预处理

数据预处理是数据分析和数据运营过程中的重要环节，它直接决定了后期所有数据工作的质量和价值输出

- 数据清洗
- 数据转换
- 数据抽样

#### 2.1数据清洗（缺失值，异常值和重复值的处理）

##### 2.1.1缺失值处理

- 数据缺分类
  - 行记录缺失，实际上就是记录丢失
  - 数据列值缺失，数据记录中某些列值空缺，具体表现形态
    - 数据库 Null
    - Python返回对象None
    - Pandas Numpy NaN
    - 个别情况下，部分缺失值会使用空字符串代替
- 缺失值处理方式
  - **直接删除**带有缺失值的行记录（整行删除）或者列字段（整列删除），删除意味着会消减数据特征，不适合直接删除缺失值的情况：
    - 数据记录不完整情况且比例较大（如超过10%），删除会损失过多有用信息。
    - 带有缺失值的数据记录大量存在着明显的数据分布规律或特征
      - 带有缺失值的数据记录的目标标签（即分类中的Label变量）主要集中于某一类或几类，如果删除这些数据记录将使对应分类的数据样本丢失大量特征信息，导致模型过拟合或分类不准确。
  - **填充缺失值** 相对直接删除而言，用适当方式填充缺失值，形成完整的数据记录是更加常用的缺失值处理方式。常用的填充方法如下：
    - 统计法
      - 对于数值型的数据，使用均值、加权均值、中位数等方填充
      - 对于分类型数据，使用类别众数最多的值填充。
    - 模型法：更多时候我们会基于已有的其他字段，将缺失字段作为目标变量进行预测，从而得到最为可能的补全值。如果带有缺失值的列是数值变量，采用回归模型补全；如果是分类变量，则采用分类模型补全。
    - 专家补全：对于少量且具有重要意义的数据记录，专家补足也是非常重要的一种途径。
    - 其他方法：例如随机法、特殊值法、多重填补等。
  - **真值转换法** 承认缺失值的存在，并且把数据缺失也作为数据分布规律的一部分，将变量的实际值和缺失值都作为输入维度参与后续数据处理和模型计算中。但是变量的实际值可以作为变量值参与模型计算，而缺失值通常无法参与运算，因此需要对缺失值进行真值转换。
    - 以用户性别字段为例，男 女 未知
  - **不处理** 数据分析和建模应用中很多模型对于缺失值有容忍度或灵活的处理方法，因此在预处理阶段可以不做处理。常见的能够自动处理缺失值的模型包括：KNN、决策树和随机森林、神经网络和朴素贝叶斯
    - KNN 模型中缺失值不参与距离计算
- 缺失值处理套路
  - 找到缺失值
  - 分析缺失值在整体样本中的占比
  - 选择合适的方式处理缺失值
- 注意 默认值的问题，需要跟开发人员沟通

##### 2.1.2异常值（极值）处理

- 处于特定分布区域或范围之外的数据通常会被定义为异常或“噪音”。产生数据“噪音”的原因很多，例如业务运营操作、数据采集问题、数据同步问题等。对异常数据进行处理前，需要先辨别出到底哪些是真正的数据异常。从数据异常的状态看分为两种：
  - 由于业务特定运营动作产生的，正常反映业务状态，而不是数据本身的异常规律。
  - 不是由于特定的业务动作引起的，而是客观地反映了数据本身分布异常

- 大多数情况下，异常值都会在数据的预处理过程中被认为是噪音而剔除，以避免其对总体数据评估和分析挖掘的影响。但在以下几种情况下，我们无须对异常值做抛弃处理。
  - 异常值由运营活动导致，正常反映了业务运营结果
    - 公司的A商品正常情况下日销量为1000台左右。由于昨日举行优惠促销活动导致总销量达到10000台，由于后端库存备货不足导致今日销量又下降到100台。在这种情况下，10000台和100台都正确地反映了业务运营的结果，而非数据异常案例。
  - 异常检测模型
    - 围绕异常值展开的分析工作，如异常客户（羊毛党）识别，作弊流量检测，信用卡诈骗识别等
  - 对异常值不敏感的数据模型
    - 如决策树
- 处理方式
  - 保留
  - 删除
  - 用统计量或预测量进行替换

##### 2.1.3 重复值处理

- 数据去重是处理重复值的主要方法，但如下几种情况慎重去重
  - 样本不均衡时，故意重复采样的数据
    - 分类模型，某个分类训练数据过少，可以采取简单复制样本的方法来增加样本数量
  - 重复记录用户检测业务规则问题
    - 事务型数据，尤其与钱相关的业务场景下出现重复数据时，如重复订单，重复出库申请...

##### 2.1.4 实战 Python数据清洗

- 缺失值处理

  ```python
  #用到pandas的api
  dataframe.isnull() #判断是否有缺失值
  dataframe.dropna() #删除缺失值
  dataframe.fillna()#填充缺失值
  ```

  - 代码

    ######生成缺失数据

    ```python
    import pandas as pd  # 导入pandas库
    import numpy as np  # 导入numpy库
    from sklearn.impute import SimpleImputer  # 导入sklearn中SimpleImputer库
    
    df = pd.DataFrame(np.random.randn(6, 4), columns=['col1', 'col2', 'col3', 'col4'])  # 生成一份数据
    df.iloc[1:2, 1] = np.nan  # 增加缺失值
    df.iloc[4, 3] = np.nan  # 增加缺失值
    print(df)
    ```

    ######查看是否包含缺失

    ```python
    nan_all = df.isnull()  # 获得所有数据框中的N值
    print(nan_all)  # 打印输出
    ```

    ```shell
    #输出结果
    col1   col2   col3   col4
    0  False  False  False  False
    1  False   True  False  False
    2  False  False  False  False
    3  False  False  False  False
    4  False  False  False   True
    5  False  False  False  False
    ```

    ######查看哪些列有缺失

    ```python
    nan_col1 = df.isnull().any()  # 获得含有NA的列
    nan_col2 = df.isnull().all()  # 获得全部为NA的列
    print(nan_col1)  # 打印输出
    print(nan_col2)  # 打印输出
    ```

    ```shell
    col1    False
    col2     True
    col3    False
    col4     True
    dtype: bool
    col1    False
    col2    False
    col3    False
    col4    False
    dtype: bool
    ```

    ######删除缺失值

    ```python
    df2 = df.dropna()  # 直接丢弃含有NA的行记录
    print(df2)  # 打印输出
    ```

    ######使用sklearn填充缺失值

    ```python
    nan_model = SimpleImputer(missing_values=np.nan, strategy='mean')  # 建立替换规则：将值为NaN的缺失值以均值做替换
    nan_result = nan_model.fit_transform(df)  # 应用模型规则
    print(nan_result)  # 打印输出
    ```

    ######使用pandas填充缺失值

    ```python
    nan_result_pd1 = df.fillna(method='backfill')  # 用后面的值替换缺失值
    nan_result_pd2 = df.fillna(method='bfill', limit=1)  # 用后面的值替代缺失值,限制每列只能替代一个缺失值
    nan_result_pd3 = df.fillna(method='pad')  # 用前面的值替换缺失值
    nan_result_pd4 = df.fillna(0)  # 用0替换缺失值
    nan_result_pd5 = df.fillna({'col2': 1.1, 'col4': 1.2})  # 用不同值替换不同列的缺失值
    nan_result_pd6 = df.fillna(df.mean()['col2':'col4'])  # 用平均数代替,选择各自列的均值替换缺失值
    # 打印输出
    print(nan_result_pd1)  # 打印输出
    print(nan_result_pd2)  # 打印输出
    print(nan_result_pd3)  # 打印输出
    print(nan_result_pd4)  # 打印输出
    print(nan_result_pd5)  # 打印输出
    print(nan_result_pd6)  # 打印输出
    ```

- 异常值处理

  - 用到的API

    ```python
    dataframe.mean()  #计算平局值
    dataframe.std()   #计算标准差
    ```

  - 判断异常值方法：Z-Score 

    计算公式 Z = X-μ/σ  其中μ为总体平均值，X-μ为离均差，σ表示标准差。z的绝对值表示在标准差范围内的原始分数与总体均值之间的距离。当原始分数低于平均值时，z为负，以上为正。

  - 代码

    ```python
    import pandas as pd  # 导入pandas库
    # 生成异常数据
    df = pd.DataFrame({'col1': [1, 120, 3, 5, 2, 12, 13],
                       'col2': [12, 17, 31, 53, 22, 32, 43]})
    print(df)  # 打印输出
    ```

    ```shell
    #输出显示
       col1  col2
    0     1    12
    1   120    17
    2     3    31
    3     5    53
    4     2    22
    5    12    32
    6    13    43
    ```

    ```python
    # 通过Z-Score方法判断异常值
    df_zscore = df.copy()  # 复制一个用来存储Z-score得分的数据框
    cols = df.columns  # 获得数据框的列名
    for col in cols:  # 循环读取每列
        df_col = df[col]  # 得到每列的值
        z_score = (df_col - df_col.mean()) / df_col.std()  # 计算每列的Z-score得分
        df_zscore[col] = z_score.abs() > 2.2  # 判断Z-score得分是否大于2.2，如果是则是True，否则为False
    print(df_zscore)  # 打印输出
    ```

    ```shell
    #输出显示
        col1   col2
    0  False  False
    1   True  False
    2  False  False
    3  False  False
    4  False  False
    5  False  False
    6  False  False
    ```

    ```python
    # 删除异常值所在的行
    df_drop_outlier = df[df_zscore['col1'] == False]
    print(df_drop_outlier)
    ```

    ```shell
    #输出显示
       col1  col2
    0     1    12
    2     3    31
    3     5    53
    4     2    22
    5    12    32
    6    13    43
    ```
    
  - 异常值处理的关键：如何判断异常

    - 有固定该业务规则的直接利用业务规则
  - 没有固定业务规则的，可以使用数学模型进行判断，如正态分布的标准差范围，分位数法等

- 数据去重

  - 用到的API

    ```python
    dataframe.duplicated()  # 判断重复数据记录
    dataframe.drop_duplicates() # 删除数据记录中所有列值相同的记录
    ```

  - 代码

    ```python
    import pandas as pd  # 导入pandas库
    # 生成重复数据
    data1, data2, data3, data4 = ['a', 3], ['b', 2], ['a', 3], ['c', 2]
    df = pd.DataFrame([data1, data2, data3, data4], columns=['col1', 'col2'])
    print(df)
    ```

    ```shell
    #显示效果
      col1  col2
    0    a     3
    1    b     2
    2    a     3
    3    c     2
    ```

    ```python
    # 判断重复数据
    isDuplicated = df.duplicated()  # 判断重复数据记录
    print(isDuplicated)  # 打印输出
    ```

    ```shell
    #显示效果
    0    False
    1    False
    2     True
    3    False
    dtype: bool
    ```

    ```python
    # 删除重复值
    print(df.drop_duplicates())  # 删除数据记录中所有列值相同的记录
    print(df.drop_duplicates(['col1']))  # 删除数据记录中col1值相同的记录
    print(df.drop_duplicates(['col2']))  # 删除数据记录中col2值相同的记录
    print(df.drop_duplicates(['col1', 'col2']))  # 除数据记录中指定列（col1/col2）值相同的记录
    ```

    ```shell
      col1  col2
    0    a     3
    1    b     2
    3    c     2
      col1  col2
    0    a     3
    1    b     2
    3    c     2
      col1  col2
    0    a     3
    1    b     2
      col1  col2
    0    a     3
    1    b     2
    3    c     2
    ```

#### 2.2 数值型数据的处理

##### 2.2.1标准化&归一化

- 数据标准化是一个常用的数据预处理操作，目的是处理不同规模和量纲的数据，使其缩放到相同的数据区间和范围，以减少规模、特征、分布差异等对模型的影响。

- 标准化（Z-Score）

  - Z-Score标准化是基于原始数据的均值和标准差进行的标准化，假设原转换的数据为x，新数据为x′，那么x'=(x-mean)/std，其中mean和std为x所在列的均值和标准差。
  - 这种方法适合大多数类型的数据，也是很多工具的默认标准化方法。标准化之后的数据是以0为均值，方差为1的正态分布。但是Z-Score方法是一种中心化方法，会改变原有数据的分布结构，不适合对稀疏数据做处理。

- 归一化（Max-Min）

  - Max-Min标准化方法是对原始数据进行线性变换，假设原转换的数据为x，新数据为x′，那么x'=(x-min)/(max-min)，其中min和max为x所在列的最小值和最大值。
  - 这种标准化方法的应用非常广泛，得到的数据会完全落入[0，1]区间内（Z-Score则没有类似区间）。这种方法能使数据归一化而落到一定的区间内，同时还能较好地保持原有数据结构。

- 代码

  ```python
  import numpy as np
  from sklearn import preprocessing
  import matplotlib.pyplot as plt
  data = np.loadtxt('data6.txt', delimiter='\t')  # 读取数据
  ```

  - 标准化

  ```python
  # Z-Score标准化
  zscore_scaler = preprocessing.StandardScaler()  # 建立StandardScaler对象
  data_scale_1 = zscore_scaler.fit_transform(data)  # StandardScaler标准化处理
  ```

  - 归一化Max-Min

  ```python
  minmax_scaler = preprocessing.MinMaxScaler()  # 建立MinMaxScaler模型对象
  data_scale_2 = minmax_scaler.fit_transform(data)  # MinMaxScaler标准化处理
  ```

  - 展示结果

  ```python
  # 展示多网格结果
  data_list = [data, data_scale_1, data_scale_2]  # 创建数据集列表
  color_list = ['black', 'green', 'blue']  # 创建颜色列表
  merker_list = ['o', ',', '+']  # 创建样式列表
  title_list = ['source data', 'zscore_scaler', 'minmax_scaler']  # 创建标题列表
  plt.figure(figsize=(13, 3))
  for i, data_single in enumerate(data_list):  # 循环得到索引和每个数值
      plt.subplot(1, 3, i + 1)  # 确定子网格
      plt.scatter(data_single[:, :-1], data_single[:, -1], s=10, marker=merker_list[i],
                  c=color_list[i])  # 自网格展示散点图
      plt.title(title_list[i])  # 设置自网格标题
  plt.suptitle("raw data and standardized data")  # 设置总标题
  ```

  ![image-20191106170906033](pics\plt_scaler.png)

##### 2.2.2 离散化/分箱/分桶

- 离散化，就是把无限空间中有限的个体映射到有限的空间中。数据离散化操作大多是针对连续数据进行的，处理之后的数据值域分布将从连续属性变为离散属性，这种属性一般包含2个或2个以上的值域。离散化处理的必要性如下：

  - 节约计算资源，提高计算效率。
  - 算法模型（尤其是分类模型）的计算需要。虽然很多模型，例如决策树可以支持输入连续型数据，但是决策树本身会先将连续型数据转化为离散型数据，因此离散化转换是一个必要步骤。
  - 增强模型的稳定性和准确度。数据离散化之后，处于异常状态的数据不会明显地突出异常特征，而是会被划分为一个子集中的一部分，因此异常数据对模型的影响会大大降低，尤其是基于距离计算的模型（例如K均值、协同过滤等）效果明显。
  - 特定数据处理和分析的必要步骤，尤其在图像处理方面应用广泛。大多数图像做特征检测（以及其他基于特征的分析）时，都需要先将图像做二值化处理，二值化也是离散化的一种。
  - 模型结果应用和部署的需要。如果原始数据的值域分布过多，或值域划分不符合业务逻辑，那么模型结果将很难被业务理解并应用。
  - 离散化通常针对连续数据进行处理，但是在很多情况下也可以针对已经是离散化的数据进行处理。这种场景一般是离散数据本身的划分过于复杂、琐碎，甚至不符合业务逻辑，需要进一步做数据聚合或重新划分。

- 常见离散化场景

  - 针对时间数据的离散化
  - 针对多值离散数据的离散化：要进行离散化处理的数据本身不是数值型数据，而是分类或顺序数据。
    - 例如，用户收入变量的值原来可能划分为10个区间，根据新的建模需求，只需要划分为4个区间，那么就需要对原来的10个区间进行合并。
  - 多值离散数据要进行离散化还有可能是划分的逻辑有问题，需要重新划分。这种问题通常都是由于业务逻辑的变更，导致在原始数据中存在不同历史数据下的不同值域定义。
    - 例如，用户活跃度变量的值，原来分为高价值、中价值和低价值3个类别。根据业务发展的需要，新的用户活跃度变量的值定义为：高价值、中价值、低价值和负价值。此时需要对不同类别的数据进行统一规则的离散化处理。
  - 针对连续数据的离散化是主要的离散化应用，在分类或关联分析中应用尤其广泛。这些算法的结果以类别或属性标识为基础，而非数值标记。
    - 连续数据的离散化结果可以分为两类：
      - 一类是将连续数据划分为特定区间的集合，例如{(0，10]，(10，20]，(20，50]，(50，100]}；
      - 一类是将连续数据划分为特定类，例如类1、类2、类3。

    - 常见实现针对连续数据化离散化的方法如下。
      - 分位数法：使用四分位、五分位、十分位等分位数进行离散化处理，这种方法简单易行。
      - 距离区间法：可使用等距区间或自定义区间的方式进行离散化，这种操作更加灵活且能满足自定义需求。另外该方法（尤其是等距区间）可以较好地保持数据原有的分布。
      - 频率区间法：将数据按照不同数据的频率分布进行排序，然后按照等频率或指定频率离散化，这种方法会把数据变换成均匀分布。好处是各区间的观察值是相同的，不足是已经改变了原有数据的分布状态。
      - 聚类法：例如使用K均值将样本集分为多个离散化的簇。
  - 针对连续数据的二值化
    - 在很多场景下，我们可能需要将变量特征进行二值化操作：每个数据点跟阈值比较，大于阈值设置为某一固定值（例如1），小于阈值设置为某一固定值（例如0），然后得到一个只拥有两个值域的二值化数据集。

- 离散化练习

  ```python
  import pandas as pd
  from sklearn.cluster import KMeans
  from sklearn import preprocessing
  # 读取数据
  df = pd.read_table('data7.txt', names=['id', 'amount', 'income', 'datetime', 'age'])  # 读取数据文件
  print(df.head(5))  # 打印输出前5条数据
  ```

  - 终端显示

  ```shell
        id  amount  income             datetime    age
  0  15093    1390   10.40  2017-04-30 19:24:13   0-10
  1  15062    4024    4.68  2017-04-27 22:44:59  70-80
  2  15028    6359    3.84  2017-04-27 10:07:55  40-50
  3  15012    7759    3.70  2017-04-04 07:28:18  30-40
  4  15021     331    4.25  2017-04-08 11:14:00  70-80
  ```

  - 针对时间数据的离散化

  ```python
  # 针对时间数据的离散化
  df['datetime'] = list(map(pd.to_datetime,df['datetime'])) # 将时间转换为datetime格式
  df['datetime'] = [i.weekday() for i in df['datetime']]# 离散化为周几
  print(df.head(5))  # 打印输出前5条数据
  ```

  - 终端显示

  ```python
        id  amount  income  datetime    age
  0  15093    1390   10.40         6   0-10
  1  15062    4024    4.68         3  70-80
  2  15028    6359    3.84         3  40-50
  3  15012    7759    3.70         1  30-40
  4  15021     331    4.25         5  70-80
  ```

  - 针对连续数据的离散化：自定义分箱区间实现离散化

  ```python
  bins = [0, 200, 1000, 5000, 10000]  # 自定义区间边界
  df['amount1'] = pd.cut(df['amount'], bins)  # 使用边界做离散化
  print(df.head(5))  # 打印输出前5条数据
  ```
  
  - 终端显示
  
  ```shell
      id  amount  income  datetime  age2        amount1
  0  15093    1390   10.40         6  0-40   (1000, 5000]
1  15064    7952    4.40         0  0-40  (5000, 10000]
  2  15080     503    5.72         5  0-40    (200, 1000]
  3  15068    1668    3.19         5  0-40   (1000, 5000]
  4  15019    6710    3.20         0  0-40  (5000, 10000]
  ```
  
  - 针对连续数据的离散化：使用聚类法实现离散化
  
  ```python
data = df['amount']  # 获取要聚类的数据，名为amount的列
  data_reshape = data.values.reshape((data.shape[0], 1))  # 转换数据形状
model_kmeans = KMeans(n_clusters=4, random_state=0)  # 创建KMeans模型并指定要聚类数量
  keames_result = model_kmeans.fit_predict(data_reshape)  # 建模聚类
  df['amount2'] = keames_result  # 新离散化的数据合并到原数据框
  print(df.head(5))  # 打印输出前5条数据
  ```
  
- 终端显示
  
```shell
        id  amount  income  datetime  age2        amount1  amount2
  0  15093    1390   10.40         6  0-40   (1000, 5000]        2
  1  15064    7952    4.40         0  0-40  (5000, 10000]        1
  2  15080     503    5.72         5  0-40    (200, 1000]        2
  3  15068    1668    3.19         5  0-40   (1000, 5000]        2
  4  15019    6710    3.20         0  0-40  (5000, 10000]        1
```

- 针对连续数据的离散化
  
```python
  df['amount3'] = pd.qcut(df['amount'], 4, labels=['bad', 'medium', 'good', 'awesome']) 
  df = df.drop('amount', 1)  # 丢弃名为amount的列
  print(df.head(5))  # 打印输出前5条数据
```

  - 终端显示

  ```shell
      id  income  datetime  age2        amount1  amount2  amount3
  0  15093   10.40         6  0-40   (1000, 5000]        2      bad
1  15064    4.40         0  0-40  (5000, 10000]        1  awesome
  2  15080    5.72         5  0-40    (200, 1000]        2      bad
  3  15068    3.19         5  0-40   (1000, 5000]        2      bad
  4  15019    3.20         0  0-40  (5000, 10000]        1  awesome
  ```

  - 针对连续数据的二值化

  ```python
binarizer_scaler = preprocessing.Binarizer(threshold=df['income'].mean())  # 建立Binarizer模型对象
  income_tmp = binarizer_scaler.fit_transform(df[['income']])  # Binarizer标准化转换
income_tmp.resize(df['income'].shape)  # 转换数据形状
  df['income'] = income_tmp  # Binarizer标准化转换
  print(df.head(5))  # 打印输出前5条数据  
  ```

  - 终端显示

  ```shell
      id  income  datetime  age2        amount1  amount2  amount3
  0  15093     1.0         6  0-40   (1000, 5000]        2      bad
  1  15064     1.0         0  0-40  (5000, 10000]        1  awesome
  2  15080     1.0         5  0-40    (200, 1000]        2      bad
  3  15068     0.0         5  0-40   (1000, 5000]        2      bad
  4  15019     0.0         0  0-40  (5000, 10000]        1  awesome
  ```

  

#### 2.3 分类数据的处理

- 在数据建模过程中，很多算法无法直接处理非数值型的变量
  
- KMeans算法基于距离的相似度计算，而字符串则无法直接计算距离。
  
- 即使算法本身支持，很多算法实现包也无法直接基于字符串做矩阵运算
  
- Numpy以及基于Numpy的sklearn。虽然这些库允许直接使用和存储字符串型变量，但却无法发挥矩阵计算的优势
  
- 举例：

  - 性别中的男和女
  - 颜色中的红、黄和蓝
  - 用户的价值度分为高、中、低
  - 学历分为博士、硕士、学士、大专、高中

- 处理方法

  - 将字符串表示的分类特征转换成数值类型，一般用one-hot编码表示，方便建模处理

- 代码：

  ```python
  import pandas as pd  # 导入pandas库
  from sklearn.preprocessing import OneHotEncoder  # 导入库
  # 生成数据
  df = pd.DataFrame({'id': [3566841, 6541227, 3512441],
                     'sex': ['male', 'Female', 'Female'],
                     'level': ['high', 'low', 'middle'],
                     'score': [1, 2, 3]})
  print(df)  # 打印输出原始数据框
  ```

  - 输出

  ```shell
          id     sex   level  score
  0  3566841    male    high      1
  1  6541227  Female     low      2
  2  3512441  Female  middle      3
  ```

  ```python
  # 使用sklearn进行标志转换
  # 拆分ID和数据列
  id_data = df[['id']]  # 获得ID列
  raw_convert_data = df.iloc[:, 1:]  # 指定要转换的列
  print(raw_convert_data)
  # 将数值型分类向量转换为标志变量
  model_enc = OneHotEncoder()  # 建立标志转换模型对象（也称为哑编码对象）
  df_new2 = model_enc.fit_transform(raw_convert_data).toarray()  # 标志转换
  # 合并数据
  df_all = pd.concat((id_data, pd.DataFrame(df_new2)), axis=1)  # 重新组合为数据框
  print(df_all)  # 打印输出转换后的数据框
  ```

  - 输出

  ```python
        sex   level  score
  0    male    high      1
  1  Female     low      2
  2  Female  middle      3
          id    0    1    2    3    4    5    6    7
  0  3566841  0.0  1.0  1.0  0.0  0.0  1.0  0.0  0.0
  1  6541227  1.0  0.0  0.0  1.0  0.0  0.0  1.0  0.0
  2  3512441  1.0  0.0  0.0  0.0  1.0  0.0  0.0  1.0
  ```

  - 使用pandas的get_dummuies ,词方法只会对非数值类型的数据做转换

  ```python
  df_new3 = pd.get_dummies(raw_convert_data)
  df_all2 = pd.concat((id_data, df_new3), axis=1)  # 重新组合为数据框
  print(df_all2)  # 打印输出转换后的数据框
  ```

  - 输出

  ```python
          id  score  sex_Female  sex_male  level_high  level_low  level_middle
  0  3566841      1           0         1           1          0             0
  1  6541227      2           1         0           0          1             0
  2  3512441      3           1         0           0          0             1
  ```

#### 2.4 时间类型数据的处理

- 数据中包含日期时间类型的数据可以通过pandas的 to_datetime 转换成datetime类型，方便提取各种时间信息

```python
import pandas as pd
car_sales = pd.read_csv('car_data.csv')
car_sales.head()
'''
       date_t     cnt
0  2012-12-31     NaN
1  2013-01-01     NaN
2  2013-01-02    68.0
3  2013-01-03    36.0
4  2013-01-04  5565.0
'''
car_sales.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1512 entries, 0 to 1511
Data columns (total 2 columns):
date_t    1512 non-null object
cnt       1032 non-null float64
dtypes: float64(1), object(1)
memory usage: 23.7+ KB
'''
car_sales.describe()
'''
              cnt
count  1032.000000
mean   1760.124031
std    1153.164214
min      12.000000
25%    1178.750000
50%    1774.000000
75%    2277.750000
max    7226.000000
'''
car_sales['date_t'].dtype
#dtype('O')

car_sales.loc[:,'date'] = pd.to_datetime(car_sales['date_t'])
car_sales.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1512 entries, 0 to 1511
Data columns (total 3 columns):
date_t    1512 non-null object
cnt       1032 non-null float64
date      1512 non-null datetime64[ns]
dtypes: datetime64[ns](1), float64(1), object(1)
memory usage: 35.5+ KB
'''
```

- 取出关键时间信息

```python
# 取出几月份
car_sales.loc[:,'month'] = car_sales['date'].dt.month
# 取出来是几号
car_sales.loc[:,'dom'] = car_sales['date'].dt.day
# 取出一年当中的第几天
car_sales.loc[:,'doy'] = car_sales['date'].dt.dayofyear
# 取出星期几
car_sales.loc[:,'dow'] = car_sales['date'].dt.dayofweek
car_sales.head()
'''
       date_t     cnt
0  2012-12-31     NaN
1  2013-01-01     NaN
2  2013-01-02    68.0
3  2013-01-03    36.0
4  2013-01-04  5565.0
'''
```

#### 2.5 样本类别分布不均衡

- 什么是样本类别分布不均衡

  - 样本类别分布不均衡主要出现在与分类相关的建模问题上，不均衡指的是不同类别的样本量差异非常大。
  - 样本分布不均衡将导致样本量少的分类所包含的特征过少，建立的模型容易过拟合。
  - 如果不同分类间的样本量差异超过10倍就需要引起警觉，超过20倍就一定要想法解决了。

- 样本分布不均衡易出现场景

  - 异常检测：大多数企业中的异常个案都是少量的，比如恶意刷单、黄牛订单、信用卡欺诈、电力窃电、设备故障等。这些数据样本所占的比例通常是整体样本中很少的一部分。以信用卡欺诈为例，刷实体信用卡欺诈的比例一般在0.1%以内
  - 客户流失：大型企业的流失客户相对于整体客户通常是少量的，尤其对于具有垄断地位的行业巨擘，例如电信、石油、网络运营商等更是如此
  - 偶发事件：罕见事件与异常检测类似，都属于发生个案较少的情况；但不同点在于异常检测通常都有是预先定义好的规则和逻辑，并且大多数异常事件都对会企业运营造成负面影响，因此针对异常事件的检测和预防非常重要；但罕见事件则无法预判，并且也没有明显的积极和消极影响倾向。例如，由于某网络大V无意中转发了企业的一条趣味广告，导致用户流量明显提升便属于此类。
  - 低频事件：这种事件是预期或计划性事件，但是发生频率非常低。例如，每年一次的“双11”购物节一般都会产生较高的销售额，但放到全年来看，这一天的销售额占比很可能只有不到1%，尤其对于很少参与活动的公司而言，这种情况更加明显。这种就属于典型的低频率事件。

- 如何处理样本分布不均衡问题

  - 抽样：抽样是解决样本分布不均衡相对简单且常用的方法，包括过抽样和欠抽样两种。
    - 1）过抽样：又称上采样（over-sampling），其通过增加分类中少数类样本的数量来实现样本均衡，最直接的方法是简单复制少数类样本以形成多条记录。这种方法的缺点是，如果样本特征少则可能导致过拟合的问题。经过改进的过抽样方法会在少数类中加入随机噪声、干扰数据，或通过一定规则产生新的合成样本，例如SMOTE算法。
    - 2）欠抽样：又称下采样（under-sampling），其通过减少分类中多数类样本的数量来实现样本均衡，最直接的方法是随机去掉一些多数类样本来减小多数类的规模。缺点是会丢失多数类样本中的一些重要信息。
      总体上，过抽样和欠抽样更适合大数据分布不均衡的情况，尤其是过抽样方法，应用极为广泛。
  - 正负样本的惩罚权重：在算法实现过程中，对于分类中不同样本数量的类别分别赋予不同的权重（一般思路分类中的小样本量类别权重高，大样本量类别权重低），然后进行计算和建模。
    - 很多模型和算法中都有基于类别参数的调整设置，以scikit-learn中的SVM为例，通过在class_weight：{dict，'balanced'}中针对不同类别来手动指定权重，如果使用其默认的方法balanced，那么SVM会将权重设置为与不同类别样本数量呈反比的权重来进行自动均衡处理，计算公式如下：n_samples/(n_classes*np.bincount(y))
  - 组合/集成：每次生成训练集时使用所有分类中的小样本量，同时从分类中的大样本量中随机抽取数据来与小样本量合并构成训练集，这样反复多次会得到很多训练集和训练模型。最后在应用时，使用组合方法（例如投票、加权投票等）产生分类预测结果。
    - 数据集中的正、负例的样本分别为100条和10000条，比例为1：100。此时可以将负例样本（类别中的大量样本集）随机分为100份（当然也可以分更多），每份100条数据；然后每次形成训练集时使用所有的正样本（100条）和随机抽取的负样本（100条）形成新的数据集。如此反复可以得到100个训练集和对应的训练模型。
      这种解决问题的思路类似于随机森林。在随机森林中，虽然每个小决策树的分类能力很弱，但是通过大量的“小树”组合形成的“森林”具有良好的模型预测能力。

- Python处理样本不均衡案例

  ```python
  import pandas as pd
  from imblearn.over_sampling import SMOTE  # 过抽样处理库SMOTE
  from imblearn.under_sampling import RandomUnderSampler  # 欠抽样处理库RandomUnderSampler
  # 导入数据文件
  df = pd.read_table('data2.txt', sep=' ',
                     names=['col1', 'col2', 'col3', 'col4', 'col5', 'label'])  # 读取数据文件
  x, y = df.iloc[:, :-1],df.iloc[:, -1]  # 切片，得到输入x，标签y
  groupby_data_orgianl = df.groupby('label').count()  # 对label做分类汇总
  print(groupby_data_orgianl)  # 打印输出原始数据集样本分类分布
  ```

  - 终端显示

  ```shell
         col1  col2  col3  col4  col5
  label                              
  0.0     942   942   942   942   942
  1.0      58    58    58    58    58
  ```

  -    

  ```python
  # 使用SMOTE方法进行过抽样处理
  model_smote = SMOTE()  # 建立SMOTE模型对象
  x_smote_resampled, y_smote_resampled = model_smote.fit_sample(x, y)  # 输入数据并作过抽样处理
  x_smote_resampled = pd.DataFrame(x_smote_resampled,
                                   columns=['col1', 'col2', 'col3', 'col4', 'col5'])  # 将数据转换为数据框并命名列名
  y_smote_resampled = pd.DataFrame(y_smote_resampled, columns=['label'])  # 将数据转换为数据框并命名列名
  smote_resampled = pd.concat([x_smote_resampled, y_smote_resampled], axis=1)  # 按列合并数据框
  groupby_data_smote = smote_resampled.groupby('label').count()  # 对label做分类汇总
  print(groupby_data_smote)  # 打印输出经过SMOTE处理后的数据集样本分类分布
  ```

  - 终端显示

  ```shell
         col1  col2  col3  col4  col5
  label                              
  0.0     942   942   942   942   942
  1.0     942   942   942   942   942
  ```

  -  

  ```python
  # 使用RandomUnderSampler方法进行欠抽样处理
  model_RandomUnderSampler = RandomUnderSampler()  # 建立RandomUnderSampler模型对象
  x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled = model_RandomUnderSampler.fit_sample(
      x,
      y)  # 输入数据并作欠抽样处理
  x_RandomUnderSampler_resampled = pd.DataFrame(x_RandomUnderSampler_resampled,
                                                columns=['col1', 'col2', 'col3', 'col4',
                                                         'col5'])  # 将数据转换为数据框并命名列名
  y_RandomUnderSampler_resampled = pd.DataFrame(y_RandomUnderSampler_resampled,
                                                columns=['label'])  # 将数据转换为数据框并命名列名
  RandomUnderSampler_resampled = pd.concat(
      [x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled],
      axis=1)  # 按列合并数据框
  groupby_data_RandomUnderSampler = RandomUnderSampler_resampled.groupby(
      'label').count()  # 对label做分类汇总
  print(groupby_data_RandomUnderSampler)  # 打印输出经过RandomUnderSampler处理后的数据集样本分类分布
  ```

  - 终端显示

  ```python
         col1  col2  col3  col4  col5
  label                              
  0.0      58    58    58    58    58
  1.0      58    58    58    58    58
  ```

#### 2.6 数据抽样

- 抽样是从整体样本中通过一定的方法选择一部分样本。抽样是数据处理的基本步骤之一，也是科学实验、质量检验、社会调查普遍采用的一种经济有效的工作和研究方法。

- 为什么要抽样

  - 数据量太大，全量计算对硬件要求太高
  - 数据采集限制。很多时候抽样从数据采集端便已经开始，例如做社会调查
  - 时效性要求。抽样带来的是以局部反映全局的思路，如果方法正确，可以以极小的数据计算量来实现对整体数据的统计分析，在时效性上会大大增强。

- 数据抽样的现实意义

  - 通过抽样来实现快速的概念验证
  - 通过抽样可以解决样本不均衡问题（欠抽样、过抽样以及组合/集成的方法）
  - 通过抽样可以解决无法实现对全部样本覆盖的数据分析场景（市场研究、客户线下调研、产品品质检验）

- 如何进行抽样

  - 抽样方法从整体上分为非概率抽样和概率抽样两种

    - 非概率抽样不是按照等概率的原则进行抽样，而是根据人类的主观经验和状态进行判断

    - 概率抽样则是以数学概率论为基础，按照随机的原则进行抽样。

      （1）简单随机抽样
      （2）等距抽样
      等距抽样是先将总体中的每个个体按顺序编号，然后计算出抽样间隔，再按照固定抽样间隔抽取个体。这种操作方法易于理解、简便易行，但当总体样本的分布呈现明显的分布规律时容易产生偏差，例如增减趋势、周期性规律等。该方法适用于个体分布均匀或呈现明显的均匀分布规律，无明显趋势或周期性规律的数据。（3）分层抽样

  - 常用概率抽样方法

    - 简单随机抽样
      - 按等概率原则直接从总样本中抽取n个样本
      - 简单、易于操作，但是它并不能保证样本能完美代表总体
      - 基本前提是所有样本个体都是等概率分布的，但真实情况却是多数样本都不是或无法判断是否是等概率分布的
      - 在简单随机抽样中，得到的结果是不重复的样本集，还可以使用有放回的简单随机抽样，这样得到的样本集中会存在重复数据。该方法适用于个体分布均匀的场景
    - 等距抽样
      - 等距抽样是先将所有样本按顺序编号，然后按照固定抽样间隔抽取个体
      - 该方法适用于个体分布均匀或呈现明显的均匀分布规律，无明显趋势或周期性规律的数据
      - 当总体样本的分布呈现明显的分布规律时容易产生偏差，例如增减趋势、周期性规律等。
  
- 抽样需要注意的几个问题

  - 数据抽样要能反映运营背景
    - 数据时效性问题：使用过时的数据（例如1年前的数据）来分析现在的运营状态。
    - 不能缺少关键因素数据：没有将运营分析涉及的主要因素所产生的数据放到抽样数据中，导致无法根据主要因素产生有效结论，模型效果差，例如抽样中没有覆盖大型促销活动带来的销售增长。
    - 必须具备业务随机性：有意/无意多抽取或覆盖特定数据场景，使得数据明显趋向于特定分布规律，例如在做社会调查时不能使用北京市的抽样数据来代表全国。
    - 业务增长性：在成长型公司中，公司的发展不都是呈现线性趋势的，很多时候会呈现指数趋势。这时需要根据这种趋势来使业务满足不同增长阶段的分析需求，而不只是集中于增长爆发区间。
    - 数据来源的多样性：只选择某一来源的数据做抽样，使得数据的分布受限于数据源。例如在做各分公司的销售分析时，仅将北方大区的数据纳入其中做抽样，而忽视了其他大区的数据，其结果必然有所偏颇。
    - 业务数据可行性问题：很多时候，由于受到经费、权限、职责等方面的限制，在数据抽样方面无法按照数据工作要求来执行，此时要根据运营实际情况调整。这点往往被很多数据工作者忽视。
  - 数据抽样要能满足数据分析和建模需求
    - 抽样样本量的问题（数据量与模型效果关系）
      - 以时间为维度分布的，至少包含一个能满足预测的完整业务周期
      - 做预测（包含分类和回归）分析建模的，需要考虑特征数量和特征值域（非数值型）的分布，通常数据记录数要同时是特征数量和特征值域的100倍以上。例如数据集有5个特征，假如每个特征有2个值域，那么数据记录数需要至少在1000（100×5×2）条以上。
      - 做关联规则分析建模的，根据关联前后项的数量（每个前项或后项可包含多个要关联的主体，例如品牌+商品+价格关联），每个主体需要至少1000条数据。例如只做单品销售关联，那么单品的销售记录需要在1000条以上；如果要同时做单品+品牌的关联，那么需要至少2000条数据。
      - 对于异常检测类分析建模的，无论是监督式还是非监督式建模，由于异常数据本来就是小概率分布的，因此异常数据记录一般越多越好。
    - 抽样样本在不同类别中的分布问题
      - 做分类分析建模问题时，不同类别下的数据样本需要均衡分布
        - 抽样样本能准确代表全部整体特征：
        - 非数值型的特征值域（例如各值频数相对比例、值域范围等）分布需要与总体一致。
        - 数值型特征的数据分布区间和各个统计量（如均值、方差、偏度等）需要与整体数据分布区间一致。
        - 缺失值、异常值、重复值等特殊数据的分布要与整体数据分布一致。
  - 异常检测类数据的处理：
    - 对于异常检测类的应用要包含全部异常样本。对于异常检测类的分析建模，本来异常数据就非常稀少，因此抽样时要优先将异常数据包含进去。
    - 对于需要去除非业务因素的数据异常，如果有类别特征需要与类别特征分布一致；如果没有类别特征，属于非监督式的学习，则需要与整体分布一致。

- 数据抽样案例

  - 简单随机抽样

  ```python
  import random  # 导入标准库
  import numpy as np  # 导入第三方库
  data = np.loadtxt('data3.txt')  # 导入普通数据文件
  shuffle_index = np.random.choice(np.arange(data.shape[0]),2000,True)#随机生成200个行号
  data_sample = data[shuffle_index] #从原始数据中取出200个行号对应的数据
  print(data_sample[:2])  # 打印输出前2条数据
  print(len(data_sample))  # 打印输出抽样样本量
  ```

  - `numpy.random.choice`(a, *size=None***,** *replace=True***,** *p=None***)** 

    - **a** : 1-D array-like or int

      > If an ndarray, a random sample is generated from its elements. If an int, the random sample is generated as if a were np.arange(a)

    - **size** : int or tuple of ints, optional

      > Output shape. If the given shape is, e.g., `(m, n, k)`, then `m * n * k` samples are drawn. Default is None, in which case a single value is returned.

    - **replace** : boolean, optional

      > Whether the sample is with or without replacement

    - **p** : 1-D array-like, optional

      > The probabilities associated with each entry in a. If not given the sample assumes a uniform distribution over all entries in a.

  - 输出

  ```shell
  [[-6.93218476 -8.41946083  6.95390168  3.95294224  5.18762752]
   [-1.28602667  8.33085434  4.13002126 -0.5114419  -5.95979968]]
  2000
  ```

  - 等距抽样

  ```python
  # 等距抽样
  data = np.loadtxt('data3.txt')  # 导入普通数据文件
  sample_count = 2000  # 指定抽样数量
  record_count = data.shape[0]  # 获取最大样本量
  width = record_count / sample_count  # 计算抽样间距
  data_sample = []  # 初始化空白列表，用来存放抽样结果数据
  i = 0  # 自增计数以得到对应索引值
  while len(data_sample) <= sample_count and i * width <= record_count - 1:  # 当样本量小于等于指定抽样数量并且矩阵索引在有效范围内时
      data_sample.append(data[int(i * width)])  # 新增样本
      i += 1  # 自增长
  print(data_sample[:2])  # 打印输出前2条数据
  print(len(data_sample))  # 打印输出样本数量
  ```

  - 输出

  ```python
  [array([-3.08057779,  8.09020329,  2.02732982,  2.92353937, -6.06318211]), array([-2.11984871,  7.74916701,  5.7318711 ,  4.75148273, -5.68598747])]
  2000
  ```


### 三 pandas 练习（链家数据基本分析）

- 数据集说明：网上爬取的链家租房信息

- 载入数据

  ```python
  import pandas as pd
  lj_data = pd.read_csv('1_LJdata.csv')
  ```

- 把列名替换成英文

  ```python
  #原始列名
  lj_data.columns
  #Index(['区域', '地址', '标题', '户型', '面积', '价格', '楼层', '建造时间', '朝向', '更新时间', '看房人数','备注', '链接地址'],dtype='object')
  lj_data.columns = ['district', 'address', 'title', 'house_type', 'area', 'price', 'floor', 'build_time', 'direction', 'update_time', 'view_num', 'extra_info', 'link']
  ```

- 查看数据基本情况

  ```python
  lj_data.head(5)
  lj_data.info()
  lj_data.shape
  lj_data.describe()
  ```

- 最贵和最便宜的房子

  ```python
  lj_data.loc[lj_data['price']==210000]
  lj_data.loc[lj_data['price']==1300]
  lj_data[lj_data['price']==lj_data['price'].min()]
  lj_data[lj_data['price']==lj_data['price'].max()]
  lj_data.sort_values(by='price').head(1)
  lj_data.sort_values(by='price').tail(1)
  ```

- 找到最近新上的10套房子

  ```python
  lj_data.sort_values(by='update_time', ascending=False).head(10)
  #查看所有更新时间
  lj_data['update_time'].unique()
  ```

- 看房人数

  ```python
  lj_data['view_num'].mean() #平均值
  lj_data['view_num'].median() #中位数
  # 不通过看房人数的房源数量
  lj_data['view_num'].value_counts().to_frame().reset_index()
  tmp_df.columns = ['view_num', 'count']
  tmp_df.sort_values(by='view_num', inplace=True)
  tmp_df.head()
  #画图
  %matplotlib inline
  tmp_df['count'].plot(kind='bar')
  ```

- 房龄最小的10套房子的平均看房人数、平均面积..

  ```python
  def get_front_4_num(x):
      try:
          return int(x[:4])
      except:
          return -1
  
  lj_data.loc[:,'house_age'] = 2018-lj_data['build_time'].apply(get_front_4_num)
  
  #面积空值判断
  lj_data = lj_data[lj_data['area'].notnull()]
  #截取面积数值部分
  lj_data.loc[:,'area'] = lj_data['area'].apply(lambda x: x[:-2]).apply(lambda x:float(x))
  #计算平均值
  ```

- 房子价格的分布

  ```python
  import numpy as np
  print(lj_data['price'].mean())   #平均值
  print(lj_data['price'].std())    #方差
  print(lj_data['price'].median())  #中位数
  ```

- 看房人数最多的朝向

  ```python
  popular_direction = lj_data.groupby('direction')[['view_num']].sum()
  popular_direction = popular_direction.reset_index()
  popular_direction[popular_direction['view_num']==popular_direction['view_num'].max()]
  ```

- 房型分布情况

  ```python
  house_type_dis = lj_data.groupby(['house_type']).count()
  %matplotlib inline
  house_type_dis['district'].plot(kind='pie') #饼图
  house_type_dis['district'].plot(kind='bar') #柱状图
  ```

- 最受欢迎的房型

  ```python
  tmp = lj_data.groupby('house_type').agg({'view_num':'sum'})
  tmp = tmp.reset_index()
  tmp[tmp['view_num']==tmp['view_num'].max()]
  ```

- 房子的平均租房价格 （元/平米）

  ```python
  lj_data.loc[:,'price_per_m2'] = lj_data['price']/lj_data['area']
  lj_data['price_per_m2'].mean()
  ```

- 热门小区

  ```python
  address_df = lj_data[['address','view_num']].groupby(['address']).sum()
  address_df = address_df.reset_index()
  address_df.sort_values(by='view_num', ascending=False)
  ```

- 出租房源最多的小区

  ```python
  tmp_df2 = lj_data[['address','view_num']].groupby(['address']).count()
  tmp_df2 = tmp_df2.reset_index()
  tmp_df2.columns =  ['address','count']
  tmp_df2.nlargest(columns='count', n=1)
  ```

- 集中供暖 平均价格

  ```python
  def center_heating(x):
      return 1 if "集中供暖" in x else 0
  
  lj_data.loc[:,'center_heating'] = lj_data['extra_info'].apply(lambda x:center_heating(x))
  lj_data['center_heating'].value_counts()
  lj_data[['center_heating','price']].groupby('center_heating').mean()
  ```

- 不同房型的平均/最大/最小面积

  ```python
  house_type_info = lj_data[['house_type','area']].groupby("house_type")\
  .agg({"area":{'mean':np.mean, 'max':np.max, 'min':np.min}})
  ```

- 哪个地铁口附近房源最多

  ```python
  import re
  #距离14号线(东段)东湖渠站731米 随时看房 精装修 集中供暖
  def find_sub_station(x):
      try:
          return re.search(pattern="(.+号线)(.+站)([0-9]+)米", string=x).group(2)
      except:
          return None
  lj_data.loc[:,'sub_station'] = lj_data['extra_info'].apply(find_sub_station)
  #统计
  lj_data[['sub_station','link']].groupby('sub_station').count()
  ```

- 是否有地铁 价格比较

  ```python
  def has_sub_station(x):
      return 1 if "距离" in x else 0
  
  lj_data.loc[:,'has_sub_station'] = lj_data['extra_info'].apply(has_sub_station)
  
  lj_data[['has_sub_station','price']].groupby('has_sub_station').agg('mean')
  ```

- 地铁附近房源距离地铁平均距离

  ```python
  #距离14号线(东段)东湖渠站731米 随时看房 精装修 集中供暖
  def cal_sub_station_distance(x):
      try:
          return int(re.search(pattern="(.+号线)(.+站)([0-9]+)米", string=x).group(3))
      except:
          return None
  
  lj_data.loc[:,'distance'] = lj_data['extra_info'].apply(cal_sub_station_distance)
  
  lj_data['distance'].mean()
  ```

- 在租房源楼层情况

  ```python
  def get_floor(x):
      if '低楼层' in x:
          return '低楼层'
      elif '中楼层' in x:
          return '中楼层'
      else:
          return '高楼层'
  
  lj_data.loc[:,'floor'] = lj_data['extra_info'].apply(get_floor)
  ```

### 四 综合案例 APP Store 数据分析

#### 4.1 案例介绍

- 案例背景
  - 对APP下载和评分数据分析帮助App开发者获取和留存用户
  - 通过对应用商店的数据分析为开发人员提供可操作的意见
- 通过数据分析要解决的问题
  - 免费和收费的App都集中在哪些类别
  - 收费app的价格是如何分布的，不同类别的价格分布怎样
  - App文件的大小和价格以及用户评分之间是否有关
- 分析流程
  - 数据概况分析
    - 数据行/列数量 
    - 缺失值分布
  - 单变量分析
    - 数字型变量的描述指标（平均值，最小值，最大值，标准差等）
    - 类别性变量（多少个分类，各自占比）
  - 多变量分析
    - 按类别交叉对比
    - 变量之间的相关性分析
  - 可视化分析
    - 分布趋势（直方图）
    - 不同组差异（柱状图）
    - 相关性（散点图/热力图）

- 数据字段说明
  - id : App ID 每个App唯一标识
  - track_name: App的名称
  - size_bytes: 以byte为单位的app大小
  - price：定价（美元）
  - rating_count_tot: App所有版本的用户评分数量
  - rating_count_ver: App当前版本的用户评分数量
  - prime_genre: App的类别
  - user_rating: App所有版本的用户评分
  - sup_devices.num: 支持的iOS设备数量
  - ipadSc_urls.num: app提供的截屏展示数量
  - lang.num 支持的语言数量

#### 4.2 数据清洗

```python
#调用基本包
import pandas as pd
#数据读取
app=pd.read_csv('applestore.csv')
#数据的基本信息
app.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7197 entries, 0 to 7196
Data columns (total 11 columns):
Unnamed: 0          7197 non-null int64
id                  7197 non-null int64
track_name          7197 non-null object
size_bytes          7197 non-null int64
price               7197 non-null float64
rating_count_tot    7197 non-null int64
user_rating         7197 non-null float64
prime_genre         7197 non-null object
sup_devices         7197 non-null int64
ipadSc_urls         7197 non-null int64
lang                7197 non-null int64
dtypes: float64(2), int64(7), object(2)
memory usage: 618.6+ KB
'''
app.head()
'''
          id                                         track_name  size_bytes  \
0  281656475                                    PAC-MAN Premium   100788224   
1  281796108                          Evernote - stay organized   158578688   
2  281940292    WeatherBug - Local Weather, Radar, Maps, Alerts   100524032   
3  282614216  eBay: Best App to Buy, Sell, Save! Online Shop...   128512000   
4  282935706                                              Bible    92774400   

   price  rating_count_tot  user_rating   prime_genre  sup_devices  \
0   3.99             21292          4.0         Games           38   
1   0.00            161065          4.0  Productivity           37   
2   0.00            188583          3.5       Weather           37   
3   0.00            262241          4.0      Shopping           37   
4   0.00            985920          4.5     Reference           37   

   ipadSc_urls  lang     size_mb  paid price_new         rating_new  \
0            5    10   96.119141     1       <10     [5000, 100000)   
1            5    23  151.232422     0        <2  [100000, 5000000)   
2            5     3   95.867188     0        <2  [100000, 5000000)   
3            5     9  122.558594     0        <2  [100000, 5000000)   
4            5    45   88.476562     0        <2  [100000, 5000000)   

  rating_level  
0   [2.5, 4.5)  
1   [2.5, 4.5)  
2   [2.5, 4.5)  
3   [2.5, 4.5)  
4   [4.5, 5.0)  
'''
#发现了unname 0这个奇怪的变量，需要进行清理
app.drop('Unnamed: 0',axis=1,inplace=True)
'''
#drop默认是对行
#inplace表示直接替换掉原有数据
#同样可以用位置来举
#app.drop(app.columns[0],axis=1,inplace=True)
'''

#对数据型变量的整体描述
app.describe()
'''
                 id    size_bytes        price  rating_count_tot  user_rating  \
count  7.190000e+03  7.190000e+03  7190.000000      7.190000e+03  7190.000000   
mean   8.633915e+08  1.990192e+08     1.602039      1.290515e+04     3.526356   
std    2.711453e+08  3.592841e+08     3.096311      7.577526e+04     1.518525   
min    2.816565e+08  5.898240e+05     0.000000      0.000000e+00     0.000000   
25%    6.006381e+08  4.687770e+07     0.000000      2.725000e+01     3.500000   
50%    9.782374e+08  9.708288e+07     0.000000      3.005000e+02     4.000000   
75%    1.082653e+09  1.817411e+08     1.990000      2.796750e+03     4.500000   
max    1.188376e+09  4.025970e+09    49.990000      2.974676e+06     5.000000   

       sup_devices  ipadSc_urls         lang      size_mb         paid  
count  7190.000000  7190.000000  7190.000000  7190.000000  7190.000000  
mean     37.365647     3.706259     5.432823   189.799515     0.435883  
std       3.732868     1.986518     7.919727   342.639972     0.495906  
min       9.000000     0.000000     0.000000     0.562500     0.000000  
25%      37.000000     3.000000     1.000000    44.706055     0.000000  
50%      37.000000     5.000000     1.000000    92.585449     0.000000  
75%      38.000000     5.000000     8.000000   173.321777     1.000000  
max      47.000000     5.000000    75.000000  3839.463867     1.000000  
'''

# 考虑将sizebytes变成mb，新增数据
app['size_mb'] = app['size_bytes'] / (1024 * 1024.0)
app.size_mb.describe()
'''
count    7197.000000
mean      189.909414
std       342.566408
min         0.562500
25%        44.749023
50%        92.652344
75%       173.497070
max      3839.463867
Name: size_mb, dtype: float64
'''
# 根据价格新增标签
app['paid'] = app['price'].apply(lambda x: 1 if x > 0 else 0)
#lambda阐述规则，X为price，为paid赋值，即当price＞0，paid为1，其他情况下，paid为0
app.paid.describe()
'''
count    7197.000000
mean        0.436432
std         0.495977
min         0.000000
25%         0.000000
50%         0.000000
75%         1.000000
max         1.000000
Name: paid, dtype: float64
'''
```

- 小结
  - 清洗异常值（unamed)
  - 处理了给分析造成难度的值(size-bytes)
  - 添加了方便分析的特征（免费/收费)

#### 4.3 单变量分析

```python
#value_counts (price,prime_genre)
#value_Coutn只能对应series，不能对整个dataframe做操作
app.price.value_counts()
'''
0.00      4056
0.99       728
2.99       683
1.99       621
4.99       394
3.99       277
6.99       166
9.99        81
5.99        52
7.99        33
14.99       21
19.99       13
8.99         9
24.99        8
13.99        6
11.99        6
29.99        6
12.99        5
15.99        4
59.99        3
17.99        3
22.99        2
23.99        2
20.99        2
27.99        2
16.99        2
49.99        2
39.99        2
74.99        1
18.99        1
34.99        1
99.99        1
299.99       1
47.99        1
21.99        1
249.99       1
Name: price, dtype: int64
'''
# 价格>50的比较少

#数据的快速分组
bins = [0,2,10,300]
labels = [  '<2', '<10','<300']
app['price_new']=pd.cut(app.price, bins, right=False, labels=labels)
#分组后查看数据分布情况
app.groupby(['price_new'])['price'].describe()
'''
            count       mean        std    min    25%    50%    75%     max
price_new                                                                  
<2         5405.0   0.361981   0.675318   0.00   0.00   0.00   0.00    1.99
<10        1695.0   4.565811   1.864034   2.99   2.99   3.99   4.99    9.99
<300         97.0  28.124021  38.886220  11.99  14.99  19.99  24.99  299.99
'''
# groupby的操作,不同类别app的价格分布
app.groupby(['prime_genre'])['price'].describe()
'''
                    count      mean        std  min  25%   50%    75%     max
prime_genre                                                                  
Book                112.0  1.790536   3.342210  0.0  0.0  0.00   2.99   27.99
Business             57.0  5.116316  10.247031  0.0  0.0  2.99   4.99   59.99
Catalogs             10.0  0.799000   2.526660  0.0  0.0  0.00   0.00    7.99
Education           453.0  4.028234  18.725946  0.0  0.0  2.99   2.99  299.99
Entertainment       535.0  0.889701   1.454022  0.0  0.0  0.00   1.99    9.99
Finance             104.0  0.421154   1.108990  0.0  0.0  0.00   0.00    5.99
Food & Drink         63.0  1.552381   3.972119  0.0  0.0  0.00   1.49   27.99
Games              3862.0  1.432923   2.486609  0.0  0.0  0.00   1.99   29.99
Health & Fitness    180.0  1.916444   2.052378  0.0  0.0  1.99   2.99    9.99
Lifestyle           144.0  0.885417   1.478410  0.0  0.0  0.00   1.24    4.99
Medical              23.0  8.776087  10.788269  0.0  0.0  3.99  16.99   34.99
Music               138.0  4.835435   8.915667  0.0  0.0  0.99   4.99   49.99
Navigation           46.0  4.124783  11.565818  0.0  0.0  0.99   3.74   74.99
News                 75.0  0.517733   1.127771  0.0  0.0  0.00   0.00    3.99
Photo & Video       349.0  1.473295   2.280703  0.0  0.0  0.99   1.99   22.99
Productivity        178.0  4.330562   8.747042  0.0  0.0  1.99   4.99   99.99
Reference            64.0  4.836875   8.285100  0.0  0.0  1.99   4.99   47.99
Shopping            122.0  0.016311   0.180166  0.0  0.0  0.00   0.00    1.99
Social Networking   167.0  0.339880   1.142210  0.0  0.0  0.00   0.00    9.99
Sports              114.0  0.953070   2.419084  0.0  0.0  0.00   0.99   19.99
Travel               81.0  1.120370   2.183772  0.0  0.0  0.00   0.99    9.99
Utilities           248.0  1.647621   2.628541  0.0  0.0  0.99   1.99   24.99
Weather              72.0  1.605417   1.831316  0.0  0.0  0.99   2.99    9.99
'''

#删除价格大于等于49.99的app
app=app[app['price']<=49.99]

#评论情况分析
app.rating_count_tot.describe()

'''
count    7.190000e+03
mean     1.290515e+04
std      7.577526e+04
min      0.000000e+00
25%      2.725000e+01
50%      3.005000e+02
75%      2.796750e+03
max      2.974676e+06
Name: rating_count_tot, dtype: float64
'''
#对用户打分的分组
bins = [0,1000,5000,100000,5000000]
app['rating_new']=pd.cut(app.rating_count_tot, bins, right=False)
#用户打分和价格的关系
app.groupby(['rating_new'])['price'].describe()
'''
                    count      mean       std  min  25%  50%   75%    max
rating_new                                                               
[0, 1000)          4587.0  1.798696  3.324682  0.0  0.0  0.0  2.99  49.99
[1000, 5000)       1193.0  1.740721  3.203853  0.0  0.0  0.0  2.99  39.99
[5000, 100000)     1192.0  0.963549  1.984895  0.0  0.0  0.0  0.99  14.99
[100000, 5000000)   218.0  0.196376  0.925160  0.0  0.0  0.0  0.00   7.99
'''
```

#### 4.4 业务数据可视化

```python
#可视化部分
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#app评分关系
plt.figure(figsize=(30,20))#调整大小
sns.relplot(x="prime_genre", y="user_rating",kind='line',data=app) #折线图
```

![](pics\plot1.png)

```python
app1=app[app['price']<=9.99]
#直方图，APP价格的分布
sns.distplot(app1['price'])
```

![image-20191130025213873](pics\app_plot2.png)

```python
#业务问题2：收费app的价格分布是如何的？不同类别之间有关系吗？

#箱线图，不同类别APP价格
plt.figure(figsize=(10,8))#调整大小
sns.boxplot(x='price',y='prime_genre',data=app[app['paid']==1])

#业务解答：价格绝大部分都集中在9.99美元以内，个别类别（如医疗）等因专业性总体价格会高于其他类别
```

![image-20191130025337133](pics\app_plot3.png)

```python
#箱线图，前五个类别的app价格

app.groupby(['prime_genre']).count().sort_values('price',ascending=False).index

top5 = ['Games', 'Entertainment', 'Education', 'Photo & Video', 'Utilities']
app5 = app[app.prime_genre.isin(top5)]

#箱线图，前五个类别的app价格

plt.figure(figsize=(10,8))#调整大小
sns.boxplot(x='price',y='prime_genre',data=app5[app['paid']==1])
```

![](pics\app_plot4.png)

- 关于箱线图

  ![image-20191130033201482](pics\app_plot10.png)

  - 箱子的中间有一条线，代表了数据的中位数
  - 箱子的上下底，分别是数据的上四分位数（Q3）和下四分位数（Q1）
  - 箱体包含了50%的数据。因此，**箱子的高度在一定程度上反映了数据的波动程度**
  - 上下边缘则代表了该组数据的最大值和最小值
  - 有时候箱子外部会有一些点，可以理解为数据中的“**异常值**” 

```python
#散点图，价格和用户评分的分布
sns.scatterplot(x='price',y='user_rating',data=app)
```

![image-20191130025648215](pics\app_plot5.png)

```python
#只保留五个类别数据
top5= ['Games','Entertainment', 'Education', 'Photo & Video',
       'Utilities']
app5 = app[app.prime_genre.isin(top5)]
#柱状图，前5个类别app的用户评分均值
#同一类别，将免费和付费的评分进行对比
plt.figure(figsize=(10,8))
sns.barplot(x='prime_genre',y='user_rating',hue='paid',data=app5)
```

![image-20191130025751240](pics\app_plot6.png)

#### 4.5 业务解读

- 问题一 免费或收费APP集中在哪些类别
  - 第一步，将数据加总成每个类别有多少个app
  - 第二步，从高到低进行排列
  - 第三步，将数据进行可视化

```python
#使用countplot--count是对数据加总，plot将数据进行可视化
#使用order对数据进行排序

plt.figure(figsize=(20,10))
sns.countplot(y='prime_genre',hue='paid',data=app,order=app['prime_genre'].value_counts().index)
plt.tick_params(labelsize=20)

#业务解答：都是高度集中在游戏类别
```

![image-20191130030804269](pics\app_plot7.png)

-  免费与收费的APP在不同评分区间的分布

```python
bins=[0,0.1,2.5,4.5,5]
app['rating_level']=pd.cut(app.user_rating,bins,right=False)
app.groupby(['rating_level'])['user_rating'].describe()
'''
               count      mean       std  min  25%  50%  75%  max
rating_level                                                     
[0.0, 0.1)     929.0  0.000000  0.000000  0.0  0.0  0.0  0.0  0.0
[0.1, 2.5)     206.0  1.650485  0.400213  1.0  1.5  2.0  2.0  2.0
[2.5, 4.5)    2903.0  3.646056  0.467987  2.5  3.5  4.0  4.0  4.0
[4.5, 5.0)    2660.0  4.500000  0.000000  4.5  4.5  4.5  4.5  4.5
'''
sns.countplot(x='paid',hue='rating_level',data=app)
```

![image-20191130031018593](pics\app_plot8.png)

-   业务问题3：APP的大小和用户评分之间有关系吗？

```python
q4=['user_rating','price','size_mb']
app[q4].corr()
'''
             user_rating     price   size_mb
user_rating     1.000000  0.073237  0.066160
price           0.073237  1.000000  0.314386
size_mb         0.066160  0.314386  1.000000
'''
sns.heatmap(app[q4].corr())
#热力图，展现变量之间两两之间关系的强弱
#业务解答：大小价格都不和评分没有直接关系，但是价格和大小之间有正相关关系
```

![image-20191130031318831](pics\app_plot9.png)



