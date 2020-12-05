- 数据查看
	- data.head() 默认展示五条记录
	- data.describe() 数据的描述信息
	- data.dtypes 数据类型
	- data.info() 数据的基本情况


- 数据审查
	- 转换数字类型变量
		- data.get_dummies(data) 
	- 丢弃删除
		- data.drop()
	- 重用名
		- data.rename(columns={'new_name':'old_name'})
	- 字段名字的更改
		- data.columns.str.lower() 
	- 排序问题
		- data.sort_value(by='字段',ascending = Flase) 
	- 取数据的交并集
		- pd.merge(data1,data2,on['字段'])
		- pd.merge(data1,data2,on['字段']，how='outer')

	- 重复问题
		- data.loc[:,'字段'].value_counts()  
		- data[data.duplicated()]   查找整个表的重复字段
		- data.duplicated(subset='判断的字段'，keep='first or last or False' 判断哪一个开始不算重复)
		-drop_duplicated()  删除重复   函数跟上面一样 
	- 遍历数据表  进行一个遍历循环 进行判断 
		使用的过程中是用的iloc进行和python原生使用
		- for index,row in data.interrows()
			if row.iloc[0]  == '':
				print(row.iloc[0])  
	- 过滤抽取
		- data[data[] == 条件]

- 时间转化
	- to_datetime
		- data['just_time']=data1['地震发生时间'].dt.time  显示只有时间  几几点几分
		- data['just_data']=data['地震发生时间'].dt.data   时间年月日时分秒
		- data['just_hour']=data['地震发生时间'].dt.hour   回是一个整数  也就是多少小时
		- data['just_hour']=data['地震发生时间'].dt.days   

	- 改变时间的格式
		- data['首次上线日期'] = pd.to_datetime(data['首次上线日期'], format='%Y%m%d')



- 缺失值/异常值处理
	- dataframe.isnull() 是否缺失值
	- dataframe.dropna() 删除缺失值
	- dataframe.fillna() 填充缺失值
	- dataframe.mean()  计算平均值
	- dataframe.std()   计算标准差
	- dataframe.duplicated()  判断重复数据记录
	- dataframe.drop_duplicates()  删除数据记录所有列值相同的记录
	- for ind,ech in enumerate(datas[:-1]):


- 数据导入
	- data = pd.read_excel(r'C:\Users\cy176\1Datars\sales.xlsx',sheet_name = 2,header = None,index_col = 0)
	读取目录下的文档是假  r
	index_col 表示哪一行为索引
	sheet_name 表示在Excel中会用的第几个小数据
	header 代表多级索引的行
	- for how,datas in zip(names,data):
	一直循环 每一次都是datas = data  然后换  下面可以操作
		- print（data.describe()） 
		- 这是属于数据中有多个表  查看每个表
	- for how，data in enumerate（data[:-1]）
		- 跟上面配套一起的   how是第几个表的数值  data是表数据



- 数据索引
	column = pd.MultiIndex.from_product([['上半年','下半年'],['90#','93#','97#']])

	使用dataFrame的columns属性，来对表格索引重新赋值
	sheet2.columns = column
	sheet2

	# 多级索引的赋值使用如下方法
	sheet2.loc['成都',('上半年','90#')] = 8900

	# 多级索引的访问可以使用如下方法
	# 这种属于跨级访问，相当于对sheet2['上半年']产生的临时对象赋值
	sheet2['上半年'].loc['成都','90#'] = 18000
	# 多级索引赋值还可以采用如下办法
	sheet2_copy = sheet2['上半年'].copy()
	sheet2_copy.loc['成都','90#'] = 9700
	sheet2['上半年'] = sheet2_copy
	sheet2
	
	# 普通一级索引的访问方式回顾
	sheet2_copy

 - sns的折线图seaborn.lineplot
 	- sns画线函数
	 	seaborn.lineplot(x=None, y=None, hue=None, 
                 size=None, style=None, data=None,
                 palette=None, hue_order=None, hue_norm=None, 
                 sizes=None, size_order=None, size_norm=None, 
                 dashes=True, markers=None, style_order=None, 
                 units=None, estimator='mean', ci=95, n_boot=1000,
                 sort=True, err_style='band', err_kws=None,
                 legend='brief', ax=None, **kwargs)
    data:是DataFrame类型的;

	可选：下面均为可选
	x,y：数据中变量的名称;
	hue:数据中变量名称(比如：二维数据中的列名)
	作用：对将要生成不同颜色的线进行分组，可以是分类或数据。
	size:数据中变量名称(比如：二维数据中的列名)
	作用：对将要生成不同宽度的线进行分组，可以是分类或数据。
	style:数据中变量名称(比如：二维数据中的列名)
	作用：对将生成具有不同破折号、或其他标记的变量进行分组。
	palette:调试板名称，列表或字典类型
	作用：设置hue指定的变量的不同级别颜色。
	hue_order:列表(list)类型
	作用：指定hue变量出现的指定顺序，否则他们是根据数据确定的。
	hue_norm:tuple或Normalize对象
	sizes:list dict或tuple类型
	作用：设置线宽度，当其为数字时，它也可以是一个元组，指定要使用的最大和最小值，会自动在该范围内对其他值进行规范化。
	units:对变量识别抽样单位进行分组，使用时，将为每个单元绘制一个单独的行。
	estimator:pandas方法的名称或回调函数或者None
	作用：用于在同一x水平上聚合y变量的多个观察值的方法，如果为None，则将绘制所有观察结果。

- 查看dataframe的列名字
  - [column for column in data]