- DATABASE
	- create   drop    show

	mysql -u用户名  -p密码
	quit  登出数据库   exit退出数据库
	use 数据库名;  使用数据库
	
	- (增)create database 数据库名 charset=utf8;

	- (删)drop database 数据库名;  删除数据库
	
	- (查)
		show databases ;  查看所有数据库
		show create database 数据库名;   

- TABLE
	- create      drop      desc

	- (增)create table 表名(字段名称   类型    条件   条件,);

	- (删)drop table  名字
	
	- (查)show create table 表名;  查看创表SQL语句
		  show tables

- ZIDAN  
	- alter （add   drop   change   modify）   desc

	- 增     
		- alter table 表名  add 列名  类型  约束条件;在表中添加的字段
		- alter table  表名 add primary key（列）
		- alter table students add isdelete bit default 0; （添加字段） update students set isdelete = 1 where id = 8; （本质就是修改操作）一般使用逻辑删除   虽然他删除了  但是本身还是存在的  就是意义上是删除了
	- 删
		- alter table  表名  drop 列名 ;删除表中字段
		- alter table 表名 drop primary key(列) 删除主键
	- 改
		- alter table 表名  modify  列名  类型  约束 ; 修改表中的字段
		- alter table  表名  change  原名  新名  类型  约束条件;   修改表中已有字段名字
	- 查
		desc table

- DATA 
	- insert into    delete    select    update

	- (增）
		- insert into 表名 values (...)  
		- insert into 表名 (列1,...) values(值1,...)   
		- insert into 表名 values(...),(...);   全列多行插入
		- insert into 表名(列1,...) values(值1,...),(值1,...);  

	- （删）
		- delete from 表名 where 条件

	- （改）
		- update 表名 set 列1=值1,列2=值2... where 条件  修改数据

	- （查）
		- select * from 表名;    查询所有列
		- select id as 序号, name as 名字, gender as 性别 from students;
		- select  id ,name ,gender from stu;  
		- select stu.id,stu.name,stu.gender from stu;  不仅仅是单表时候带着表名
		- select 列1,列2,... from 表名;   
		- select  distinct  列1.... from 表名  取出重复数据
		- select 字段  form  表名  where 条件;   where条件查询语法格式  where后面跟着运算符  如 id>1    id=1  多个（id>2 and  name = 10）
		- select * from 表名 limit start,count   start表示开始行索引，默认是0  count表示查询条数


