# 一、flask
    app = Flask(__name__)   
可通过这个函数的template_floder参数设置render_template函数的默认目录   
``` 
@app.route('/show/info')    
def show_info():    
    return render_template("Web.html")  
```
创建网址'/show/info'和函数'show_info'的关系，点开网址自动运行该函数   
该函数默认去该项目的templates文件夹中查找项目
# 二、浏览器能识别的标签标签
## 2.1 编码(head)
    <meta charset="UTF-8">
网站编码
## 2.2 标题(head)
    <title>我的网站<\title>
网页的名字
## 2.3 h标签(body)
``` 
<h1>一号标题</h1>
<h2>二号标题</h2>
<h3>三号标题</h3>
<h4>四号标题</h4>
<h5>五号标题</h5>
<h6>六号标题</h6>
```
h标签，行内标签，自己有多大就占多少位置    
一号字体最大，六号字体最小
## 2.4 div和span(body)
div，块级标签，一个标签占一整行    
span，行内标签，自己有多大就占多少位置   
一般是div标签中嵌套span标签
## 2.5 超链接
    <a href="https://www.bilibili.com/">文本，网页跳转提示,比如说，跳转哔哩哔哩</a>
跳转别人的网站 
    
    <a href=/show/info"></a>    
跳转到自己的网站
## 2.6 图片
### 2.6.1 图片显示
    <img src="/static/a.jpg">
图片地址可以是项目的文件地址，项目的默认目录static，用于存放静态资源

    <img src="https://pic.netbian.com/tupian/35280.html" alt="">
也可以是引用别人网站的图片地址，但是引用有风险,会触发网站的防盗链，显示不了图片
### 2.6.2 图片大小调整
    <img src="/static/rurumu_闭眼.jpg" style="width: 100px">    
使用style参数自定义图片大小，图片长度和宽度的设置使用像素点px，只定义长度或者宽度的话，图片会等比例缩放     
img标签也是行内标签 

    <img src="/static/rurumu_闭眼.jpg" style="width: 10%">
使用百分比给参数赋值
## 2.7 小结
### 学习的标签
```
<h1></h1>
<div></div>
<span></span>
<a></a>
<img></img>
```
### 默认目录
默认目录templates和static    
templates目录存放html文件，static目录存放静态资源
templates和staitc必须是平级目录
### 图片链接
```angular2html
<div>
    <a href="https://www.bilibili.com/">
        <img src="/static/rurumu_闭眼.jpg" alt="">
    </a>
</div>
```
a标签嵌套img标签
## 2.8 列表标签
### 无序列表
```angular2html
<ul>
    <li>我爱中国</li>
    <li>我爱中国</li>
    <li>我爱中国</li>
</ul>
```
效果为在文本前加上一个点
### 有序列表
```angular2html
<ol>
    <li>我爱中国</li>
    <li>我爱中国</li>
    <li>我爱中国</li>
</ol>
```
效果为在完本前加上序号
## 2.9 表格标签
```angular2html
<table border="1">
    <thead>
        <tr>    <th>学号</th> <th>姓名</th> <th>年龄</th>     </tr>
    </thead>
    <tbody>
        <tr>    <th>1</th>  <th>杨鱼</th> <th>18</th>     </tr>
        <tr>    <th>1</th>  <th>杨鱼</th> <th>18</th>     </tr>
        <tr>    <th>1</th>  <th>杨鱼</th> <th>18</th>     </tr>
    </tbody>
</table>
```
table标签的border参数可以给表格加上边框
## 2.10 input系列
```angular2html
<input type="text">#文本框
<input type="password">#密码框
<input type="file">#文件框
<input type="radio" name="单选框框名">选项文本 #单选框，两个单选框框名要一致才能互斥
<input type="checkbox">选项文本 #复选框
<input type="button" value="提交"> #按钮，绑定输入框的名字
<input type="submit" value="提交"> #提交按钮
```
## 2.11 下拉框
### 单选下拉框
```angular2html
<select name="" id="">
    <option value=""></option>
    <option value=""></option>
    <option value=""></option>
</select>
```
### 多选下拉框
```angular2html
<select name="" id="" multiple>
    <option value=""></option>
    <option value=""></option>
    <option value=""></option>
</select>
```
## 2.12 多行文本
```angular2html
<textarea name="" id="" cols="30" rows="10"></textarea> # 该参数为10行，30列
```
## 案例 用户注册
```angular2html
<form action="提交的地址" method="get">
    用户名：<input type="text" name="name">
    密码：<input type="password" name="password">
    <input type="submit" value="submit按钮">
</form>
```
页面上的数据提交到后台
### form标签要包裹要提交数据的标签
提交方式：method='get'
提交的地址：action='/xxx/xx/x'
在form标签里面必须有一个submit标签
### 在form里的一些标签：input/select/textarea
```angular2html
<input type="text" name="name">
```
一定要有name属性