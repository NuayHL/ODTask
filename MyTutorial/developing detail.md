## Json
主要有
```
json.load()
json.loads()
json.dump()
json.dumps()
```
其中以s结尾的是对str字符串进行转换或者输出的函数，其余是对文件进行操作的。

## print相关 
### 格式化输出
在print中，格式符为真实值预留位置，并控制显示的格式。

格式符可以包含一个类型码，用于控制显示的格式
```
>>> print("test %s" %str)
test str
```
| type | usage             | type | usage              |
|------|:------------------|------|:-------------------|
| %s   | 字符串 (采用str()的显示)  |  %b  | 二进制整数              |
| %r   | 字符串 (采用repr()的显示) |  %d  | 十进制整数              | 
| %c   | 单个字符              |  %i  | 十进制整数              |
| %f   | 浮点数               |  %o  | 八进制整数              |
| %F   | 浮点数，与上相同          |  %x  | 十六进制整数             |
| %e   | 指数 (基底写为e)        |  %g  | 指数(e)或浮点数 (根据显示长度) |
| %E   | 指数 (基底写为E)        |  %G  | 指数(E)或浮点数 (根据显示长度) |
| %%   | 字符"%"             |      |                    |

可以用如下的方式，对格式进行进一步的控制：

``%[flags][width].[precision][typecode]``

flags可以有+,-,' '或0。+表示右对齐。-表示左对齐。' '为一个空格，表示在正数的左侧填充一个空格，从而与负数对齐。0表示使用0填充。

width表示显示宽度

precision表示小数点后精度

typecode表示上面的d、f、s 如%d、%f、%s

比如：
```
print("%+10x" % 10)
print("%04d" % 5)
print("%6.3f" % 2.3)
```

## Cv2

```
cv2.imread()  #format: np.ndarray <h,w,c>
```
```
dst = cv2.resize(src,dsize,dst=None,fx=None,fy=None,interpolation=None)
```
- src：输入图像
- dsize：输出图像的大小。如果该参数为 0，表示缩放之后的大小需要通过公式计算，dsize = Size(round(fx*src.cols),round(fy*src.rows))。其中 fx 与 fy 是图像 Width 方向和 Height 方向的缩放比例。
- fx：Width 方向的缩放比例，如果是 0，按照 dsize * width/src.cols 计算
- fy：Height 方向的缩放比例，如果是 0，按照 dsize * height/src.rows 计算
- interpolation：插值算法类型，或者叫做插值方式，默认为双线性插值
- 方法返回结果 dst 表示输出图像。


## COCOapi

## Numpy
```
np.repeat()
np.tile()

```




