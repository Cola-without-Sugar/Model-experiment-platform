## CenterNet

> 

### 复现

首先运行demo文件[centernet运行](https://blog.csdn.net/weixin_42533799/article/details/123087868)

### 创新

### 附录

#### 报错记录

anji@monitor

**1. 编译报错**

首先使用`cd REALLY_WANT_TO_Learning\0-project-test\centernet-master\src\lib\models\networks\DCNv2`

在使用python+c++混合编译的情况下运行

`python setup.py build develop`

```shell
1 D:\Anoconda\lib\site-packages\torch\utils\cpp_extension.py:381: 	UserWarning: Attempted to use ninja as the BuildExtension backend but we could not find ninja.. Falling back to using the slow distutils backend.warnings.warn(msg.format('we could not find ninja.'))
2 D:\Anoconda\lib\sitepackages\torch\utils\cpp_extension.py:316: UserWarning: Error checking compiler version for cl: [WinError 2] 系统找不到指定的文件。
3 warnings.warn(f'Error checking compiler version for {compiler}: {error}')
4 RuntimeError: Error compiling objects for extension
5 LINK : fatal error LNK1181: 无法打开输入文件
```

* 报错的第一条是指无法找到ninja包，所以使用`pip install ninja` 来进行安装

* 报错的第二条是指无法找到版本匹配的MSVC，MSVC是一个编译器，其编译器的名字叫"cl.exe"是微软专为VS开发的一款编译器，检查vs版本或者在2019下可以使用`call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"`

* 第三个报错同第二个

* 第四个报错的原因据查询得到可能是由于错误的pytorch版本与cuda的版本匹配问题，检查一下因为ninja版本的问题，需要的ninjia版本为1.3，所以实际上版本发生了冲突，解决方法是通过将`torch/utils/cpp_extension.py`中的`['ninja','-v']`改成`['ninja','--version']`

  > pytorch默认使用ninjia作为[backend](https://so.csdn.net/so/search?q=backend&spm=1001.2101.3001.7020)
  
  ~~ninja版本因为更新速度较慢与torch相比无法兼容较高版本的pytorch，所以方式一般是降级pytorch，因为pytorch的版本与cuda的版本相对应，那么就只能通过安装多个cuda的版本来进行协调~~
  
  ~~pytorch版本需要降到1.5以下，但是降低pytorch版本需要降低python版本，而降低python版本会导致一些功能无法实现，且目前并不知道会不会使用到cuda等gpu完成训练~~
  
  ~~在查看源码的过程中，发现pytorch需要的ninjia为1.3版本，查看后，目前的ninjia为1.11.1，也许更改ninjia的版本也许可以~~

**3.测试demo**

`AttributeError: module 'torch.jit' has no attribute 'is_tracing'`

这个错误是由于torch与torchvision的版本不匹配导致的，但是由于版本1.5与torchvision版本0.6虽然匹配，但是依然没有，怀疑是因为torchvision在1.5版本之后不再使用相关的程序代码

`ModuleNotFoundError: No module named '_ext' `

是由于编译的版本不通过，需要重新下载对应1.x版本的[DCNv2](https://github.com/lbin/DCNv2/tree/pytorch_1.7)进行重新编译

运行opencv的时候显示错误

```shell
cv2.error: OpenCV(4.6.0) D:\a\opencv-python\opencv-python\opencv\modules\highgui\src\window.cpp:1267: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function ‘cvShowImage'
```

解决方法，更新opencv-contrib-python的版本
`pip install opencv-contrib-python -U`

* 紧接着出现了`No matching distribution found for skbuild`
	解决方法，安装`pip install scikit-build`

* `Problem with the CMake installation, aborting build. CMake executable is cmake`
	解决方法：更新cmake的版本，在unbuntu下运行` sudo apt-get install cmake`

在安装完成后，仍然报错，猜测由于未更新两个编译版本就进行安装，导致的问题，所以使用`sudo apt-get libgtk2.0-dev/pkg-config`，然后重新进行编译运行

> 在安装的过程中，如果不想使用pip自动缓存的chcae进行处理的话，可以使用命令
>
> `pip --no-cache-dir install 包名`

```cv2.error: OpenCV(4.7.0) /tmp/pip-build-00_q1gmj/opencv-contrib-python/opencv/modules/highgui/src/window_gtk.cpp:635: error: (-2:Unspecified error) Can't initialize GTK backend in function 'cvInitSystem'```

由于使用了虚拟机进行运行程序，只有一个终端命令行，所以无法运行图像显示的窗口文件。所以无法调用imshow等需要对窗口进行操作的功能。

**4. 训练模型**

`遇到ValueError: signal number 32 out of range`

这是由于num_work线程无法开启的原因

**解决方法**：

* 将num_workers改成0，可以运行，但是无法开启多线程可能会对之后的程序运行产生影响
* python版本出现的问题，在3.6.6之后的版本修复了这个问题
