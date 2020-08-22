1.Spark“数字人体”AI挑战赛——脊柱疾病智能诊断大赛   
由于本人天池账号突然无法正常登录，导致错过复赛，特将此项目开源    
本算法采用了cascade faster-rcnn的变体，主要特点如下：        
&nbsp;--删去了bbox的回归，而改用多通道热图回归（解决本次大赛数据并没有提供bbox标注，而只提供了关键点坐标的问题）    
&nbsp;--使用训练样本的坐标标注生成脊柱形态模板，在原本独立的几个热图通道之间建立了空间联系，减小搜索空间，提高搜索效率和精度
&nbsp;--采用相邻的多张轴状图进行关键点检测，经过三维坐标变换映射后，通过取中位数的方式减小搜索误差    
&nbsp;--支持图像360度任意角度旋转，并且不影响识别精度    
&nbsp;--定位和分类共享卷积网络，不需要截图等操作，提高运行效率    
本算法的定位精度尚佳（6mm以内能达到96%左右），但是分类效果一般（testA榜f1在0.54左右）        
这个问题的解决方案有很多，我肯定不是最好的，但是我也希望能尽我自己的一份力，为算法的发展做出点贡献   
2.本项目依赖于nn_tools，可以在github本人（wolaituodiban）项目中中自行搜索下载  
3.文件结构  
--code  
&nbsp;|--core  
&nbsp;|--disease        症状分类相关代码（因为是baseline，椎间盘默认预测v1，锥体默认预测v2)  
&nbsp;|--key_point      定位相关代码  
&nbsp;|--static_files   定义输出格式的静态文件  
&nbsp;|--structure      封装了DICOM的类，包含基础的自动寻找T2序列和中间帧的功能，拓展了一些简单空间几何变换函数  
&nbsp;|--data_utils.py  数据处理的常用函数  
&nbsp;|--dicom_utils.py 读取dicom文件的常用函数  
&nbsp;|--visiliation.py 简单的可视化函数  
&nbsp;|--main.py            入口，请在项目根目录下运行python -m code.main  
--data  
&nbsp;|--lumbar_train150    训练集  
&nbsp;|--train              校验集  
&nbsp;|--lumbar_testA50     A榜测试集  
--models                  存放模型文件的目录  
--predictions             存放预测结果的目录  
--requirements.txt        项目的依赖，请自行安装      
4.关于环境安装的建议  
conda create -n py37torch15 python=3.7  
conda activate py37torch15  
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch  
pip install -r requirements.txt    
