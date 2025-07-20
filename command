该项目终极目标是利用jittor框架复现Gold-YOLO模型，实现论文中描述的实时目标检测任务。
参考/home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch内代码，将其使用jittor框架在/home/kyc/project/GOLD-YOLO/Gold-YOLO_jittor内实现,,翻译代码可以借助convert.py脚本和官方文档https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html

jittor环境是 conda activate jt
pytorch环境是 conda activate yolo_py
先迁移核心模型，再将结构逐步对齐
目录结构模块化即可
精细化迁移
选择small作为优先目标

从网上下载数据集,编写一个流程自检脚本，类似/home/kyc/project/GOLD-YOLO/eg.py,将推理结果可视化,我们认为,通过自检的要求是,能够准确识别图片中的物体类别和数目，且检测框位置与真实标注位置差距不大。

仔细全面严格检查jittor版本Gold-YOLO代码功能，与pytorch版本严格对齐，意思是尽量不要简化，需要深入实现pytorch版本的功能，对齐完需要进行严格的流程自检，修复警告与错误，不断重复这个过程，直到自检成功为止。
自检成功需要满足以下条件：
图片需为来自数据集的真实图片,/home/kyc/project/GOLD-YOLO/data/coco2017_50/train2017
对任意一张真实图片过拟合都能成功
检测出来的物体数量与真实标注物体数量一致
检测出来的物体种类与真实标注物体种类一致
检测出来的物体框位置与真实标注物体框位置差距不大



编写一个流程自检脚本，参考/home/kyc/project/GOLD-YOLO/eg.py,从/home/kyc/project/GOLD-YOLO/data/coco2017_50/train2017数据集中随机从一张图片进行过拟合测试,并使用训练出来的模型测试同一张图片将推理结果可视化,我们认为,通过自检的要求是,能够准确识别图片中的物体类别和数目，且检测框位置与真实标注位置差距不大,若是自检失败删除模型，修复问题重新自检，直到通过为止。出现问题可以参考convert.py脚本和官方文档https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html


自检成功需要满足以下条件：
对任意图片都能成功
检测出来的物体数量与真实标注物体数量一致
检测出来的物体种类与真实标注物体种类一致
检测出来的物体框位置与真实标注物体框位置差距不大

