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


现在将自检要求做修改如下:
图片需为来自数据集的真实图片,/home/kyc/project/GOLD-YOLO/data/coco2017_50/train2017
对任意5张真实图片过拟合训练，输出这五张图片的推理结果可视化
检测出来的物体数量与真实标注物体数量一致
检测出来的物体种类与真实标注物体种类一致
检测出来的物体框位置与真实标注物体框位置差距不大



现在看到模型已经能够正确识别位置信息,但是类别标签好像错了,检查一些类别索引

现在的问题是，由于图片尺寸不一，在统一拉伸到统一尺寸后，绘制出来的标注框的位置与真实物体的位置会有横向或纵向的偏移


编写一个流程自检脚本，参考/home/kyc/project/GOLD-YOLO/eg.py,从/home/kyc/project/GOLD-YOLO/data/coco2017_50/train2017数据集中随机从一张图片进行过拟合测试,并使用训练出来的模型测试同一张图片将推理结果可视化,我们认为,通过自检的要求是,能够准确识别图片中的物体类别和数目，且检测框位置与真实标注位置差距不大,若是自检失败删除模型，修复问题重新自检，直到通过为止。出现问题可以参考convert.py脚本和官方文档https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html


自检成功需要满足以下条件：
对任意图片都能成功
检测出来的物体数量与真实标注物体数量一致
检测出来的物体种类与真实标注物体种类一致
检测出来的物体框位置与真实标注物体框位置差距不大




新芽第二阶段（培育期）重点培育和考察动手编程的能力，请大家从自己第一阶段汇报主题的相关文献中，选择一篇尚未有 Jittor 开源实现的论文用 Jittor 框架（https://github.com/Jittor/jittor）进行实现并开源在个人  Github 上。第二轮代码面试的 Jittor 代码开源链接，请将环境配置、数据准备脚本、训练脚本、测试脚本、与 PyTorch 实现对齐的实验 Log、性能 Log 都放在 README 中。如果计算资源有限，用少量数据的训练效果和 PyTorch 版本的结果对齐。请将训练过程 Log、Loss 曲线，结果、可视化等对齐情况进行记录。

你必须始终记住，要用完整"满血"的small模型进行训练，而不是你遇到问题自作主张进行简化，遇到问题一定要深入修复，确保模型完全还原pytorch的small版本


计划如下：使用coco2017val(约几千张图片)作为训练集，评估两种版本的性能对比，分析一下训练时间大概需要多久



jittor运行环境是conda activate jt
pytorch运行环境是conda activate yolo_py
jittor版本损失数值太小了，是否有异常
整理数据集，提取出训练集和测试集，最后需要统一在测试集上评估性能


conda activate jt && python full_official_small.py --num-images 1000 --batch-size 4 --epochs 50 --name "validated_full_pytorch_small"

这样吧，先conda activate yolo_py,用pytorch small版本进行完整训练，因为pytorch是官方实现不会有错，成功以后再将jittor与pytorch版本对齐


现在我的问题是，在这个实验里，我们的目的是训练出jittor和pytorch版本做对比，且计算资源有限，如果只用不到一千张coco作为训练集学习80类目标会不会效果很差？有没有其他更合适的数据集？比如单一识别人像？或者说换成pascalVOC数据集，只有20类目标？


有几个问题，为什么pytorch版本参数量和jittor版差别这么多
为什么损失后面几乎不下降   轮次 10/50: 验证损失 = 1.000000
   轮次 11/50: 训练损失 = 1.389221
   轮次 16/50: 训练损失 = 1.389208
   轮次 20/50: 验证损失 = 1.000000
   轮次 21/50: 训练损失 = 1.389172

改为nano版本吧，先将jittor版本换为nano版本实现，再比较pytorch版本和jittor版本 nano版本参数量差异