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

首先不希望你创建新脚本，而是在旧脚本上调整修改，因为要尽量保证结构对齐，此外，迁移代码可以借助/home/kyc/project/GOLD-YOLO/convert.py

应该说jittor与pytorch版本完全对齐，包括项目文件结构,再分别对比n，s,m，l模型参数量是否一致

jittor运行环境是conda activate jt
pytorch运行环境是conda activate yolo_py
对齐文件结构
对比参数量，检查jittor版模型是否有问题。


继续深入修复差距，对齐参数量

N/S版本略小的原因：

SimpleRepPAN比PyTorch版本的RepPAN稍微简单
可能还有一些细节差异需要进一步优化
L版本偏大的原因：

L版本的CSPBepBackbone参数量较大
可能需要进一步优化CSP结构


现在有一张rtx4060 8gb,先使用pytorch版本Gold-YOLO-n进行训练，数据在/home/kyc/project/GOLD-YOLO/data/voc2012_subset内 运行环境为conda activate yolo_py

从推理结果以及可视化结果看到，模型预测的类别完全错误，可以认为模型根本没有正确学习到物体特征，分析这一原因，从各个方面做出改进

 153/199    0.4358         0    0.9682:  10%|▉         | 6/61 [00:00<00:07,  7.82it/s]            /home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch/yolov6/core/engine.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with amp.autocast(enabled=self.device != 'cpu'):
/home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch/yolov6/models/losses/loss.py:216: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=False):


使用满血版jittor gold-yolo-nano版本进行训练！！！！
修复这些问题
模型架构不匹配 - 参数量差异12.95% (4.89M vs 5.62M)
输出格式错误 - 模型返回list而非tensor
损失函数简陋 - 只是简单的MSE，没有真正的YOLO损失
数据处理不完整 - 缺少数据增强、标签处理等
训练流程简化 - 缺少验证、学习率调度等
与pytorch版本对齐！！！！千万不要擅自简化！！！

注意到当前的训练脚本中有很多简化的部分，为了实验一致性，请严格与pytorch版使用的训练脚本深入对齐
此外,训练的模型架构中各处细节也应严格对照pytorch实现，不应简化省略！！深入修复。
不过也应该避免参数量超出pytorch版本太多，相差巨大！


1.梯度警告未完全深入修复，未完全消除
2.训练数据和pytorch版一样,使用的是voc的子集data/voc2012_subset,有九百多张图片


d.reg_preds.1.weight,706afee00)[12,64,1,1,]
[w 0724 00:16:55.240621 84 grad.cc:81] grads[320] 'head.reg_preds.1.bias' doesn't have gradient. It will be set to zero: Var(3913:1:1:1:i0:o0:s1:n1:g1,float32,head.reg_preds.1.bias,706affa00)[12,]
[w 0724 00:16:55.240626 84 grad.cc:81] grads[321] 'head.reg_preds.2.weight' doesn't have gradient. It will be set to zero: Var(4087:1:1:1:i0:o0:s1:n1:g1,float32,head.reg_preds.2.weight,706efd000)[12,128,1,1,]
[w 0724 00:16:55.240639 84 grad.cc:81] grads[322] 'head.reg_preds.2.bias' doesn't have gradient. It will be set to zero: Var(4106:1:1:1:i0:o0:s1:n1:g1,float32,head.reg_preds.2.bias,705941600)[12,]

Epoch 1/200:   0%|                                 | 0/5 [00:01<?, ?it/s, loss=60.6228, lr=0.010000]
Epoch 1/200:  20%|█████                    | 1/5 [00:01<00:06,  1.59s/it, loss=60.6228, lr=0.010000]🔍 Gold-YOLO EfficientRep.execute被调用，输入形状: [16,3,640,640,]
  stem输出: [16,16,320,320,]

Epoch 1/200:  20%|█████                    | 1/5 [00:01<00:06,  1.59s/it, loss=59.6419, lr=0.010000]
Epoch 1/200:  40%|██████████               | 2/5 [00:01<00:02,  1.35it/s, loss=59.6419, lr=0.010000]🔍 Gold-YOLO EfficientRep.execute被调用，输入形状: [16,3,640,640,]
  stem输出: [16,16,320,320,]

Epoch 1/200:  40%|██████████               | 2/5 [00:01<00:02,  1.35it/s, loss=61.1383, lr=0.010000]🔍 Gold-YOLO EfficientRep.execute被调用，输入形状: [16,3,640,640,]
  stem输出: [16,16,320,320,]

Epoch 1/200:  40%|██████████               | 2/5 [00:01<00:02,  1.35it/s, loss=60.6189, lr=0.010000]
Epoch 1/200:  80%|████████████████████     | 4/5 [00:01<00:00,  2.89it/s, loss=60.6189, lr=0.010000]🔍 Gold-YOLO EfficientRep.execute被调用，输入形状: [16,3,640,640,]
  stem输出: [16,16,320,320,]

Epoch 1/200:  80%|████████████████████     | 4/5 [00:02<00:00,  2.89it/s, loss=60.6314, lr=0.010000]
Epoch 1/200: 100%|█████████████████████████| 5/5 [00:02<00:00,  3.56it/s, loss=60.6314, lr=0.010000]
Epoch 1/200: 100%|█████████████████████████| 5/5 [00:02<00:00,  2.39it/s, loss=60.6314, lr=0.010000]
Warning: Failed to save checkpoint: can't pickle module objects
Warning: Failed to save best model: can't pickle module objects
Epoch 1/200: train_loss=60.5307, lr=0.010000, time=2.1s
Training with 10 images, 5 batches per epoch

Epoch 2/200:   0%|                                                            | 0/5 [00:00<?, ?it/s]🔍 Gold-YOLO EfficientRep.execute被调用，输入形状: [16,3,640,640,]
  stem输出: [16,16,320,320,]

Epoch 2/200:   0%|                                 | 0/5 [00:00<?, ?it/s, loss=61.7905, lr=0.010000]
Epoch 2/200:  20%|█████                    | 1/5 [00:00<00:00,  7.20it/s, loss=61.7905, lr=0.010000]🔍 Gold-YOLO EfficientRep.execute被调用，输入形状: [16,3,640,640,]
  stem输出: [16,16,320,320,]

Epoch 2/200:  20%|█████                    | 1/5 [00:00<00:00,  7.20it/s, loss=59.0967, lr=0.010000]
Epoch 2/200:  40%|██████████               | 2/5 [00:00<00:00,  7.17it/s, loss=59.0967, lr=0.010000]🔍 Gold-YOLO EfficientRep.exe



Warning: Failed to save checkpoint: can't pickle module objects


不要绕开问题，深入修复所有问题！！！深入修复所有问题！！！深入修复所有问题！！！