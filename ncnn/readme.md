




# onnx导出ncnn模型

./3rdparty/ncnn/bin/onnx2ncnn ./model/CIFAR_10_20.onnx ./model/CIFAR_10_20.param ./model/CIFAR_10_20.bin    

# 优化ncnn模型
./3rdparty/ncnn/bin/ncnnoptimize ./model/CIFAR_10_20.param ./model/CIFAR_10_20.bin ./model/CIFAR_10_20-opt.param ./model/CIFAR_10_20-opt.bin 0  

# 产生补偿表 calibration table file 
find dataset/imagenet-sample-images/ -type f > dataset/imagenet-sample-images/imagelist.txt     
cat dataset/imagenet-sample-images/imagelist.txt    
./3rdparty/ncnn/bin/ncnn2table ./model/CIFAR_10_20-opt.param ./model/CIFAR_10_20-opt.bin ./dataset/imagenet-sample-images/imagelist.txt ./model/CIFAR_10_20.table mean=[0.5,0.5,0.5] norm=[0.5,0.5,0.5] shape=[32,32,3] pixel=BGR thread=8 method=kl    

# int8量化
./3rdparty/ncnn/bin/ncnn2int8 ./model/CIFAR_10_20-opt.param ./model/CIFAR_10_20-opt.bin ./model/CIFAR_10_20-int8.param ./model/CIFAR_10_20-int8.bin ./model/CIFAR_10_20.table   




