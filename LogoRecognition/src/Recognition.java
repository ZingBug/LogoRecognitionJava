import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.ANN_MLP_ANNEAL;
import org.opencv.ml.TrainData;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.ml.Ml;
import utils.ImageViewer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;


public class Recognition {
    //图片特征数
    //本实验检测窗口定为64×64，就是整张图片的大小，块大小16×16，胞元8×8，步进8×8，这样一张图片就有((64-16)/8+1)*((64-16)/8+1)*(16*16/(8*8)*9=1764维特征
    private int feature_num=1764;
    //样本数量
    private final int sample_num=1000;
    //样本种类
    private final int class_num=5;
    //图片列行
    private int image_cols=64;
    private int image_rows=64;
    //训练数据，每一行一个样本
    private Mat trainMat=new Mat(sample_num,feature_num,CvType.CV_32FC1);
    //训练样本标签
    private Mat lableMat=new Mat(sample_num,class_num,CvType.CV_32FC1);
    //特征集
    private MatOfFloat descriptors;
    private MatOfPoint locations;
    //神经网络
    private ANN_MLP bp=ANN_MLP.create();
    //神经网络训练结果信息保存路径
    private String xmlPath="car.xml";
    //样本标签
    private String[] logoLables={"雪铁龙","大众","一汽","福田","本田"};

    private void getHOG(Mat img)
    {
        descriptors=new MatOfFloat();
        locations=new MatOfPoint();
        HOGDescriptor hog=new HOGDescriptor(new Size(64,64),new Size(16,16),new Size(8,8),new Size(8,8),9);
        hog.compute(img,descriptors,new Size(64,64),new Size(0,0),locations);
    }

    private void packTrainData(Mat img,int lable,int counter)
    {
        //填装数据
        getHOG(img);
        int cur=0;
        for(float d:descriptors.toArray())
        {
            trainMat.put(counter,cur++,d);
        }
        lableMat.put(counter,lable,1.0);
    }

    private void loadTrainData()
    {
        //先初始化label,把所有点置0
        for(int i=0;i<sample_num;i++)
        {
            for(int j=0;j<class_num;j++)
            {
                lableMat.put(i,j,0.0);
            }
        }
        //导入训练数据
        int label;
        int counter=0;
        int num;
        for(int i=0;i<class_num;i++)
        {
            label=i;
            num=0;
            String path=System.getProperty("user.dir")+"\\Cardata\\train\\"+i+"\\path.txt";
            try
            {
                File file=new File(path);
                InputStreamReader reader=new InputStreamReader(new FileInputStream(file));
                BufferedReader br=new BufferedReader(reader);
                String line=br.readLine();
                while (line!=null)
                {
                    num++;
                    Mat image=Imgcodecs.imread(line);
                    Imgproc.resize(image,image,new Size(64,64));
                    Imgproc.cvtColor(image,image,Imgproc.COLOR_RGB2GRAY);
                    packTrainData(image,label,counter++);
                    line=br.readLine();
                }
                System.out.println(num);
                br.close();
                reader.close();
            }
            catch (Exception e)
            {
                e.printStackTrace();
            }
        }
        System.out.println(counter);
    }
    private void trainData()
    {
        //训练神经网络
        //设置各层节点数
        int[] layer={feature_num,48,class_num};
        Mat layerMat=new Mat(1,layer.length,CvType.CV_32FC1);
        for(int i=0;i<layer.length;i++)
        {
            layerMat.put(0,i,layer[i]);
        }
        bp.setLayerSizes(layerMat);
        //设置激活函数
        bp.setActivationFunction(ANN_MLP.SIGMOID_SYM);
        //设置终止条件，最大迭代次数和最小误差
        TermCriteria termCriteria=new TermCriteria(TermCriteria.EPS,10000,0.001);
        bp.setTermCriteria(termCriteria);
        //设置训练方法为反向传播算法BP
        bp.setTrainMethod(ANN_MLP.BACKPROP);
        //设置权值更新冲量
        //bp.setBackpropMomentumScale(0.01);
        //bp.setBackpropWeightScale(0.01);
        //开始训练
        boolean trained=bp.train(trainMat,Ml.ROW_SAMPLE,lableMat);
        if(trained)
        {
            bp.save(xmlPath);
            System.out.println("训练结束");
        }
        else
        {
            System.out.println("训练失败");
        }
        bp.clear();

        System.out.println("layer: "+layerMat.dump());
    }
    public void init()
    {
        loadTrainData();
        //开始训练
        trainData();
    }

    public void predict(Mat image)
    {
        ANN_MLP ann=ANN_MLP.load(xmlPath);
        Imgproc.resize(image,image,new Size(64,64));
        Imgproc.cvtColor(image,image,Imgproc.COLOR_RGB2GRAY);
        Mat sample=new Mat(1,feature_num,CvType.CV_32FC1);
        getHOG(image);
        int cur=0;
        for(float d:descriptors.toArray())
        {
            sample.put(0,cur++,d);
        }
        System.out.println(cur);
        Mat predict=new Mat(1,class_num,CvType.CV_32FC1);
        ann.predict(sample,predict,0);

        System.out.println("sample--"+sample.dump());
        System.out.println("outputs: "+predict.dump());

        Core.MinMaxLocResult maxLoc=Core.minMaxLoc(predict);
        System.out.println(maxLoc.maxLoc+"---"+maxLoc.minVal);
        int index=(int)maxLoc.maxLoc.x;
        System.out.println("character:"+logoLables[index]);
    }
}
