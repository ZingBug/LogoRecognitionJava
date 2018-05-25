import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.HOGDescriptor;
import utils.ImageViewer;

public class Main {
    public static void test(Mat image)
    {
        //定义hog对象
        HOGDescriptor hog=new HOGDescriptor();

        //设置SVM分类器
        hog.setSVMDetector(HOGDescriptor.getDefaultPeopleDetector());//采用已经训练好的行人检测分类器

        //在测试图像上检测行人
        MatOfRect regions=new MatOfRect();
        MatOfDouble o=new MatOfDouble();
        hog.detectMultiScale(image,regions,o);

        //显示
        for(Rect rect:regions.toArray())
        {
            Imgproc.rectangle(image,new Point(rect.x,rect.y),new Point(rect.x+rect.width,rect.y+rect.height),new Scalar(0,0,255));
        }


        ImageViewer viewer=new ImageViewer(image,"Test");
        viewer.imshow();
        viewer.waitKey(0);
    }


    public static void main(String[] args)
    {
        //导入dll文件
        String dllUrl=System.getProperty("user.dir")+"\\opencv\\x64\\opencv_java341.dll";
        System.load(dllUrl);

        String path="F:\\GitHub\\LogoRecognitionJava\\LogoRecognition\\bentian1.jpg";
        Mat image=Imgcodecs.imread(path);
        int flag=1;

        Recognition recognition=new Recognition();
        if(flag==0)
        {
            long startTime=System.currentTimeMillis();//记录开始时间
            recognition.init();
            long endTime=System.currentTimeMillis();//记录结束时间
            float excTime=(float)(endTime-startTime)/1000;
            System.out.println("执行时间： "+excTime);
        }
        else
        {
            recognition.predict(image);
        }
        ImageViewer viewer=new ImageViewer(image,"Test");
        viewer.imshow();
        viewer.waitKey(0);

        //
    }
}
