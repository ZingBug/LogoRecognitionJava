package utils;

import org.opencv.core.Mat;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

/**
 * An Gui extention for opencv3.4 using Swing.
 */
public class ImageViewer extends JPanel implements KeyListener {
    public ImageViewer(Mat m, String window) {
        super();
        init(m, window);
    }
    public ImageViewer(Mat m)
    {
        this(m,"ImageViewer");
    }
    //Elements for paint
    private Mat mat;
    private boolean firstPaint = true;
    private int type;
    private JFrame jframe = new JFrame();
    private Image image;

    private void init(Mat m, String window){
        this.mat = m;
        if(mat.channels() == 1)
            type = BufferedImage.TYPE_BYTE_GRAY;
        else
            type = BufferedImage.TYPE_3BYTE_BGR;
        mat2BufIm(this.mat);
        jframe.add(this);
        jframe.setSize(mat.cols(), mat.rows());
        jframe.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        jframe.setTitle(window);
        jframe.addKeyListener(this);
        jframe.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                super.windowClosing(e);
                System.exit(0);
            }
        });
    }

    public void updateImage(Mat m)
    {
        this.mat=m;
        if(mat.channels() == 1)
            type = BufferedImage.TYPE_BYTE_GRAY;
        else
            type = BufferedImage.TYPE_3BYTE_BGR;
        mat2BufIm(this.mat);
        jframe.add(this);
        jframe.setSize(mat.cols(), mat.rows());
    }

    @Override
    public void paintComponent(Graphics g)
    {
        g.drawImage(this.image,0,0,null);
    }

    private void mat2BufIm(Mat matrix) {
        int bufferSize = matrix.channels() * matrix.cols() * matrix.rows();
        byte[] buffer = new byte[bufferSize];
        matrix.get(0, 0, buffer); // 获取所有的像素点
        BufferedImage bufferedImage = new BufferedImage(matrix.cols(), matrix.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) bufferedImage.getRaster().getDataBuffer()).getData();
        System.arraycopy(buffer, 0, targetPixels, 0, buffer.length);
        this.image=bufferedImage;
    }

    public void imshow(){
        if(firstPaint){
            jframe.setVisible(true);
            firstPaint = false;
        }
        this.repaint();
    }

    //Elements for waitKey.
    private  Object mt = new Object();
    private  int lastKey = 0;
    private  int key = 0;
    public  int waitKey(int millisecond){
        //TODO 实现监听键盘
        try {
            if(millisecond == 0){
                synchronized(mt){
                    mt.wait();
                }
            }
            Thread.sleep(millisecond);
        } catch (InterruptedException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        int ret = -1;
        if(key != lastKey){
            ret = key;
            lastKey = key;
        }
        return ret;
    }
    @Override
    public void keyPressed(KeyEvent e) {
        // TODO Auto-generated method stub
        synchronized(mt){
            mt.notifyAll();
        }
        this.key = e.getKeyCode();

    }
    @Override
    public void keyReleased(KeyEvent arg0) {
        // TODO Auto-generated method stub

    }
    @Override
    public void keyTyped(KeyEvent arg0) {
        // TODO Auto-generated method stub

    }
}

