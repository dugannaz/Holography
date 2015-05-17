import ij.ImagePlus;
import ij.gui.ImageWindow;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.awt.image.BufferedImage;
import java.io.File;

public class Image {

	byte[] pixel;
	int width;
	int height;
	ImageWindow iw;
	ImagePlus ip;
	
	public Image(int width, int height) {
		this.width=width;
		this.height=height;
		pixel = new byte[width*height];
	}
	
	public Image(int size) {
		pixel = new byte[size];
	}
	
	native byte[] readTiff(byte[] file);
	
	public int[] zeroPad(int blockX, int blockY) {
		
		int[] pad = new int[2];
		
		int padX = width%(blockX);
		if (padX >0)
			pad[0] = blockX-padX;

		int padY = height%(blockY);
		if (padY >0)
			pad[1] = blockY-padY;
		
		width += padX;
		height += padY;
		return pad;
	}
	
	public void readImage(File file) {
		
		pixel = readTiff(file.getAbsolutePath().getBytes());
	}
	
	public void show() {
		
		int[] pixels = new int[width*height];
		
		for (int i=0; i < width*height; i++)
            pixels[i] = pixel[i] & 0xFF;
		
		BufferedImage bi = new BufferedImage(width, height, 
				BufferedImage.TYPE_BYTE_GRAY);
        bi.getRaster().setPixels(0, 0, width, height, pixels);
        ip = new ImagePlus("myImage",bi);
        iw = new ImageWindow(ip);
        
	}
	
public void update() {
		
		int[] pixels = new int[width*height];
	
		for (int i=0; i < width*height; i++)
            pixels[i] = pixel[i] & 0xFF;
		
		BufferedImage bi = new BufferedImage(width, height, 
				BufferedImage.TYPE_BYTE_GRAY);
        bi.getRaster().setPixels(0, 0, width, height, pixels);
        ip.setImage(bi);
	}
	
	public void update(Image im) {
		
		int[] pixels = new int[width*height];
		
		for (int i=0; i < width*height; i++)
            pixels[i] = im.pixel[i] & 0xFF;
		
		BufferedImage bi = new BufferedImage(width, height, 
				BufferedImage.TYPE_BYTE_GRAY);
        bi.getRaster().setPixels(0, 0, width, height, pixels);
        ip.setImage(bi);
	}
	
}
