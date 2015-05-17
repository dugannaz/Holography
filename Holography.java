import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.io.File;

import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JFrame;

/*
 * Holographic image processing
 */

public class Holography implements MouseWheelListener, KeyListener{

	Image hologram;
	JFrame frame;
	long method;
	int block=15;
	int index;
	
	public Holography () {
		
		frame = new JFrame();
		frame.setSize(500,500);
		
		frame.setLayout(null);
		
		hologram = new Image();
		//hologram = new Image(2040,1088);
		//hologram = new Image(400,400);
		
	}
	
	// native c methods
	native long initReconstruct(int width, int height, byte[] img);
	native long initAutoFocus(int width, int height, byte[] img);
	native byte[] reconstructAt(long mp, float dist);
	native byte[] zmap(long mp, float startDist, float step, int n);
	
	/*
	 * read tiff image initialize reconstruction
	 */
	void init() {
		
		final JFileChooser fc = new JFileChooser();
		
		fc.addActionListener(new ActionListener(){

            public void actionPerformed(ActionEvent e){
            	File file = fc.getSelectedFile();
            	
				hologram.readImage(file);
				
				hologram.show();
				
				
            }});
		
		fc.showOpenDialog(frame);
		
		//method = initReconstruct(hologram.width, hologram.height, hologram.pixel);
		method = initAutoFocus(hologram.width, hologram.height, hologram.pixel);
		//hologram.zeroPad(block, block);

	}
	
	/*
	 * Reconstruct at certain dist
	 */
	public void ReconstructAt(float dist) {
		
		hologram.pixel = reconstructAt(method, dist);
	}
	
	/*
	 * Generate 3D information
	 */
	public void build3D(int size) {

		Build3D b3d = new Build3D(hologram, 50, this);
		//b3d.reconStack();
		b3d.zMap();
	}
	
	/*
	 * Interactive reconstruction
	 */
	public void reconShow() {
		
		index = 0;
		float dist = 4.f*(float)index/1000.0f + 0.0f;
		ReconstructAt(dist);
		hologram.show();
		
        //hologram.iw.addMouseWheelListener(this); 
        hologram.iw.addKeyListener(this);
   
	}
	
	public static void main(String[] args) {
		
		Holography holography = new Holography();
		
		holography.init();
		
		//holography.reconShow();
		holography.build3D(50);

	}
	
public void mouseWheelMoved(MouseWheelEvent e) {
	    
		int notches = e.getWheelRotation(); 
		
		while (notches > 0) {
			index++;
			
		}
		
		while (notches < 0) {
			index--;
			
		}
		
	}

	@Override
	public void keyPressed(KeyEvent e) {
		
		int key = e.getKeyCode();
		
		int oldindex=index;
		if (key==38) {
			if (index < 49) index++;
		} else if (key==40) {
			if (index>0) index--;
		}
		if (index != oldindex) {
			float dist = 4.f*(float)index/1000.0f + 0.0f;
			ReconstructAt(dist);
			hologram.update();
		}
	}

	@Override
	public void keyReleased(KeyEvent arg0) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void keyTyped(KeyEvent e) {
		
			
	}
	
	static {
	    System.loadLibrary ( "holography" );
	}

}
