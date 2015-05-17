
/*
 * Generate 3D information of an holographic image.
 */

public class Build3D {

	Image hologram;
	Image zmap;
	int filterSize;
	Holography holography;
	int size;
	Image[] stack;
	
	public Build3D(Image image, int stackSize, Holography hol) {
		
		//hologram = new Image(width, height);
		size = stackSize;
		zmap = new Image(image.width, image.height);
		stack = new Image[size];
		for (int i=0; i<size; i++)
			stack[i] = new Image(image.width, image.height);
		holography = hol;
	}
	
	/*
	 * Reconstruct for a stack of distances and show results continuously
	 */
	void reconStack() {
		
		for (int i=0; i<size; i++) {
			float dist = 4.f*(float)i/1000.0f + 0.0f;
			holography.ReconstructAt(dist);
			stack[i].pixel = holography.hologram.pixel;
		}
		
		zmap.show();
		for (int i=0; i<size; i++) {
			try {
			    Thread.sleep(500);            
			} catch(InterruptedException ex) {
			    Thread.currentThread().interrupt();
			}
			zmap.update(stack[i]);
		}
		
	}
	
	/*
	 * Reconstruct for a stack of images and autofocus to generate zmap for
	 * focused image.
	 */
	void zMap() {
		
		zmap.pixel = holography.zmap(holography.method, 0.f, 4.f*1.f/1000.0f, 50);
		zmap.show();
		try {
			CudaPlot.plot(zmap);
		} catch (Exception ef) {
			ef.printStackTrace();
		}
	}
}
