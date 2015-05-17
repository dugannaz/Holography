
import org.jzy3d.analysis.AbstractAnalysis;
import org.jzy3d.analysis.AnalysisLauncher;
import org.jzy3d.chart.factories.AWTChartComponentFactory;
import org.jzy3d.colors.Color;
import org.jzy3d.colors.ColorMapper;
import org.jzy3d.colors.colormaps.ColorMapRainbow;
import org.jzy3d.maths.Range;
import org.jzy3d.maths.Scale;
import org.jzy3d.plot3d.builder.Builder;
import org.jzy3d.plot3d.builder.Mapper;
import org.jzy3d.plot3d.builder.concrete.OrthonormalGrid;
import org.jzy3d.plot3d.primitives.Shape;
import org.jzy3d.plot3d.rendering.canvas.Quality;

public class CudaPlot extends AbstractAnalysis {
	
	Image img;
	float[][] data;
	
    public static void plot(Image img) throws Exception {
    	
        AnalysisLauncher.open(new CudaPlot(img));
    }

    public CudaPlot(Image img) {
    	this.img = img;
    }
    
    @Override
    public void init() {
        // Define a function to plot
    	
    	data = new float[img.height][img.width]; 
    	
    	for (int i=0;i<img.height; i++)
    		for (int j=0;j<img.width; j++)
    			data[i][j] = img.pixel[i*img.width+j];
    	
        Mapper mapper = new Mapper() {
            public double f(double x, double y) {
            	
            	//int ix = (int)x+1100;
            	//int iy = (int)y+544;
            	int ix = (int)x+img.width/2;
            	int iy = (int)y+img.height/2;
           
            	//if (data[iy][ix] > 50 && data[iy][ix] < 70)
            		return data[iy][ix]/100.0;
            	//else
            	//	return 0.0;
            }
        };

        // Define range and precision for the function to plot
        //Range range = new Range(-540, 540);
        int minDim = java.lang.Math.min(img.width, img.height);
        minDim = (minDim/2)-2;
        Range range = new Range(-minDim, minDim);
       
        int steps = 100;

        // Create the object to represent the function over the given range.
        final Shape surface = Builder.buildOrthonormal(new OrthonormalGrid(range, steps, range, steps), mapper);
        surface.setColorMapper(new ColorMapper(new ColorMapRainbow(), surface.getBounds().getZmin(), surface.getBounds().getZmax(), new Color(1, 1, 1, .5f)));
        surface.setFaceDisplayed(true);
        surface.setWireframeDisplayed(false);
       

        // Create a chart
        chart = AWTChartComponentFactory.chart(Quality.Advanced, getCanvasType());
        chart.getScene().getGraph().add(surface);
        
		chart.setScale(new Scale(0,2), false);
    }
}
