/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package gr.iti.mklab.reveal.forensics.maps.ela;

import gr.iti.mklab.reveal.forensics.util.Util;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 *
 * @author markzampoglou
 */
public class ELAExtractor {

    public BufferedImage displaySurface = null, displaySurface_temp = null;
    public double elaMin;
    public double elaMax;
    public int sc_width = 600;
    public int sc_height = 600;

    public ELAExtractor(String fileName) throws IOException {
        getJPEGELA(fileName);
    }

    private BufferedImage getJPEGELA(String fileName) throws IOException {

        int quality=75;
        int displayMultiplier=20;

        BufferedImage origImage;
        origImage = ImageIO.read(new File(fileName));

        BufferedImage recompressedImage = Util.recompressImage(origImage, quality);
        double[][][] imageDifference = Util.getImageDifferenceD(origImage, recompressedImage);
        elaMin =Util.minDouble3DArrayD(imageDifference);
        elaMax =Util.maxDouble3DArrayD(imageDifference);
        int[][][] intDifference = new int[imageDifference.length][imageDifference[0].length][imageDifference[0][0].length];
        for (int ii=0;ii<imageDifference.length;ii++){
            for (int jj=0;jj<imageDifference[0].length;jj++){
                for (int kk=0;kk<imageDifference[0][0].length;kk++){
                    intDifference[ii][jj][kk]=(int) Math.sqrt(imageDifference[ii][jj][kk])*displayMultiplier;
                    if (intDifference[ii][jj][kk]>255){
                        intDifference[ii][jj][kk]=255;
                    }
                }
            }
        }
        displaySurface_temp =Util.getBufferedIm(intDifference);
        
        // Scale the image result in order to save execution time
        // The scaled map is not misguided
        if (displaySurface_temp.getHeight() > displaySurface_temp.getWidth()){
			if (displaySurface_temp.getHeight() > sc_height){
				sc_width = (sc_height * displaySurface_temp.getWidth())/ displaySurface_temp.getHeight();
				displaySurface = Util.scaleImage(displaySurface_temp, sc_width, sc_height);
			}else{
				displaySurface = displaySurface_temp;
			}
		}else{
			if (displaySurface_temp.getWidth() > sc_width){
				sc_height = (sc_width * displaySurface_temp.getHeight())/ displaySurface_temp.getWidth(); 
				displaySurface = Util.scaleImage(displaySurface_temp, sc_width, sc_height);				
			}else{
				displaySurface = displaySurface_temp;
			}
		}
        return origImage;
    }
}
