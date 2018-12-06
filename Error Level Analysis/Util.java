/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package gr.iti.mklab.reveal.forensics.util;

import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;

import javax.imageio.ImageWriteParam;
import javax.imageio.ImageWriter;
// import javax.imageio.stream.ImageOutputStream;
import javax.imageio.stream.MemoryCacheImageOutputStream;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.DoubleStream;

import javax.imageio.IIOImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.OptionalDouble;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

/**
 *
 * @author markzampoglou
 */
public class Util {

    public static BufferedImage recompressImage(BufferedImage imageIn, int quality) {
        // Apply in-memory JPEG compression to a BufferedImage given a quality setting (0-100)
        // and return the resulting BufferedImage
        float fQuality = (float) (quality / 100.0);
        BufferedImage outputImage = null;
        try {
            ImageWriter writer;
            Iterator<ImageWriter> iter = ImageIO.getImageWritersByFormatName("jpeg");
            writer = iter.next();
            ImageWriteParam iwp = writer.getDefaultWriteParam();
            iwp.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
            iwp.setCompressionQuality(fQuality);
            byte[] imageInByte;
            try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
                MemoryCacheImageOutputStream mcios = new MemoryCacheImageOutputStream(baos);
                writer.setOutput(mcios);
                IIOImage tmpImage = new IIOImage(imageIn, null, null);
                writer.write(null, tmpImage, iwp);
                writer.dispose();
                baos.flush();
                imageInByte = baos.toByteArray();
            }
            InputStream in = new ByteArrayInputStream(imageInByte);
            outputImage = ImageIO.read(in);
        } catch (Exception ex) {
            Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
        }
        return outputImage;
    }

    public static int[][][] getRGBArray(BufferedImage imageIn) {
        // possible 10-fold speed increase in:
        // http://stackoverflow.com/questions/6524196/java-get-pixel-array-from-image
        // (but ensure all major bases are covered)
        int ImW = imageIn.getWidth();
        int ImH = imageIn.getHeight();
        Color tmpColor;
        int[][][] rgbValues = new int[3][ImW][ImH];

        for (int ii = 0; ii < ImW; ii++) {
            for (int jj = 0; jj < ImH; jj++) {
                tmpColor = new Color(imageIn.getRGB(ii, jj));
                rgbValues[0][ii][jj] = tmpColor.getRed();
                rgbValues[1][ii][jj] = tmpColor.getGreen();
                rgbValues[2][ii][jj] = tmpColor.getBlue();
            }
        }
        return rgbValues;
    }

    public static BufferedImage getBufferedIm(int[][][] rgbValues) {
        int ImW = rgbValues[0].length;
        int ImH = rgbValues[0][0].length;
        BufferedImage ImageOut = new BufferedImage(ImW, ImH, 5); // 5 for PNG;
        Color tmpColor;
        for (int ii = 0; ii < ImW; ii++) {
            for (int jj = 0; jj < ImH; jj++) {
                tmpColor = new Color((int) Math.round(rgbValues[0][ii][jj]), (int) Math.round(rgbValues[1][ii][jj]), (int) Math.round(rgbValues[2][ii][jj]));
                ImageOut.setRGB(ii, jj, tmpColor.getRGB());
            }
        }
        return ImageOut;
    }

    public static float[][][] getImageDifference(BufferedImage image1, BufferedImage image2) {
        Color tmpColor1, tmpColor2;
        int width = image1.getWidth();
        int height = image1.getHeight();
        float[][][] outputMap=new float[3][width][height];
        for (int ii = 0; ii < width; ii++) {
            for (int jj = 0; jj < height; jj++) {
                tmpColor1 = new Color(image1.getRGB(ii, jj));
                tmpColor2 = new Color(image2.getRGB(ii, jj));
                outputMap[0][ii][jj] = (float) (tmpColor1.getRed() - tmpColor2.getRed()) * (tmpColor1.getRed() - tmpColor2.getRed());
                outputMap[1][ii][jj] = (float) (tmpColor1.getGreen() - tmpColor2.getGreen()) * (tmpColor1.getGreen() - tmpColor2.getGreen());
                outputMap[2][ii][jj] = (float) (tmpColor1.getBlue() - tmpColor2.getBlue()) * (tmpColor1.getBlue() - tmpColor2.getBlue());
            }
        }
        return outputMap;
    }
    
    public static double[][][] getImageDifferenceD(BufferedImage image1, BufferedImage image2) {
        Color tmpColor1, tmpColor2;
        int width = image1.getWidth();
        int height = image1.getHeight();
        double red_temp, green_temp, blue_temp;
        
        double[][][] outputMap=new double[3][width][height];
        for (int ii = 0; ii < width; ii++) {
            for (int jj = 0; jj < height; jj++) {
                tmpColor1 = new Color(image1.getRGB(ii, jj));
                tmpColor2 = new Color(image2.getRGB(ii, jj));
                red_temp = tmpColor1.getRed() - tmpColor2.getRed(); 
                green_temp = tmpColor1.getGreen() - tmpColor2.getGreen();
                blue_temp = tmpColor1.getBlue() - tmpColor2.getBlue();
                outputMap[0][ii][jj] = (double) (red_temp) * (red_temp);               
                outputMap[1][ii][jj] = (double) (green_temp) * (green_temp);
                outputMap[2][ii][jj] = (double) (blue_temp) * (blue_temp);     
            }
        }
        return outputMap;
    } 

    public static float[][][] getResizedImageDifference(BufferedImage image1, BufferedImage image2, int newWidth, int newHeight){
        // Calculate the subsampled difference between two buffered images
        float[][][] outputMap=new float[3][newWidth][newHeight];
        float widthModifier=(float)image1.getWidth()/newWidth;
        float heightModifier=(float)image1.getHeight()/newHeight;
        Color tmpColor1, tmpColor2;
        for (int ii = 0; ii < newHeight; ii++) {
            for (int jj = 0; jj < newWidth; jj++) {
                try {
                    tmpColor1 = new Color(image1.getRGB(Math.round(jj * widthModifier),Math.round(ii * heightModifier)));
                    tmpColor2 = new Color(image2.getRGB(Math.round(jj * widthModifier),Math.round(ii * heightModifier)));
                    outputMap[0][jj][ii] = (float) (tmpColor1.getRed() - tmpColor2.getRed()) * (tmpColor1.getRed() - tmpColor2.getRed());
                    outputMap[1][jj][ii] = (float) (tmpColor1.getGreen() - tmpColor2.getGreen()) * (tmpColor1.getGreen() - tmpColor2.getGreen());
                    outputMap[2][jj][ii] = (float) (tmpColor1.getBlue() - tmpColor2.getBlue()) * (tmpColor1.getBlue() - tmpColor2.getBlue());
                } catch(Exception e) {
                    System.out.println(newHeight + " " + newWidth + " " + image1.getHeight() + " " + image1.getWidth() + " " + ii + " " + jj + " " + Math.round(ii * heightModifier) + " " + Math.round(jj * widthModifier) + " " + heightModifier + " " + widthModifier);
                    e.printStackTrace();
                    return outputMap;
                }
            }
        }
        return outputMap;
    }

    public static int[][] get2DArrayDifference(int[][] Im1, int[][] Im2){
        // Subtract two int images and return the result
        int[][] imOut = new int[Im1.length][Im1[0].length];
        for (int ii=0;ii<Im1.length;ii++){
            for (int jj=0;jj<Im1[0].length;jj++){
                imOut[ii][jj]=Im1[ii][jj]-Im2[ii][jj];
            }
        }
        return imOut;
    }

    public static int[][] get2DArraySum(int[][] Im1, int[][] Im2){
        // Add two int images and return the result
        int[][] imOut = new int[Im1.length][Im1[0].length];
        for (int ii=0;ii<Im1.length;ii++){
            for (int jj=0;jj<Im1[0].length;jj++){
                imOut[ii][jj]=Im1[ii][jj]+Im2[ii][jj];
            }
        }
        return imOut;
    }

    public static float[][] meanFilterSingleChannelImage(float[][] imIn, int meanFilterSize) {
        // Mean filter a 2D double array
        // meanFilterSize is assumed to be odd
        int offset = (meanFilterSize - 1) / 2;
        int numOfFilterElements = meanFilterSize * meanFilterSize;
        int imWidth = imIn.length;
        int imHeight = imIn[0].length;
        DescriptiveStatistics blockValues;
        float[][] filteredImage = new float[imWidth - 2 * offset][imHeight - 2 * offset];
        float sum;
        for (int ii = offset; ii <= imWidth - meanFilterSize + offset; ii = ii + 1) {
            sum = 0;
            for (int B_ii = ii - offset; B_ii < ii + offset + 1; B_ii++) {
                for (int B_jj = 0; B_jj < 2 * offset + 1; B_jj++) {
                    sum = sum + imIn[B_ii][B_jj];
                }
            }
            filteredImage[ii - offset][0] = sum / numOfFilterElements;
            for (int jj = offset + 1; jj <= imHeight - meanFilterSize + offset; jj = jj + 1) {
                for (int B_ii = ii - offset; B_ii < ii + offset + 1; B_ii++) {
                    sum = sum - imIn[B_ii][jj - offset - 1];
                    sum = sum + imIn[B_ii][jj + offset];
                }
                filteredImage[ii - offset][jj - offset] = sum / numOfFilterElements;
            }
        }
        return filteredImage;
    }

    public static int[][] sumFilterSingleChannelVert(int[][] imIn, int filterSize) {
        // Sum filter a 2D double array across columns
        // filterSize should be odd
        int offset = (filterSize - 1) / 2;
        int imWidth = imIn.length;
        int imHeight = imIn[0].length;
        int[][] filteredImage = new int[imWidth][imHeight - 2 * offset];
        int sum;
        for (int ii = 0; ii < imWidth ; ii++) {
            sum = 0;
            for (int B_jj = 0; B_jj < 2 * offset + 1; B_jj++) {
                    sum = sum + imIn[ii][B_jj];
            }
            filteredImage[ii][0] = sum;
            for (int jj = offset + 1; jj <= imHeight - filterSize + offset; jj = jj + 1) {
                sum = sum - imIn[ii][jj - offset - 1];
                sum = sum + imIn[ii][jj + offset];
                filteredImage[ii][jj - offset] = sum;
            }
        }
        return filteredImage;
    }

    public static int[][] sumFilterSingleChannelHorz(int[][] imIn, int filterSize) {
        // Sum filter a 2D double array across rows
        // filterSize should be odd
        int offset = (filterSize - 1) / 2;
        int imWidth = imIn.length;
        int imHeight = imIn[0].length;
        int[][] filteredImage = new int[imWidth-2*offset][imHeight];
        int Sum;
        for (int jj = 0; jj < imHeight ; jj++) {
            Sum = 0;
            for (int B_ii = 0; B_ii < 2 * offset + 1; B_ii++) {
                Sum = Sum + imIn[B_ii][jj];
            }
            filteredImage[0][jj] = Sum;
            for (int ii = offset + 1; ii <= imWidth - filterSize + offset; ii = ii + 1) {
                Sum = Sum - imIn[ii - offset - 1][jj];
                Sum = Sum + imIn[ii + offset][jj];
                filteredImage[ii - offset][jj] = Sum;
            }
        }
        return filteredImage;
    }

    public static float[][] medianFilterSingleChannelImage(double[][] imIn, int medianFilterSize) {
        // Median filter a 2D double array
        // medianFilterSize should be odd
        int offset = (medianFilterSize - 1) / 2;
        int imWidth = imIn.length;
        int imHeight = imIn[0].length;
        DescriptiveStatistics blockValues;

        float[][] filteredImage = new float[imWidth - 2 * offset][imHeight - 2 * offset];

        for (int ii = offset; ii <= imWidth - medianFilterSize + offset; ii = ii + 1) {
            for (int jj = offset; jj <= imHeight - medianFilterSize + offset; jj = jj + 1) {
                blockValues = new DescriptiveStatistics();
                for (int B_ii = ii - offset; B_ii < ii + offset + 1; B_ii++) {
                    for (int B_jj = jj - offset; B_jj < jj + offset + 1; B_jj++) {
                        blockValues.addValue(Math.abs(imIn[B_ii][B_jj]));
                    }
                }
                filteredImage[ii - offset][jj - offset] = (float) blockValues.getPercentile(50);
            }
        }
        return filteredImage;
    }

    public static int[][] medianFilterSingleChannelVert(int[][] imIn, int filterSize) {
      	 // Median filter a 2D double array across columns
          // filterSize should be odd
           int offset = (filterSize - 1) / 2;
           int imWidth = imIn.length;
           int imHeight = imIn[0].length;        
           int[][] filteredImage = new int[imWidth][imHeight - 2 * offset];
           int counter = 0;
           for (int jj = 0; jj < imWidth; jj++) {
              	for (int ii = 0; ii < imHeight - 2 * offset; ii++){
             		counter = 0;
             		int[] temp_array = new int[filterSize];
           		for (int kk = ii; kk < (ii + filterSize); kk++){
           			temp_array[counter] = imIn[jj][kk];        			
           			counter = counter + 1;        			
           		}      
           		Arrays.sort(temp_array);
           		filteredImage[jj][ii] = temp_array[offset + 1];           		
           	}
           }   	
           return filteredImage;
      }

	public static int[][] medianFilterSingleChannelHorz(int[][] imIn, int FilterSize) {
	        // Median filter a 2D double array across rows
	        // FilterSize should be odd   	
	        int offset = (FilterSize - 1) / 2;
	        int imWidth = imIn.length;
	        int imHeight = imIn[0].length;        
	        int[][] filteredImage = new int[imWidth - 2 * offset][imHeight];
	        int counter = 0;
	        for (int jj = 0; jj < imHeight; jj++) {
	           	for (int ii = 0; ii < imWidth - 2 * offset; ii++){
	          		counter = 0;
	          		int[] temp_array = new int[FilterSize];
	        		for (int kk = ii; kk < (ii + FilterSize); kk++){
	        			temp_array[counter] = imIn[kk][jj];        			
	        			counter = counter + 1;        			
	        		}      
	        		Arrays.sort(temp_array);
	        		filteredImage[ii][jj] = temp_array[offset + 1];        		
	        	}
	        }   	
	        return filteredImage;
	    }

    public static float[][][] meanFilterThreeChannelImage(float[][][] ImIn, int meanFilterSize) {
        // Apply mean filtering to each image channel separately
        int offset = (meanFilterSize - 1) / 2;
        float[][][] filteredIm = new float[3][ImIn[0].length - 2 * offset][ImIn[0][0].length - 2 * offset];
        filteredIm[0] = meanFilterSingleChannelImage(ImIn[0], meanFilterSize);
        filteredIm[1] = meanFilterSingleChannelImage(ImIn[1], meanFilterSize);
        filteredIm[2] = meanFilterSingleChannelImage(ImIn[2], meanFilterSize);
        return filteredIm;
    }

    public static float[][] convertToSingleChannel(float[][][] imIn) {
        // Convert three-channel image to single channel by averaging
        int imWidth = imIn[0].length;
        int imHeight = imIn[0][0].length;
        float[][] outIm = new float[imWidth][imHeight];
        for (int ii = 0; ii < imWidth; ii++) {
            for (int jj = 0; jj < imHeight; jj++) {
                outIm[ii][jj] = (imIn[0][ii][jj] + imIn[1][ii][jj] + imIn[2][ii][jj]) / 3;
            }
        }
        return outIm;
    }

    public static float SingleChannelMean(float[][] imIn) {
        // Calculate the mean value of a single channel image
        int imWidth = imIn.length;
        int imHeight = imIn[0].length;
        float sum = 0;
        for (int ii = 0; ii < imWidth; ii++) {
            for (int jj = 0; jj < imHeight; jj++) {
                sum = sum + imIn[ii][jj];
            }
        }
        float mean = sum / (imWidth * imHeight);
        return mean;
    }

    public static List<Integer> getArrayLocalMinima(float[] valuesIn) {
        // Calculate the local minima in a 2D array
        // The first and last values are excluded, flat
        List<Integer> minima = new ArrayList();
        for (int ii = 1; ii < valuesIn.length - 1; ii++) {
            if ((valuesIn[ii - 1] > valuesIn[ii]) & (valuesIn[ii + 1] > valuesIn[ii])) {
                minima.add(ii);
            }
        }
        return minima;
    }

    public static double[][] normalizeIm(float[][] imIn) {
        // Normalize single-channel image pixel values to [0, 1]
        int imWidth = imIn.length;
        int imHeight = imIn[0].length;
        double imOut[][] = new double[imWidth][imHeight];
        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;
        double colMin, colMax;
        for (float[] imInRow : imIn) {
            List<Float> b = Arrays.asList(ArrayUtils.toObject(imInRow));
            colMin = (float) Collections.min(b);
            if (colMin < min) {
                min = colMin;
            }
            colMax = (float) Collections.max(b);
            if (colMax > max) {
                max = colMax;
            }
        }
        double spread = max - min;
        for (int ii = 0; ii < imWidth; ii++) {
            for (int jj = 0; jj < imHeight; jj++) {
                imOut[ii][jj] = (imIn[ii][jj] - min) / spread;
            }
        }
        return imOut;
    }
    
    public static double[][] normalizeIm(double[][] imIn) {
        // Normalize single-channel image pixel values to [0, 1]
        int imWidth = imIn.length;
        int imHeight = imIn[0].length;
        double imOut[][] = new double[imWidth][imHeight];
        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;
        double colMin, colMax;
        for (double[] imInRow : imIn) {
        	List<Double> b = Arrays.asList(ArrayUtils.toObject(imInRow));
            colMin = (double) Collections.min(b);
            if (colMin < min) {
                min = colMin;
            }
            colMax = (double) Collections.max(b);
            if (colMax > max) {
                max = colMax;
            }
        }
        double spread = max - min;
        for (int ii = 0; ii < imWidth; ii++) {
            for (int jj = 0; jj < imHeight; jj++) {
                imOut[ii][jj] = (imIn[ii][jj] - min) / spread;
            }
        }
        return imOut;
    }

    public static BufferedImage visualizeWithJet(double[][] inputGrayImage) {
        // Take a [0,1] single-channel image and return a Jet visualization
        double[][] map = JetMap.colorMap;
        BufferedImage outIm = new BufferedImage(inputGrayImage.length, inputGrayImage[0].length, 5);
        Color rgb;
        byte bytevalue;
        for (int ii = 0; ii < inputGrayImage.length; ii++) {
            for (int jj = 0; jj < inputGrayImage[0].length; jj++) {
                bytevalue = (byte) Math.round(inputGrayImage[ii][jj] * 63);
                rgb = new Color((float) map[bytevalue][0], (float) map[bytevalue][1], (float) map[(byte) Math.round(inputGrayImage[ii][jj]) * 63][2]);
                outIm.setRGB(ii, jj, rgb.getRGB());
            }
        }
        return outIm;
    }

    public static double[][] blockNoiseVar(double[][] inputMap, int blockSize) {
        // Calculate the block variance of a noise map
        int blockedWidth = (int) Math.floor(inputMap.length / blockSize) * blockSize;
        int blockedHeight = (int) Math.floor(inputMap[0].length / blockSize) * blockSize;
        double[][] blockedIm = new double[blockedWidth / blockSize][blockedHeight / blockSize];
        DescriptiveStatistics blockValues;
        for (int ii = 0; ii < blockedWidth; ii = ii + blockSize) {
            for (int jj = 0; jj < blockedHeight; jj = jj + blockSize) {
                blockValues = new DescriptiveStatistics();
                for (int B_ii = ii; B_ii < ii + blockSize; B_ii++) {
                    for (int B_jj = jj; B_jj < jj + blockSize; B_jj++) {
                        blockValues.addValue(Math.abs(inputMap[B_ii][B_jj]));
                    }
                }
                blockedIm[ii / blockSize][jj / blockSize] = Math.sqrt(blockValues.getPercentile(50) / 0.6745);
            }
        }
        return blockedIm;
    }

    public static double[][] blockVar(double[][] inputIm, int blockSize) {
        // block variance of input int image
        int blockedWidth = (int) Math.floor(inputIm.length / blockSize) * blockSize;
        int blockedHeight = (int) Math.floor(inputIm[0].length / blockSize) * blockSize;
        double[][] blockedIm = new double[blockedWidth / blockSize][blockedHeight / blockSize];
        DescriptiveStatistics blockValues;
        for (int ii = 0; ii < blockedWidth; ii = ii + blockSize) {
            for (int jj = 0; jj < blockedHeight; jj = jj + blockSize) {
                blockValues = new DescriptiveStatistics();
                for (int B_ii = ii; B_ii < ii + blockSize; B_ii++) {
                    for (int B_jj = jj; B_jj < jj + blockSize; B_jj++) {
                        blockValues.addValue(Math.abs(inputIm[B_ii][B_jj]));
                    }
                }
                blockedIm[ii / blockSize][jj / blockSize] = blockValues.getPopulationVariance();
            }
        }
        return blockedIm;
    }

    public static String getImageFormat(File inputImage) throws IOException {
        // Return a string describing the image format
        String format = null;
        ImageInputStream iis = ImageIO.createImageInputStream(inputImage);
        Iterator<ImageReader> iter = ImageIO.getImageReaders(iis);
        if (!iter.hasNext()) {
            throw new RuntimeException("No readers found! I don't think this is an image file");
        }
        ImageReader reader = iter.next();
        format = reader.getFormatName();
        return format;
    }

    public static float minDouble2DArray(float[][] arrayIn) {
        // Calculate the minimum value of a 2D float array
        float min = Float.MAX_VALUE;
        float colMin;
        for (float[] arrayInRow : arrayIn) {
            List b = Arrays.asList(ArrayUtils.toObject(arrayInRow));
            colMin = (float) Collections.min(b);
            if (colMin < min) {
                min = colMin;
            }
        }
        return min;
    }
    
    public static double minDouble2DArray(double[][] arrayIn) {
        // Calculate the minimum value of a 2D float array
    	double min = Double.MAX_VALUE;
    	double colMin;
        for (double[] arrayInRow : arrayIn) {
            List b = Arrays.asList(ArrayUtils.toObject(arrayInRow));
            colMin = (double) Collections.min(b);
            if (colMin < min) {
                min = colMin;
            }
        }
        return min;
    }

    public static float maxDouble2DArray(float[][] arrayIn) {
        // Calculate the maximum value of a 2D float array
        float max = -Float.MAX_VALUE;
        float colMax;
        for (float[] arrayInRow : arrayIn) {
            List b = Arrays.asList(ArrayUtils.toObject(arrayInRow));
            colMax = (float) Collections.max(b);
            if (colMax > max) {
                max = colMax;
            }
        }
        return max;
    }
    
    public static double maxDouble2DArray(double[][] arrayIn) {
        // Calculate the maximum value of a 2D float array
    	double max = -Double.MAX_VALUE;
        double colMax;
        for (double[] arrayInRow : arrayIn) {
            List b = Arrays.asList(ArrayUtils.toObject(arrayInRow));
            colMax = (double) Collections.max(b);
            if (colMax > max) {
                max = colMax;
            }
        }
        return max;
    }

    public static float minDouble3DArray(float[][][] arrayIn) {
        // Calculate the minimum value of a 3D float array
        float min = Float.MAX_VALUE;
        float colMin;
        for (float[][] twoDInRow : arrayIn) {
            for (float[] arrayInRow : twoDInRow) {
                List b = Arrays.asList(ArrayUtils.toObject(arrayInRow));
                colMin = (float) Collections.min(b);
                if (colMin < min) {
                    min = colMin;
                }
            }
        }
        return min;
    }
    
    public static double minDouble3DArrayD(double[][][] arrayIn) {
        // Calculate the minimum value of a 3D float array
        double min = Double.MIN_VALUE;
        double colMin;
        for (double[][] twoDInRow : arrayIn) {
        		colMin = Arrays.stream(twoDInRow)
              	    .flatMapToDouble(a -> Arrays.stream(a))
              	    .min().getAsDouble();
        	 if (colMin < min) {
                  min = colMin;
              }
        }
        return min;
    }

    public static double maxDouble3DArray(float[][][] arrayIn) {
        // Calculate the maximum value of a 3D float array
        double max = -Double.MAX_VALUE;
        double colMax;
        for (float[][] twoDInRow : arrayIn) {
            for (float[] arrayInRow : twoDInRow) {
                List b = Arrays.asList(ArrayUtils.toObject(arrayInRow));
                colMax = (float) Collections.max(b);
                if (colMax > max) {
                    max = colMax;
                }
            }
        }
        return max;
    }
    
    public static double maxDouble3DArrayD(double[][][] arrayIn) {
        // Calculate the maximum value of a 3D float array
        double max = -Double.MAX_VALUE;
        double colMax;
        for (double[][] twoDInRow : arrayIn) {
        	colMax = Arrays.stream(twoDInRow)
                  	    .flatMapToDouble(a -> Arrays.stream(a))
                  	    .max()
                  	    .getAsDouble();
                if (colMax > max) {
                    max = colMax;
                }
        }
        return max;
    }

    public static float[][] scaleDownArray(float[][] origMap, int newWidth, int newHeight){
        // scale down an input float array using nearest neighbour interpolation
        float[][] outputMap=new float[newWidth][newHeight];
        float widthModifier=(float)origMap.length/newWidth;
        float heightModifier=(float)origMap[0].length/newHeight;
        for (int ii=0;ii<newWidth;ii++) {
            for (int jj=0;jj<newHeight;jj++) {
                try {
                    outputMap[ii][jj] = origMap[Math.round(ii * widthModifier)][Math.round(jj * heightModifier)];
                }catch (Exception e) {
                    System.out.println(outputMap.length + " " + outputMap[0].length + " " + origMap.length + " " + origMap[0].length + " " + ii + " " + jj + " " + Math.round(ii * heightModifier) + " " + Math.round(jj * widthModifier) + " " + heightModifier + " " + widthModifier);
                }
            }
        }
        return outputMap;
    }

    public static int[][] mirrorPadImage(int[][] imOrig, int padWidth, int padHeight){
        // pad an image using mirroring
        int[][] paddedY = new int[imOrig.length+2*padWidth][imOrig[0].length];
        for (int ii=padWidth;ii<imOrig.length+padWidth;ii++){
            for (int jj=0;jj<paddedY[0].length;jj++){
                paddedY[ii][jj]=imOrig[ii-padWidth][jj];
            }
        }
        //mirror
        for (int ii=0;ii<padWidth;ii++){
            for (int jj=0;jj<paddedY[0].length;jj++){
                paddedY[ii][jj]=paddedY[2*padWidth-ii][jj];
                paddedY[paddedY.length-ii-1][jj]=paddedY[paddedY.length-2*padWidth+ii-1][jj];
            }
        }
        int[][] padded= new int[imOrig.length+2*padWidth][imOrig[0].length+2*padHeight];
        for (int ii=0;ii<padded.length;ii++){
            for (int jj=padHeight;jj<paddedY[0].length+padHeight;jj++){
                padded[ii][jj]=paddedY[ii][jj-padHeight];
            }
        }
        //mirror
        for (int ii=0;ii<paddedY.length;ii++){
            for (int jj=0;jj<padHeight;jj++){
                padded[ii][jj]=padded[ii][2*padHeight-jj];
                padded[ii][padded[0].length-jj-1]=padded[ii][padded[0].length-2*padHeight+jj-1];
            }
        }
        return padded;
    }

    public static int rem(int x, int y) {
        // remainder of integer division
        int out;
        if (y != 0) {
            int n = (int) Math.floor(x / y);
            out = x - n * y;
        } else {
            out = 0;
        }
        if (x * out < 0) {
            out = out * -1;
        }
        return out;
    }
    
    /**
     *  author olgapapa 
     * 
     */
    public static int getRowTotalInt(int[][] array, int row) {
        int total = 0;
        for (int col = 0; col < array[row].length; col++) {
            total += array[row][col];
        }
        return total;
    }
    
    public static int getColumnTotalInt(int[][] array, int col) {
        int total = 0;
        for (int row = 0; row < array.length; row++) {
            total += array[row][col];
        }
        return total;
    }    
    
    public static double getRowTotal(double[][] array, int row) {
        double total = 0;
        for (int col = 0; col < array[row].length; col++) {
            total += array[row][col];
        }
        return total;
    }

    public static double getColumnTotal(double[][] array, int col) {
        double total = 0;
        for (int row = 0; row < array.length; row++) {
            total += array[row][col];
        }
        return total;
    }
    
    public static double getTotal(double[][] array) {
        double total = 0;

        for (int row = 0; row < array.length; row++) {
            for (int col = 0; col < array[row].length; col++) {
                total += array[row][col];
            }
        }
        return total;
    }

    public static double getAverage(double[][] array) {
        return getTotal(array) / getElementCount(array);
    }
    
    public static double getHighestInRow(double[][] array, int row) {
        double highest = array[row][0];

        for (int col = 1; col < array[row].length; col++) {
            if (array[row][col] > highest) {
                highest = array[row][col];
            }
        }
        return highest;
    }

    public static double getLowestInRow(double[][] array, int row) {
        double lowest = array[row][0];

        for (int col = 1; col < array[row].length; col++) {
            if (array[row][col] < lowest) {
                lowest = array[row][col];
            }
        }
        return lowest;
    }

    public static int getElementCount(double[][] array) {
        int count = 0;

        for (int row = 0; row < array.length; row++) {
            count += array[row].length;
        }
        return count;
    }
    
    /**
     * Calculate histogram
     */
    
    public static int[] createhistogram(double[] zmat, int bins) throws IOException{
    	
    	  OptionalDouble max = DoubleStream.of(zmat).max();
    	  OptionalDouble min = DoubleStream.of(zmat).min();
    	  int[] result = new int[bins];
    	  double binSize = (max.getAsDouble() - min.getAsDouble())/bins;
    	  
    	  for (double d : zmat) {
    	    int bin = (int) ((d - min.getAsDouble()) / binSize);
    	    if (bin < 0) { /* this data is smaller than min */ System.out.println("this data point is smaller than min " + d);}
    	    else if (bin >= bins) { /* this data point is bigger than max */
    	    	result[bin-1] += 1;    	
    	    	}
    	    else {
    	      result[bin] += 1;
    	    }
    	  }
    	
    	  return result;
    }    
    
    public static double[] normalize(int[] hist, int scale){
    	double[] histnorm = new double[hist.length];
    	int count = 0;
    	 for (double d : hist) {    		 
    		 histnorm[count] = d/scale;
    		 count = count + 1;
    	 }    	
    	return histnorm;   	
    }
    
    /**
     * Convert RGB [0..256] byte values and place them into a YCbCr [0..256]
     * byte array.
     */
    public static double[] RGBtoYCbCr(int r, int g, int b) {
        double Y, Cb, Cr;
			          
        	Y =  0.0627451  + 0.256788 * r + 0.5041294 * g + 0.09790588 * b; // 16/255
           	Cb = 0.50196    - 0.148f * r   - 0.291f * g    + 0.439f * b; // 128/255
        	Cr = 0.50196    + 0.439f * r   - 0.368f * g    - 0.071f * b;       	
        return new double[] {Y, Cb, Cr};
    }  
    
    public static double getMean(double[] data)
    {
    	int size = data.length;
        double sum = 0.0;
        for(double a : data)
            sum += a;
        return sum/size;
    }

    public static double getStdDev(double[] data)
    {
        return Math.sqrt(getVariance(data));
    }
    
    public static double getVariance(double[] data)
    {
    	int size = data.length;
        double mean = getMean(data);
        double temp = 0;
        for(double a :data)
            temp += (mean-a)*(mean-a);
        return temp/(size -1);
    }
    
    public static BufferedImage scaleImage(BufferedImage image, int width, int height) {
        assert (width > 0 && height > 0);
        // create image of new size
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics g = img.getGraphics();
        ((Graphics2D) g).setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);
      //  ((Graphics2D) g).setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(image, 0, 0, img.getWidth(), img.getHeight(), null);
        return img;
    }
    
    
}
