/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package classifier;

import java.util.Random;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;
/**
 *
 * @author daviddonley
 */
public class Classifier {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        
        
    DataSource source = new DataSource("/Users/daviddonley/Desktop/iris.csv");
    Instances data = source.getDataSet();
    
    if (data.classIndex() == -1)
        data.setClassIndex(data.numAttributes() - 1);
    
    //Randomize the Data
    data.randomize(new Random(1));
    
    //Split the data into 70 percent for the training.
    RemovePercentage seventy = new RemovePercentage();
    seventy.setPercentage(70);
    seventy.setInputFormat(data);
    //Assign the seventy percent of the data to the instance
    Instances training = Filter.useFilter(data, seventy);
    
    //Split the other data into the remainding percent
    seventy.setInvertSelection(true);
    seventy.setInputFormat(data);
    //Assign the 30 persent to the testing
    Instances test = Filter.useFilter(data, seventy);
    
    //Call the HardCodedClasifier
    HardCodedClassifier hardcode = new HardCodedClassifier();
    hardcode.buildClassifier(training);
    
    //Use Evaluation to evaluate the training
    Evaluation evaluate = new Evaluation(training);
    evaluate.evaluateModel(hardcode, test);
    
    //Print the results
    System.out.println(evaluate.toSummaryString("\nDATA RESULTS\n", true));
    }
    
}











