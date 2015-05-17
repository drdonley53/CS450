/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package classifier;

import java.util.Arrays;
import java.util.Enumeration;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author daviddonley
 */
public class ID3 extends Classifier{
    
    Instances data;
    
    @Override
    public void buildClassifier(Instances dataset) throws Exception {
       data = dataset;
       createTheTree();
       DrawTheTree();
    }
    
     @Override
    public double classifyInstance(Instance i) throws Exception {
        return 0;
    }
    
    private void createTheTree() throws Exception{
        Enumeration getClasses = data.classAttribute().enumerateValues();
        Enumeration attributes = data.enumerateAttributes();
        
        //This is how i handeled missing data. I would skip it and move onto the
        //Next set of data.
        if (data.numInstances() == 0) {
            return;
        } else if (data.numInstances() == data.classAttribute().numValues()) {
            return;
        }
        
        double[] gain = new double[data.numAttributes()];

        // claculate the inforgain.
        Enumeration enumer = data.enumerateAttributes();
        while (enumer.hasMoreElements()) {
            Attribute attribute = (Attribute) enumer.nextElement();
            gain[attribute.index()] = computeInfoGain(attribute.index());
        }
        Attribute mostInformativeAttribute = data.attribute(Utils.maxIndex(gain));
    }
    
    private void DrawTheTree(){
        
        
        
    }
    
    private double computeInfoGain(int index) throws Exception {
        double maxEntropy = getTotalEntropy();       
        double[] classValueCounts = new double[data.numClasses()];
        double attributeValueProbability;
        double attributeValueEntropy = 0;
        double classValueProbability;
        double totalAttributeValuesEntropy = 0;
        double gain;
        int attributeValueCount = 0;
        int attributeValueIndex;
        int classValueIndex;

        data.sort(index);

        for (int i = 0; i < data.numInstances(); i++) {
            if (i + 1 < data.numInstances() && data.instance(i).value(index) == data.instance(i + 1).value(index)) {
                classValueIndex = (int) data.instance(i).classValue();
                classValueCounts[classValueIndex]++;
                attributeValueCount++;
            } else {
                attributeValueProbability = attributeValueCount / (double) data.numInstances();
                for (int k = 0; k < data.numClasses(); k++) {
                    classValueProbability = classValueCounts[k] / attributeValueCount;
                    attributeValueEntropy += calcEntropy(classValueProbability);
                }
                totalAttributeValuesEntropy += (attributeValueEntropy * attributeValueProbability);
                attributeValueCount = 0;
                attributeValueEntropy = 0;
                Arrays.fill(classValueCounts, 0);
            }
        }

        gain = maxEntropy - totalAttributeValuesEntropy;
        return gain;
    }
    
    
     /*******************************
      * @param data
      * @param att
      * @return double
      * @throws Exception
      * This function calls the entropy This was sudocode from the book
      */
    
     private double calcEntropy(double p) {
        double result;
        if (p != 0) {
            result = -p * Utils.log2(p);
        } else {
            result = 0;
        }
        return result;
    }
    
     private double getTotalEntropy() {
         
        int classIndex;
        double[] classCounts = new double[data.numClasses()];
        for (int i = 0; i < data.numInstances(); i++) {
            classIndex = (int) data.instance(i).classValue();
            classCounts[classIndex]++;
        }

        // calculate total entropy for class set
        double totalEntropy = 0;
        double classProbability;
        for (int i = 0; i < data.numClasses(); i++) {
            classProbability = classCounts[i] / data.numInstances();
            totalEntropy += calcEntropy(classProbability);
        }
        return totalEntropy;
    }
}
