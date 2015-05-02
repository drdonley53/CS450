/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package classifier;

import static java.lang.Math.abs;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author daviddonley
 */
public class KNearest extends Classifier{
    
    private Instances training;
    private Integer k;
    
    public KNearest(Integer k)
    {
        this.k = k;
    }
    
    public int calculateDistance(Instance test, Instance train){
        double numOne = 0;
        double finalNum = 0;
        int FinalNum = 0;
        
        int attributes = train.numAttributes();
        for(int i = 0; i < attributes; i++){
            
            double numTwo = 0;
            
            if(train.attribute(i).isNumeric()){
                numTwo = Math.abs(train.value(i) - test.value(i));
            }
            else{
                    numTwo = 1;
            }
                numOne += Math.pow(numTwo, train.numAttributes() - 1);
        }
        
        finalNum = Math.pow(numOne, 1.0/(train.numAttributes() - 1));
        FinalNum = (int)finalNum;
        return FinalNum;
    }
    
    public double timeToClassify(Map<Integer, Double> mapping){
        
        int count = 0;
        double index;        
        int array[] = new int[training.numClasses()];
        for (Map.Entry<Integer, Double> entry : mapping.entrySet()) {
            if (count >= k)
                break;       
            index = entry.getValue();
            array[(int)index]++;                                  
            count++;
        }
        
        int max = 0;
        int tempClassTally;
        for (int i = 0; i < training.numClasses(); i++) {
            tempClassTally = array[i];
            if (tempClassTally > array[max])
                max = i;
        }  
        return max;
    }
    
    
    @Override
    public void buildClassifier(Instances dataset) throws Exception {
        training = new Instances(dataset);
    }
    
    @Override
    public double classifyInstance(Instance test) throws Exception {
        
        Map<Integer, Double> map = new TreeMap<>();
        int number = 0;
        for(int i = 0; i < training.numInstances(); i++)
        {
            number = calculateDistance(test, training.instance(i));
            map.put(number, training.instance(i).classValue());
        }           
        
        return timeToClassify(map);
    }
    
}
