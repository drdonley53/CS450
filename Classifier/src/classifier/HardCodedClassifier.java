/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package classifier;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;


/**
 *
 * @author daviddonley
 */

public class HardCodedClassifier extends Classifier{
    
    @Override
    public void buildClassifier(Instances dataset) throws Exception {
        
    }
    
    @Override
    public double classifyInstance(Instance i) throws Exception {
        return 0;
    }
}
