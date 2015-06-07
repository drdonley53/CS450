/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package classifier;
import static java.lang.Math.exp;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Vector;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author daviddonley
 */
public class MultiPrecep extends Classifier{
    
    private Instances data;
    
    private NeuralNet mNN;
    private List<Double> mOutputs;

    public MultiPrecep() {
        this.mOutputs = new ArrayList();
    }

    @Override
    public void buildClassifier(Instances i) throws Exception {
        data = i;
        buildNueralNetwork();
       
    }

    public void buildNueralNetwork() {
        mNN = new NeuralNet(data.numAttributes() - 1, data.numClasses(),
            1, 4);
        List<Double> attributeValues = new ArrayList();

        for (int i = 0; i < data.numInstances(); i++) {
            for (int j = 0; j < data.instance(i).numAttributes() - 1; j++) {
                attributeValues.add(data.instance(i).value(j));
            }
            mNN.forward(attributeValues);
            mNN.backPropagation(data.instance(i).value(data.instance(i).classIndex()));
            attributeValues.clear();
        }
    }
    
    
    @Override
    public double classifyInstance(Instance instance) {
        List<Double> attributeValues = new ArrayList();
        for (int i = 0; i < instance.numAttributes() - 1; i++) {
            attributeValues.add(instance.value(i));
        }
        mOutputs = mNN.forward(attributeValues);
        if (mOutputs.size() > 0) {
            return mOutputs.indexOf(Collections.max(mOutputs));
        } else {
            return 0;
        }
    }
  
}


class SNeuron{
    public int m_NumInputs;
    public List<Double> weights = new ArrayList();
    
    public SNeuron(int numInputs){
        m_NumInputs = numInputs + 1;
        
        for(int i = 0; i < m_NumInputs; i++){
            weights.add(Math.random());
        }
    }
}

class SNeuronLayer{
    
    public int m_NumNeurons;
    public List<Double> activations;
    public List<Double> error;
    public List<SNeuron> vecNeurons;// = new ArrayList(); 
    
    public SNeuronLayer(int numNeurons, int numInputsPerNeuron){
        vecNeurons = new ArrayList();
        m_NumNeurons = numNeurons;
        
        for(int i = 0; i < m_NumNeurons; i++){
            SNeuron neuron = new SNeuron(numInputsPerNeuron);
            vecNeurons.add(neuron);
        }
    }
}

class NeuralNet{
    private int m_numInputs;
    private int m_numOutputs;
    private int m_numHiddenLayers;
    private int m_neuronsPerHiddenLayer;
    private double responce;
    
    private List<SNeuronLayer> vecLayers = new ArrayList();
    
    private double mBias = 1.0;
    private double mResponse = 1.0;
    private double learn = 0.25;

    
    public NeuralNet(int pNumInputs, int pNumOutputs,
            int pNumHiddenLayers, int pNeuronsPerHiddenLayer){
        
        
        m_numInputs = pNumInputs;
        m_numOutputs = pNumOutputs;
        m_numHiddenLayers = pNumHiddenLayers;
        m_neuronsPerHiddenLayer = pNeuronsPerHiddenLayer;
        
        if(m_numHiddenLayers >  0){
            
            SNeuronLayer firstLayer = new SNeuronLayer(m_neuronsPerHiddenLayer, m_numInputs);
            vecLayers.add(firstLayer);
            
            for(int i = 0; i < m_neuronsPerHiddenLayer - 1; i++){
             SNeuronLayer theLayer = new SNeuronLayer(m_neuronsPerHiddenLayer, m_neuronsPerHiddenLayer); //m_numInputs
             vecLayers.add(theLayer);
            }
            
            SNeuronLayer outerLayer = new SNeuronLayer(m_numOutputs, m_neuronsPerHiddenLayer);
            vecLayers.add(outerLayer);
        }
        else{
         SNeuronLayer oLayer2 = new SNeuronLayer(m_numOutputs, m_numInputs);
         vecLayers.add(oLayer2);
        }
    }
    
    
    public List<Double> forward(List<Double> inputs){
        
        List<Double> outputs = new ArrayList();
        
        int cWeights = 0;
        
        if(inputs.size() != m_numInputs)
        {
            System.out.println("The number of outputs is: " + m_numInputs);
            System.out.println("The number of inputs is : " + inputs.size());
            System.out.println("This is incorrect.");
            return outputs;
        }
        //System.out.println(m_numInputs);
        for(int i = 0; i < m_numHiddenLayers + 1; i++){
            
            cWeights = 0;
            
            if (i > 0) {
                inputs = new ArrayList(outputs);
            }
            outputs.clear();
            
            for(int j = 0; j < vecLayers.get(i).m_NumNeurons; j++){
                
                int newNumInputs = vecLayers.get(i).vecNeurons.get(j).m_NumInputs;
                double netInput = 0.0;
                
                for(int k = 0; k < newNumInputs - 2; k++){
                    netInput += vecLayers.get(i).vecNeurons.get(j).weights.get(k) * inputs.get(cWeights++);
                }
                netInput += vecLayers.get(i).vecNeurons.get(j).weights.get(newNumInputs - 1) * mBias;
                outputs.add(Sigmoid(netInput, responce));
                
                cWeights = 0;
                
            }
            vecLayers.get(i).activations = new ArrayList(outputs);
        }
        
        return outputs;
    }
    
    void backPropagation(double index){
        double target;
        double active;
        double error;
        double weight;
        double sum = 0;
        List<Double> errors = new ArrayList();
        
        
        for (int i = 0; i < vecLayers.get(m_numHiddenLayers).m_NumNeurons; i++) {
            if (i == index) {
                target = 1.0;
            } else {
                target = 0.0;
            }
            active = vecLayers.get(m_numHiddenLayers).activations.get(i);
            error = active * (1 - active) * (active - target);
            errors.add(error);
        }
        
        vecLayers.get(m_numHiddenLayers).error = new ArrayList(errors);

        if (m_numHiddenLayers > 0) {
            for (int i = m_numHiddenLayers; i > 0; i--) {
                errors.clear();
                for (int j = 0; j < vecLayers.get(i - 1).m_NumNeurons; j++) {
                    active = vecLayers.get(i - 1).activations.get(j);
                    for (int k = 0; k < vecLayers.get(i).m_NumNeurons; k++) {
                        sum += vecLayers.get(i).error.get(k) * vecLayers.get(i).vecNeurons.get(k).weights.get(j);
                    }
                    error = active * (1 - active) * sum;
                    errors.add(error);
                }
                vecLayers.get(i - 1).error = new ArrayList(errors);
            }
        }
        
        for (int i = 0; i < m_numHiddenLayers + 1; i++) {
            for (int j = 0; j < vecLayers.get(i).m_NumNeurons; j++) {
                int numInputs = vecLayers.get(i).vecNeurons.get(j).m_NumInputs;
                for (int k = 0; k < numInputs - 1; k++) {
                    weight = vecLayers.get(i).vecNeurons.get(j).weights.get(k);
                    weight -= learn * vecLayers.get(i).error.get(j)
                            * vecLayers.get(i).activations.get(j);
                    vecLayers.get(i).vecNeurons.get(j).weights.set(k, weight);
                }
            }
        }
    }
    
    public double Sigmoid(double netInput, double Response){
        return (1.0 / (1.0 + Math.pow(Math.E, (-1.0 * netInput/ Response))));
    }
}
