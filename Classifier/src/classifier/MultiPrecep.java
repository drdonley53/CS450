/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package classifier;
import static java.lang.Math.exp;
import java.util.ArrayList;
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
        mNN = new NeuralNet(data.numAttributes(), data.numClasses(),
            1, 4);        
    }
    
    @Override
    public double classifyInstance(Instance instance) {
        List<Double> attributeValues = new ArrayList();
        //System.out.println(instance.numAttributes());
        for (int i = 0; i < instance.numAttributes(); i++) {
            attributeValues.add(instance.value(i));
        } 
        mOutputs = mNN.update(attributeValues);
        //System.out.println(attributeValues.size());
        return mOutputs.get(0);
    }
  
}


class SNeuron{
    public int m_NumInputs;
    public ArrayList<Double> weights = new ArrayList<Double>();
    
    public SNeuron(int numInputs){
        m_NumInputs = numInputs + 1;
        
        for(int i = 0; i < m_NumInputs + 1; i++){
            weights.add(Math.random());
        }
    }
}

class SNeuronLayer{
    
    public int m_NumNeurons;
    public ArrayList<SNeuron> vecNeurons = new ArrayList<SNeuron>(); 
    
    public SNeuronLayer(int numNeurons, int numInputsPerNeuron){
        
        m_NumNeurons = numNeurons;
        
        SNeuron neuron = new SNeuron(numInputsPerNeuron);
        
        for(int i = 0; i < m_NumNeurons; i++){
            vecNeurons.add(neuron);
        }
    }
    
    public ArrayList<SNeuron> getNeurons() {
        return vecNeurons;
    }
}

class NeuralNet{
    private int m_numInputs;
    private int m_numOutputs;
    private int m_numHiddenLayers;
    private int m_neuronsPerHiddenLayer;
    private double responce;
    
    private ArrayList<SNeuronLayer> vecLayers = new ArrayList<SNeuronLayer>();
    
    private double mBias = 1.0;
    private double mResponse = 1.0;

    
    public NeuralNet(int pNumInputs, int pNumOutputs,
            int pNumHiddenLayers, int pNeuronsPerHiddenLayer){
        
        
        m_numInputs = pNumInputs;
        m_numOutputs = pNumOutputs;
        m_numHiddenLayers = pNumHiddenLayers;
        m_neuronsPerHiddenLayer = pNeuronsPerHiddenLayer;
        
        if(m_numHiddenLayers >  0){
            
            SNeuronLayer fLayer = new SNeuronLayer(m_neuronsPerHiddenLayer, m_numInputs);
            vecLayers.add(fLayer);
            
            for(int i = 0; i < m_neuronsPerHiddenLayer - 1; i++){
             SNeuronLayer theLayer = new SNeuronLayer(m_neuronsPerHiddenLayer, m_numInputs); //m_neurosPerHiddenLayer 
             vecLayers.add(theLayer);
            }
            SNeuronLayer oLayer = new SNeuronLayer(m_numOutputs, m_neuronsPerHiddenLayer);
            vecLayers.add(oLayer);
        }
        else{
         SNeuronLayer oLayer2 = new SNeuronLayer(m_numOutputs, m_numInputs);
         vecLayers.add(oLayer2);
        }
    }
    
    public ArrayList<Double> GetWeights(){
        
        ArrayList<Double> weights = new ArrayList<Double>();
        
        for(int i = 0; i < m_numHiddenLayers + 1; i++){
            for(int j = 0; j < vecLayers.get(i).m_NumNeurons; j++){
                for(int k = 0; k < vecLayers.get(i).vecNeurons.get(j).m_NumInputs; k++){
                    weights.add(vecLayers.get(i).vecNeurons.get(j).weights.get(k));
                }
            }
        }
        return weights;
    }
    
    public int getNumWeights(){
        int weights = 0;
        
        for(int i = 0; i < m_numHiddenLayers + 1; i++){
            for(int j = 0; j < vecLayers.get(i).m_NumNeurons; j++){
                for(int k = 0; k < vecLayers.get(i).vecNeurons.get(j).m_NumInputs; k++){
                    weights++;
                }
            }
        }
        return weights;
    }
    
    public void PutWeights(ArrayList<Double> weights){
        
        int cweight = 0;
        
        for(int i = 0; i < m_numHiddenLayers + 1; i++){
            for(int j = 0; j < vecLayers.get(i).m_NumNeurons; j++){
                for(int k = 0; k < vecLayers.get(i).vecNeurons.get(j).m_NumInputs; k++){
                    vecLayers.get(i).vecNeurons.get(j).weights.set(k, weights.get(cweight++));
                }
            }
        }
        
    }
    
    
    public List<Double> update(List<Double> inputs){
        
        List<Double> outputs = new ArrayList<Double>();
        
        int cWeights = 0;
        
        if(inputs.size() != m_numInputs)
        {
            return outputs;
        }
        
        for(int i = 0; i < m_numHiddenLayers + 1; i++){
            /*if(i > 0)
            {
               inputs = outputs; 
            }
            outputs.clear();*/
            cWeights = 0;
            
            for(int j = 0; j < vecLayers.get(i).m_NumNeurons; j++){
                
                int newNumInputs = vecLayers.get(i).vecNeurons.get(j).m_NumInputs;
                double netInput = 0.0;
                
                for(int k = 0; k < newNumInputs - 2; k++){
                    netInput += vecLayers.get(i).vecNeurons.get(j).weights.get(k) * inputs.get(cWeights++);
                    outputs.add(Sigoid(netInput, responce));
                }
                
                netInput += vecLayers.get(i).vecNeurons.get(j).weights.get(newNumInputs - 1) * mBias;
                
                if (netInput > 0)
                    outputs.add(1.0);
                else
                    outputs.add(0.0);
                            

                cWeights = 0;
                
            }
        }
        return outputs;
    }
    
    public double Sigoid(double netInput, double Response){
        return (1.0 / (1.0 + Math.pow(Math.E, (-1.0 * netInput / Response))));
    }
}
