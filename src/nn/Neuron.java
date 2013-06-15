package nn;


import java.util.LinkedList;

/**
 * Neuron
 * @author Kenneth Cason
 * @version 1.0
 */

public class Neuron {

	private double value; // this contains the value of this node
	
	private double learningRateCoefficient; // learning rate * learning rate coefficient = total learning rate
	
	private LinkedList<Neuron> links; // the links to other nodes

	private LinkedList<Double> weights; // weights 

	public Neuron() {
		value = 0;
		links = new LinkedList<Neuron>();
		weights = new LinkedList<Double>();
	}

	/**
	 * getValue - returns the value
	 * @Return double - the value
	 */
	public double getValue() {
		return value;
	}

	/**
	 * setValue - sets the value of this node
	 * @Param double - the value
	 */
	public void setValue(double val) {
		value = val;
	}
	
	/**
	 * 
	 * @param rate
	 */
	public void setLearningRateCoefficient(double rate) {
		learningRateCoefficient = rate;
	}

	/**
	 * 
	 * @return
	 */
	public double getLearningRateCoefficient() {
		return learningRateCoefficient;
	}
	
	/**
	 * setWeight - sets a specific weight
	 * @Param int - the specific array
	 * @Param double the value of the weights
	 */
	public void setWeight(int i, double weight) {
		weights.set(i, weight);
	}
	
	/**
	 * setWeights - set bias weights
	 * @Param LinkedList<Double> weights
	 */
	public void setWeights(LinkedList<Double> _weights) {
		weights = _weights;
	}
	
	/**
	 * getWeight - returns a specific weight
	 * @Param int - the weight to return
	 * @Return double - the nodes weight
	 */
	public double getWeight(int i) {
		return weights.get(i);
	}
	
	/**
	 * getWeights - returns all the links weights
	 * @Return LinkedList<MLLNode> - the integer LinkedList<MLLNode> of weights
	 */
	public LinkedList<Double> getWeights() {
		return weights;
	}

	/**
	 * connectNode - connect to another node
	 * @Param MLLNode -  node 
	 */
	public void connectNode(Neuron node) {
		links.add(node);
		weights.add(0.0);
	}

	/**
	 * getAllLinked - returns all the linked elements
	 * @Return LinkedList<MLLNode> - the linked elements
	 */
	public LinkedList<Neuron> getAllLinked() {
		return links;
	}

	/**
	 * get - returns a specific linked element
	 * @Param int - which element to return
	 * @Return MLLNode - the specific linked element
	 */
	public Neuron get(int i) {
		return links.get(i);
	}
	
	/**
	 * deletedLinkedElement - deletes a specific linked element
	 * @Param int - which element to delete
	 * @Return double - the linked element being deleted
	 */
	public Neuron remove(int i) {
		weights.remove(i);
		return links.remove(i);
	}
	
	/**
	 * deleteAllLinks - deletes all links
	 * @Return LinkedList<MLLNode> - returns all the deleted elements
	 */
	public LinkedList<Neuron> deleteAllLinks() {
		LinkedList<Neuron> temp = links;
		links.clear();
		weights.clear();
		return temp;
	}
	
	/**
	 * isEqual - compare 2 pieces of value
	 * @Para double - the value to compare
	 * @Return boolean - returns true if equal, else false
	 */
	public boolean isEqual(double value) {
	    if (this.value == value) {
		    return true;
		}
		return false;
	}
	
	/**
	 * compare - compare 2 pieces of value
	 * @Para double - the value to compare
	 * @Return double - returns the absolute value of the difference
	 * between values
	 */
	public double compare(double value) {
		return Math.abs(this.value - value);
	}

}
