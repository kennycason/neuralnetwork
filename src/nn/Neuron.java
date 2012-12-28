package nn;


import java.util.LinkedList;

/**
 * Neuron
 * @author Kenneth Cason
 * @version 1.0
 */

public class Neuron {

	private double value; 						// this contains the value of this node
	
	private double threshold;
	
	private double learningRateCoefficient; 	// learning rate * learning rate coefficient = total learning rate
	
	private LinkedList<Neuron> links; 			// the links to other nodes
												// [0] = layerID
												// [1] = position in layer
	private LinkedList<Double> weights; 		// weights 
	
	private LinkedList<Neuron> prev;			// links to the nodes that connect to this node
	
	private boolean check;

	// used for keeping up with the nodes position in the layer

	/**
	 * Constructor for objects of class MLLNode
	 * @Param double - the value being stored in the node
	 * @param _threshold
	 */
	public Neuron(double val, double thresh) {
		value = val;
		threshold = thresh;
		check = false;
		prev = new LinkedList<Neuron>();
		links = new LinkedList<Neuron>();
		weights = new LinkedList<Double>();
	}
	
	public Neuron() {
		value = 0;
		threshold = 0.65;
		check = false;
		prev = new LinkedList<Neuron>();
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
	 * @return threshold
	 */
	public double getThreshold() {
		return threshold;
	}
	
	/**
	 * 
	 * @param outputThreshold
	 */
	public void setThreshold(double val) {
		threshold = val;
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
	//	System.out.println("Orig new Weight length:" + weights.size());
		weights.add(0.0);
		//System.out.println("CONNECTING");
	//	System.out.println("Nodes new Weight length:" + weights.size());
	//	System.out.println("This "+this);
	//	System.out.println("node "+node);
	//	System.out.println("Prev "+node.getPrev());
		node.getPrev().add(this);
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
	 * getPrev - returns Linked List of previous nodes 
	 * @Param int - which element to return
	 * @Return MLLNode - the specific linked element
	 */
	public LinkedList<Neuron> getPrev() {
		return prev;
	}

	/**
	 * getPrev - returns Linked List of previous nodes 
	 * @Param int - which element to return
	 * @Return MLLNode - the specific linked element
	 */
	public Neuron getPrev(int i) {
		return prev.get(i);
	}
	
	/**
	 * getCheck - returns whether or not the node is checked
	 * @Return boolean - checked or not
	 */
	public boolean getCheck() {
		return check;
	}

	/**
	 * check - check the node
	 */
	public void check() {
		check = true;
	}

	/**
	 * unCheck - uncheck the node
	 */
	public void unCheck() {
		check = false;
	}

	
	/**
	 * deletedLinkedElement - deletes a specific linked element
	 * @Param int - which element to delete
	 * @Return double - the linked element being deleted
	 */
	public Neuron remove(int i) {
		// first remove the previous link in the connecting node, i
		links.get(i).getPrev().remove(this);
		weights.remove(i);
		return links.remove(i);
	}
	
	/**
	 * deleteAllLinks - deletes all links
	 * @Return LinkedList<MLLNode> - returns all the deleted elements
	 */
	public LinkedList<Neuron> deleteAllLinks() {
		// first remove the previous link in all connecting nodes
		for(Neuron n : links) {
			n.getPrev().remove(this);
		}
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
	public boolean isEqual(double _value) {
		// System.out.println(this.value+" "+_value);
	    if (this.value == _value) {
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
	public double compare(double _value) {
		return Math.abs(this.value - _value);
	}


}
