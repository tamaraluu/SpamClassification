import java.io.*;
import java.util.*;
// Name: Tamara Luu
// Date: 05/28/25
// CSE 123
// P3: Spam Classifier
// TA: Niyati Trivedi
// This class will be representing a classifier that creates and learns from labeled data to sort
// new data into categories. It can be created from files or from lists of data and labels.
// The classifier will be where a structured decision proccess is represented to make predictions.
public class Classifier {
    
    private ClassifierNode overallRoot;

    // Behavior: This method is a constructor for a Classifier.
    // Exceptions: This method has two exceptions. We have throw an IllegalArgumentException if
    //             input is null. On the other hand, we throw an IllegalStateExcpetion if 
    //             contents of Classifier is null after proccessing
    // Parameters: This has one scanner parameter called input, which will be representing a 
    //             classifier from a load file. The input file is formatted in pre-order traversal
    public Classifier(Scanner input){
        if(input == null){
            throw new IllegalArgumentException();
        }
        this.overallRoot = buildClassifier(input);
        if(this.overallRoot == null){
            throw new IllegalStateException();
        }
    }

    // Behavior: This method will be where it will help go through the information
    //           in a file. Information are ordered by appearance in the file.
    // Return: This method returns a ClassifierNode representing the root of the decision tree.
    //         If the line starts with "Feature: ", it creates a branch node with the extracted
    //         feature, parsed threshold, and recursively built left and right subtrees.
    //         Otherwise, it creates and returns a leaf node with the label from the line.
    //         The entire tree is built recursively by reading the input line-by-line.
    // Parameters: This has one scanner parameter called input, which will be representing a
    //             classifier from a load file. The input file is formatted in pre-order traversal
    private ClassifierNode buildClassifier(Scanner input){
        if(!input.hasNextLine()){
            return null;
        }
        String line = input.nextLine();
        if(line.startsWith("Feature: ")){ 
            String feature = line.substring("Feature: ".length()); 
            double threshold = Double.parseDouble
                (input.nextLine().substring("Threshold: ".length()));
            ClassifierNode left = buildClassifier(input);
            ClassifierNode right = buildClassifier(input);
            return new ClassifierNode(feature,threshold,left,right);
        } else {
            return new ClassifierNode(line); // if no feature
        }
    }

    // Behavior: This method is a constructor for a Classifer, which will be used to create
    //           and train a classifier from the input data and corresponding labels. Overall,
    //           creating a network.
    // Exceptions: This method throws an illegalArgumentException when data is null or when
    //             labels is null or data and labels are not the same size or data is empty.
    // Parameters: This method has two parameters:
    //             data - Takes a list of TextBlock objects to be classified
    //             labels -  list of String labels representing the correct classifications for
    //                       each data point.
    public Classifier(List<TextBlock> data, List<String> labels) {
        if(data == null || labels == null || data.size() != labels.size() || data.isEmpty()){
            throw new IllegalArgumentException();
        }
        this.overallRoot = makeClassifierHelper(0, data, labels, null);
    }

    // Behavior: This method will be used to help construct and train a new Classifer using the
    //           given input data and corresponding labels to update our network tree.
    // Return: Returns a ClassifierNode representing the root of the full classification tree
    //         built from the provided data and labels.
    // Parameters: 
    //   - index: an integer representing the current position in the data and labels lists.
    //   - data: a List of TextBlock objects representing the text data to classify.
    //   - labels: a List of String labels that correspond to each TextBlock.
    //   - root: the current root of the classification tree being built or updated.
    private ClassifierNode makeClassifierHelper(int index, List<TextBlock> data, 
        List<String> labels, ClassifierNode root){
        if(index == data.size()){
            return root;
        }
        TextBlock currentData = data.get(index);
        String expected = labels.get(index);
        if(root == null){
            root = new ClassifierNode(expected, currentData);
        } else {
            String prediction = classify(root,currentData);
            if(!prediction.equals(expected)){
                root = updateTree(root, currentData, expected);
            }
        }
        return makeClassifierHelper(index + 1, data, labels, root);
    }

    // Behavior: Updates the classification tree rooted at the given node to correctly classify
    //           the provided TextBlock and label. If the current node is a leaf, it is replaced
    //           with a new branch node that separates the original and new examples based on their
    //           most distinguishing feature. If the node is a branch, the method recursively
    //           updates the appropriate subtree.
    // Parameters:
    //   - node: the current node in the tree being examined or updated
    //   - data: the new TextBlock that was misclassified and should be incorporated
    //   - label: the correct label for the given TextBlock
    // Return: A ClassifierNode representing the updated root of the classification tree,
    //         now modified to include the new training example and correctly classify it.
    private ClassifierNode updateTree(ClassifierNode node, TextBlock data, String label) {
        if (node.isLeaf()) {
            String bestFeature = data.findBiggestDifference(node.textData);
            double threshold = midpoint(data.get(bestFeature), node.textData.get(bestFeature));

            ClassifierNode leaf1 = new ClassifierNode(label, data);

            if (data.get(bestFeature) < threshold) {
                return new ClassifierNode(bestFeature, threshold, leaf1, node);
            } else {
                return new ClassifierNode(bestFeature, threshold, node, leaf1);
            }
        } else {
            double value = data.get(node.feature);
            if (value < node.threshold) {
                node.left = updateTree(node.left, data, label);
            } else {
                node.right = updateTree(node.right, data, label);
            }
            return node;
        }
    }

    // Behavior/ Return : This method will be where we return the appropiate label that
    //                    classifier predicts.
    // Exceptions: This method throws an IllegalArgumentException if textblock input is null
    // Parameters: We have one parameter.
    //             input - A TextBlock reprsenting text data to be classified
    public String classify(TextBlock input) {
        if(input == null){
            throw new IllegalArgumentException();
        }
        return classify(overallRoot, input);
    }

    // Behavior: This method will be used to determine the appropiate classification label for
    //           given TextBlock input by traversing the classification tree starting at the 
    //           specified root node.
    // Return: This method returns a String representing the predicted label for the input.
    // Parameters: 
    //   - root: a ClassifierNode representing the current node in the classification tree.
    //   - input: a TextBlock representing the text data to be classified.
    // Exception: This method throws an IllegalArgumentException if the root is null.
    private String classify(ClassifierNode root, TextBlock input){
        if (root == null) {
            throw new IllegalArgumentException();
        }
        if (root.isLeaf()) {
            return root.label;
        }
        double value = input.get(root.feature); 
        if (value < root.threshold){
            return classify(root.left, input);
        } else{
            return classify(root.right, input);
        }
    }

    // Behavior: This method will be where we save classification network to file
    // Exception: We throw an illegalArgumentException if output is null
    // Parameters: We have one PrintStream parameter called output, which will represent
    //             the current classifier to be saved. The format will be saved in
    //             pre-order traversal
    public void save(PrintStream output) {
        if(output == null){
            throw new IllegalArgumentException();
        }
        save(overallRoot, output);
    }

    // Behavior: This method will be used to help save our current classifier network
    //           to a file
    // Parameters: This method has two parameters.
    //             root: a ClassifierNode representing the current node in the classification tree
    //             output: PrintStream parameter which will represent the current classifier to be
    //                      saved. The format will be in pre-order traversal which will be where
    //                       Every branch node will print two lines of data, one for feature 
    //                       preceded by "Feature: " and one for threshold preceded by
    //                       "Threshold: ". For leaf nodes, you should only print the label.
    private void save(ClassifierNode root, PrintStream output){
        if(root != null){
            if(root.isLeaf()){
                output.println(root.label);
            } else {
                output.println("Feature: " + root.feature);
                output.println("Threshold: " + root.threshold);
                save(root.left, output);
                save(root.right, output);
            }
        }
    }

    // This inner class will be where represent nodes of a network
    private static class ClassifierNode {
        public final String label;
        public final String feature;
        public final double threshold;
        public final TextBlock textData;
        public ClassifierNode left;
        public ClassifierNode right;

        // Behavior: This method will be a constructor for ClassifierNode
        // Parameters: We have two paramters. We have a string called label,
        //             which will represent labels associated with text data.
        //             We have a textblock called textData, which will represent
        //             the given text data we are classifying. 
        private ClassifierNode(String label, TextBlock textData){ // leaf node constructor
            this.label = label;
            this.feature = null;
            this.textData = textData;
            this.threshold = 0.0;
        }

        // Behavior: This method will be a constructor for ClassifierNode when classifying
        //           labels in our network
        // Parameters: We have one string parameter called label, which will represent labels
        //             for ClassifierNode
        private ClassifierNode(String label){
            this(label,new TextBlock(""));
        }

        // Behavior: Constructs a decision node in the classifier tree that directs classification 
        //           based on a feature and threshold.
        // Parameters: We have 4 parameters.
        //   feature: the feature name used for making decisions.
        //   threshold: the cutoff value to determine left or right subtree traversal.
        //   left: the left child node representing the subtree for values less than the threshold.
        //   right: the right child node representing the subtree for values greater than or equal
        //          to the threshold.
        private ClassifierNode(String feature, double threshold, ClassifierNode left, 
            ClassifierNode right){
            this.feature = feature;
            this.label = null;
            this.threshold = threshold;
            this.textData = new TextBlock(""); 
            this.left = left;
            this.right = right;
        }

        // Behavior: This method will be used to help determine if a node is a leaf node.
        // Return: This method will return true if left and right nodes do not have any
        //         left or right subnodes.
        private boolean isLeaf(){
            return left == null && right == null;
        }
    }

    ////////////////////////////////////////////////////////////////////
    // PROVIDED METHODS - **DO NOT MODIFY ANYTHING BELOW THIS LINE!** //
    ////////////////////////////////////////////////////////////////////

    private static double midpoint(double one, double two) {
        return Math.min(one, two) + (Math.abs(one - two) / 2.0);
    }

    public Map<String, Double> calculateAccuracy(List<TextBlock> data, List<String> labels) {
        if (data.size() != labels.size()) {
            throw new IllegalArgumentException(
                    String.format("Length of provided data [%d] doesn't match provided labels [%d]",
                                  data.size(), labels.size()));
        }

        Map<String, Integer> labelToTotal = new HashMap<>();
        Map<String, Double> labelToCorrect = new HashMap<>();
        labelToTotal.put("Overall", 0);
        labelToCorrect.put("Overall", 0.0);

        for (int i = 0; i < data.size(); i++) {
            String result = classify(data.get(i));
            String label = labels.get(i);

            labelToTotal.put(label, labelToTotal.getOrDefault(label, 0) + 1);
            labelToTotal.put("Overall", labelToTotal.get("Overall") + 1);
            if (result.equals(label)) {
                labelToCorrect.put(result, labelToCorrect.getOrDefault(result, 0.0) + 1);
                labelToCorrect.put("Overall", labelToCorrect.get("Overall") + 1);
            }
        }

        for (String label : labelToCorrect.keySet()) {
            labelToCorrect.put(label, labelToCorrect.get(label) / labelToTotal.get(label));
        }
        return labelToCorrect;
    }
}