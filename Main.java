import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;

public class Main {

    // DATA //
    static String TrainingFilename = "mnist_train.csv";
    static String TestingFilename = "mnist_test.csv";
    static int inputs = 784;
    static int outputs = 10;
    static int hiddenlayer = 70;
    static double eta = 3; // Learning Rate
    static int epochs = 150;
    static double MiniBatchSize = 10.0;
    //Contains all inputs and output data, index 0 is inputs, 1 is expected output
    static List<double[][]> Data = new ArrayList<double[][]>();
    static List<List<double[][]>> TrainingData = new ArrayList<List<double[][]>>();

    // BIASES //
    static double[] L1Biases = new double[hiddenlayer];
    static double[] L2Biases = new double[outputs];
    static double[][] NetworkBiases = { L1Biases, L2Biases };

    // WEIGHTS //
    static double[][] L1Weights = new double[hiddenlayer][inputs];
    static double[][] L1WeightsGradient = new double[hiddenlayer][inputs];

    static double[][] L2Weights = new double[outputs][hiddenlayer];
    static double[][] L2WeightsGradient = new double[outputs][hiddenlayer];
    static double[][][] NetworkWeights = { L1Weights, L2Weights };

    // STUFF //
    static double[][] Network_A_Values = { L1Biases.clone(), L2Biases.clone() };
    static double[][] BiasGradient = { new double[L1Biases.length], new double[L2Biases.length] };
    static double[][] SummedBiasGradient = { new double[L1Biases.length], new double[L2Biases.length] };
    static double[][][] WeightGradient = { L1WeightsGradient, L2WeightsGradient };
    static List<int[]> IncorrectAnswers = new ArrayList<int[]>();
    static List<int[]> CorrectAnswers = new ArrayList<int[]>();
    static List<int[]> Answers = new ArrayList<int[]>();


    public static void main(String[] args) {
        
        boolean TrainedNetwork = false;
        while (true) {
            System.out.println("Type the digit of the option you want:\n" +
                            "[1] Train the network\n" + 
                            "[2] Load a pre-trained network\n" +
                            "[3] Display network accuracy on training data\n" +
                            "[4] Display network accuracy on testing data\n" +
                            "[5] Run network on testing data showing images and labels\n" + 
                            "[6] Display the misclassified testing images\n" +
                            "[7] Save the network state to a file\n" +
                            "[0] Exit");
            String answer = System.console().readLine();

            // Exit
            if (answer.equals("0")) {
                System.exit(0);

            // Train the network
            } else if (answer.equals("1")) {

                LoadData(true);
                if (!TrainedNetwork){
                    RandomizeNetwork();
                }
                train();
                TrainedNetwork = true;

            // Load pre-trained network
            } else if (answer.equals("2")) {

                LoadTrainedNetwork();
                TrainedNetwork = true;

            // Display network accuracy on training data
            } else if (answer.equals("3")) {

                if (!TrainedNetwork) {System.out.println("You must do option 1 or 2 first");}
                else {
                    LoadData(true);
                    test(true, 0, 0);
                }

            // Display network accuracy on testing data
            } else if (answer.equals("4")) {

                if (!TrainedNetwork) {System.out.println("You must do option 1 or 2 first.");}
                else {
                    LoadData(false);
                    test(true, 0, 0);
                }

            // Run network on testing data showing images and labels 
            } else if (answer.equals("5")) {

                if (!TrainedNetwork) {System.out.println("You must do option 1 or 2 first.");}
                else {
                    LoadData(false);
                    for (int i = 0; i < Data.size(); i++) {
                        test(false, i, i + 1);
                        Display(Answers.getLast());
                        // Wait for user input
                        System.out.println("\nEnter 1 to continue or anything else to return to the menu.");
                        if (!System.console().readLine().equals("1")){
                            break;
                        }
                    }
                }

            // Display the misclassified testing images
            } else if (answer.equals("6")) {

                if (!TrainedNetwork) {System.out.println("You must do option 1 or 2 first.");}
                else {
                    LoadData(false);
                    test(false, 0, 0);
                    //Print each incorrect answer
                    for (int i = 0; i < IncorrectAnswers.size(); i++) {
                        Display(IncorrectAnswers.get(i));
                        // Wait for user input
                        System.out.println("\nEnter 1 to continue or anything else to return to the menu.");
                        if (!System.console().readLine().equals("1")){
                            break;
                        }
                    }
                }

            // Save the network state to a file
            } else if (answer.equals("7")) {
                
                if (!TrainedNetwork) {System.out.println("You must do option 1 or 2 first.");}
                else {
                    SaveNetwork();
                }

            // Bad option
            } else {
                System.out.println("Not a valid option");
            }
        }
    }

    public static void train(){
        //Reset Statistics
        CorrectAnswers.clear();
        IncorrectAnswers.clear();
        Answers.clear();

        for (int epoch = 1; epoch <= epochs; epoch++) 
        {
            System.out.println(String.format("------ Epoch %d ------", epoch));
            // For each mini-batch
            for (int b = 0; b < TrainingData.size(); b++) 
            {
                //Zero out Weight and Bias gradients
                for (int level = 0; level < WeightGradient.length; level++) {
                    for (int node = 0; node < WeightGradient[level].length; node++) {
                        SummedBiasGradient[level][node] = 0;
                        for (int weight = 0; weight < WeightGradient[level][node].length; weight++) {
                            WeightGradient[level][node][weight] = 0;
                        } 
                    }
                }


                var Batch = TrainingData.get(b);
                // Go through each data-set in Mini-Batch
                for (int d = 0; d < Batch.size(); d++) 
                {
                    var Dataset = Batch.get(d);
                    //Propogate the data through the network
                    propagate(Dataset, b * (int)MiniBatchSize + d);

                    //Back Propagate
                    backPropagate(Dataset);
                }

                //System.out.print(String.format("-- End Mini-Batch %d --\n\nRevised Weights and Biases:", b+1));

                //Update Network Weights and biases
                for (int level = 0; level < NetworkWeights.length; level++) {
                    var Weights = NetworkWeights[level];
                    var Biases = NetworkBiases[level];

                    //System.out.println(String.format("\nWeights: Layer %d",level+1));
                    // For each node update weights
                    for (int node = 0; node < Weights.length; node++) {
                        // Update Weights for each connection for each connection
                        var Node = Weights[node];
                        for (int i = 0; i < Node.length; i++) {
                            Node[i] = Node[i] - (eta/MiniBatchSize) * WeightGradient[level][node][i];
                            //System.out.print(String.format("%f  ", Node[i]));
                        }
                        //System.out.println();
                        
                    }

                    // For each node update biases
                    //System.out.println(String.format("\nBiases: Layer %d",level+1));
                    for (int node = 0; node < Weights.length; node++){
                        Biases[node] = Biases[node] - (eta/MiniBatchSize) * SummedBiasGradient[level][node];
                        //System.out.print(String.format("%f  ", Biases[node]));
                    }
                    //System.out.println("\n");
                }
            }

            PrintStatistics();
        }
    }

    public static void test(boolean PrintStats, int StartIndex, int EndIndex){
        //Reset Statistics
        CorrectAnswers.clear();
        IncorrectAnswers.clear();
        Answers.clear();

        if (EndIndex == 0){
            EndIndex = Data.size();
        }

        for (int i = StartIndex; i < EndIndex; i++) {
            propagate(Data.get(i), i);
        }
        if (PrintStats){
            PrintStatistics();
        }
    }

    public static void propagate(double[][] Data, int index) {
        double[] X_Inputs = Data[0];
        double[] Y_Outputs = Data[1];
        for (int level = 0; level < NetworkWeights.length; level++) {
            var LevelWeights = NetworkWeights[level];
            var LevelBiases = NetworkBiases[level];
            int NodeCount = LevelWeights.length;
            double[] LVL_Z_Values = new double[NodeCount];
            double[] LVL_A_Values = new double[NodeCount];

            for (int i = 0; i < NodeCount; i++) {
                for (int j = 0; j < X_Inputs.length; j++) {
                    LVL_Z_Values[i] += X_Inputs[j] * LevelWeights[i][j];
                }
                LVL_Z_Values[i] += LevelBiases[i];
                LVL_A_Values[i] = sigmoid(LVL_Z_Values[i]);
            }

            //Update inputs to next level's outputs
            X_Inputs = LVL_A_Values;
            Network_A_Values[level] = LVL_A_Values.clone();
        }

        // Determine if network got the correct answer
        double[] OutputLayerA = Network_A_Values[NetworkBiases.length - 1];
        // Get max output
        double NetworkMax = 0;
        double OutputMax = 0;
        int NetworkMaxi = 0;
        int OutputMaxi = 0;

        for (int i = 0; i < OutputLayerA.length; i++) {
            if (OutputLayerA[i] > NetworkMax){
                NetworkMax = OutputLayerA[i];
                NetworkMaxi = i;
            }
            if (Y_Outputs[i] > OutputMax){
                OutputMax = Y_Outputs[i];
                OutputMaxi = i;
            }
        }

        //Network Answer, Correct Answer, Data index
        int[] info = {NetworkMaxi, OutputMaxi, index};
        //Correct Answer
        if (OutputMaxi == NetworkMaxi){
            CorrectAnswers.add(info);
        }
        //Incorrect Answer
        else{
            IncorrectAnswers.add(info);
        }
        Answers.add(info);
    }

    public static void backPropagate(double[][] Data){
        // Iterate through each level starting at the last
        double[][] A_values = {Data[0], Network_A_Values[0], Network_A_Values[1]};
        double[] Y_Outputs = Data[1];
        for (int level = NetworkWeights.length - 1; level >= 0; level--) {
            // For each node in the level, Compute the bias and weight gradient.
            //System.out.println(String.format("Level %d Bias Gradient:", level+1));
            // For each node
            for (int j = 0; j < Network_A_Values[level].length; j++) {

                // For every connection to node, compute the bias gradient
                // A Value of current node
                double A_Node = Network_A_Values[level][j];
                double sum = 0;
                // If node is an output node, calculate bias based on what outputs should be.
                if (level == Network_A_Values.length - 1) {
                    // Not really a sum just used for same variable to use one equation
                    sum = (A_Node - Y_Outputs[j]);
                }   
                // Not an output node
                else {
                    double[][] weights = NetworkWeights[level + 1];
                    // For each weight connecting current node to the n ext layer.
                    for (int k = 0; k < weights.length; k++) {
                        sum += weights[k][j] * BiasGradient[level + 1][k];
                    }
                }
                // Compute Bias Gradient
                BiasGradient[level][j] = sum * A_Node * (1 - A_Node);
                SummedBiasGradient[level][j] += sum * A_Node * (1 - A_Node);

                //Compute weight gradient
                for (int i = 0; i < NetworkWeights[level][j].length; i++) {
                    WeightGradient[level][j][i] += A_values[level][i] * BiasGradient[level][j];
                }

            }
            //System.out.println(String.format("\n\nLevel %d Weight Gradient:", level+1));
            // Print Weight Gradient
            for (int i = 0; i < NetworkWeights[level].length; i++) {
                for (int j = 0; j < NetworkWeights[level][i].length; j++) {
                    //System.out.print(String.format("%f\t", WeightGradient[level][i][j]));
                }
                //System.out.println();
            }
            //System.out.println();
        }
    }

    public static void RandomizeNetwork(){
        //Randomize weights
        for (var LevelWeights : NetworkWeights) {
            for (double[] NodeWeights : LevelWeights) {
                for (int i = 0; i < NodeWeights.length; i++) {
                    NodeWeights[i] = (Math.random() - 0.5) * Math.random();
                }
            }
        }

        //Randomize Biases
        for (double[] LevelBiases : NetworkBiases) {
            for (int i = 0; i < LevelBiases.length; i++) {
                LevelBiases[i] = (Math.random() - 0.5) * Math.random();
            }
        }
        System.out.println("- Network has been randomized.");
    }
    
    //Reads data from file, scales inputs, stores in global variable Data, and creates mini-batches
    public static void LoadData(boolean training){

        //Don't load data again when it's already loaded
        if (Data.size() == 60000 && training){
            return;
        }
        else if (Data.size() == 10000 && !training){
            return;
        }

        Data.clear();
        TrainingData.clear();
        String filename = training ? TrainingFilename : TestingFilename;
        FileInputStream Istream;
        String line;
        try {
            Istream = new FileInputStream(filename);
        } catch (FileNotFoundException e) {
            System.out.println(String.format("Training data file could not be found please place %s in the same directory", filename));
            return;
        }
        BufferedReader br = new BufferedReader(new InputStreamReader(Istream));
        try {
            line = br.readLine();
        } catch (IOException e) {
            System.out.println(e.getMessage());
            line = null;
        }

        //Populate the list with each line that has been split into an array for each pixel.
        while (line != null){
            String[] elements = line.split(",");
            double[] ScaledInput = new double[elements.length - 1];
            double[] ExpectedOutput = new double[10];

            //Convert all inputs to doubles and scale them between 0.0 and 1.0
            for (int i = 0; i < ScaledInput.length; i++) {
                ScaledInput[i] = Double.parseDouble(elements[i + 1]) / 255.0;
            }
            //Create correct output layer array with all zeros except a 1 for the index with the correct answer.
            double CorrectAnswer = Double.parseDouble(elements[0]);
            ExpectedOutput[((int)CorrectAnswer)] = 1;
            double[][] DataElement = {ScaledInput, ExpectedOutput};
            Data.add(DataElement);

            //Get next line
            try {
                line = br.readLine();
            } catch (IOException e) {
                System.out.println(e.getMessage());
                line = null;
            }
        }

        //Close the Buffered Reader
        try {
            br.close();
        } catch (IOException e) { }

        //Create the training data which is a list of mini-batches
        if (training){
            int i = 0;
            int Datasize = Data.size();

            //Prioritize mini-batch size
            while (i <= Datasize - (int)MiniBatchSize){
                TrainingData.add(Data.subList(i, i + (int)MiniBatchSize));
                i += (int)MiniBatchSize;
            }
            System.out.println(String.format("- Data loaded from file. %d Data sets grouped into %d mini-batches.", Datasize, TrainingData.size()));
        }
        else{
            System.out.println(String.format(" - %d Data-sets loaded from file.", Data.size()));
        }
    }

    //Display a dataset on the command line as a 28x28 ascii representation
    public static void Display(int[] NetworkAnswer){
        double[][] data = Data.get(NetworkAnswer[2]);
        double[] image = data[0];
        String correct = NetworkAnswer[0] == NetworkAnswer[1] ? "Correct." : "Incorrect.";
        System.out.println(String.format("\nTest case #%d\t%s\tNetwork Answer: %d\tCorrect Answer: %d",NetworkAnswer[2], correct, NetworkAnswer[0], NetworkAnswer[1]));
        for (int i = 0; i < image.length; i++) {
            if (i % 28 == 0){
                System.out.println();
            }
            System.out.print(AsciiMap(image[i]));
        }
    }

    //Given a value between 0 and 1 return a ascii character to represent that intensity.
    public static String AsciiMap(double value){
        if (value > 0.9){
            return "&";
        }
        else if (value > 0.7){
            return "W";
        }
        else if (value > 0.5){
            return "c";
        }
        else if (value > 0.3){
            return "-";
        }
        else if (value > 0){
            return ".";
        }
        else{
            return " ";
        }
    }

    //Load network weights and biases to a file
    public static void LoadTrainedNetwork() {
        FileInputStream Istream;
        try {
            Istream = new FileInputStream("Network.txt");
        } catch (FileNotFoundException e) {
            System.out.println("\nSaved network file could not be found!\nTitle your network Network.txt and try again.");
            return;
        }
        BufferedReader br = new BufferedReader(new InputStreamReader(Istream));
        try {
            String line = br.readLine();
            int node = 0;
            int level = 0;

            while (line != null){
                if (line.equals("- NEXT LEVEL -")){
                    level += 1;
                    node = 0;
                    line = br.readLine();
                    continue;
                }

                String[] elements = line.split(",");
                NetworkBiases[level][node] = Double.parseDouble(elements[0]);
                for (int i = 0; i < elements.length - 1; i++) {
                    NetworkWeights[level][node][i] = Double.parseDouble(elements[i + 1]);
                }
                line = br.readLine();
                node += 1;
            }
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }
        //Close the Buffered Reader
        try {
            br.close();
        } catch (IOException e) { }
        System.out.println(" - Network Loaded.");
    }

    //Save network weights and biases to a file
    public static void SaveNetwork(){
        Writer writer = null;

        try {
            writer = new BufferedWriter(new OutputStreamWriter(
            new FileOutputStream("Network.txt"), "utf-8"));

            //Write the number of nodes in hidden layer to file

            //Write Biases and weights to the file
            for (int level = 0; level < NetworkBiases.length; level++) {
                double[] biases = NetworkBiases[level];
                for (int node = 0; node < biases.length; node++) {
                    writer.write(String.valueOf(biases[node]));
                    //Write all the weights for node
                    for (int i = 0; i < NetworkWeights[level][node].length; i++) {
                        writer.write(",");
                        writer.write(String.valueOf(NetworkWeights[level][node][i]));
                    }
                    writer.write("\n");
                }
                if (level != NetworkBiases.length - 1){
                    writer.write("- NEXT LEVEL -\n");
                }
            }
        } 
        catch (IOException ex) {
        // Report
        } 
        finally {
            try {writer.close();} 
            catch (Exception ex) { }
        }
        System.out.println(" - Network has been saved");
    }

    public static double sigmoid(double z) {
        return 1.0 / (1 + Math.exp(-z));
    }

    public static int GetMax(int[] items){
        return -1;
    }

    public static void PrintStatistics(){
        double accuracy = (double)CorrectAnswers.size() / Data.size();
        //Stats for each digit, 0th index is incorrect 1st is correct
        int[] numCorrect = new int[10];
        //Total number of each digit
        int[] numCount;

        //Get number of corrects for each digit
        for (int j = 0; j < CorrectAnswers.size(); j++) {
            int[] answer = CorrectAnswers.get(j);
            numCorrect[answer[1]] += 1;
        }
        //Add up all corrects
        numCount = numCorrect.clone();
        //Add up all incorrects
        for (int i = 0; i < IncorrectAnswers.size(); i++) {
            int[] answer = IncorrectAnswers.get(i);
            numCount[answer[1]] += 1;
        }

        //print stats
        for (int i = 0; i < numCorrect.length; i++) {
            System.out.print(String.format("%d = %d/%d (%.2f%%)\t", i, numCorrect[i], numCount[i], ((double)numCorrect[i] / numCount[i]) * 100));
            if (i == 5) {
                System.out.println();
            }
        }
        System.out.println(String.format("Accuracy: %.3f%% (%d/%d)\n", accuracy * 100, CorrectAnswers.size(), Data.size()));
        //Reset Statistics
        CorrectAnswers.clear();
        IncorrectAnswers.clear();
        Answers.clear();
    }
}