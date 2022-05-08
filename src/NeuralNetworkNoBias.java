import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class NeuralNetworkNoBias {
    private double[][] hidden_layer_weights;
    private double[][] output_layer_weights;
    private final int num_inputs;
    private final int num_hidden;
    private final int num_outputs;
    private final double learning_rate;

    public NeuralNetworkNoBias(int num_inputs, int num_hidden, int num_outputs, double[][] initial_hidden_layer_weights, double[][] initial_output_layer_weights, double learning_rate) {
        //Initialise the network
        this.num_inputs = num_inputs;
        this.num_hidden = num_hidden;
        this.num_outputs = num_outputs;

        this.hidden_layer_weights = initial_hidden_layer_weights;
        this.output_layer_weights = initial_output_layer_weights;

        this.learning_rate = learning_rate;
    }


    //Calculate neuron activation for an input

    public double sigmoid(double input) {

        return 1/(1 + Math.exp(-input));

    }
    //Feed forward pass input to a network output
    public double[][] forward_pass(double[] inputs) {

        // Calculate the output values for all the hidden nodes.
        double[] hidden_layer_outputs = new double[num_hidden];

        for (int i = 0; i < num_hidden; i++) {

            double weighted_sum = 0;

            for(int j=0; j<num_inputs; j++) {
                weighted_sum = weighted_sum + inputs[j] * hidden_layer_weights[j][i];
            }

            double output = sigmoid(weighted_sum);
            hidden_layer_outputs[i] = output;

        }

        // Calculate the output values for all the output nodes.
        double[] output_layer_outputs = new double[num_outputs];

        for (int i = 0; i < num_outputs; i++) {

            double weighted_sum = 0;

            for(int j=0; j<num_hidden; j++) {
                weighted_sum = weighted_sum + hidden_layer_outputs[j] * output_layer_weights[j][i];
            }

            double output = sigmoid(weighted_sum);
            output_layer_outputs[i] = output;

        }

        return new double[][]{hidden_layer_outputs, output_layer_outputs};
    }

    public double[][][] backward_propagate_error(double[] inputs, double[] hidden_layer_outputs,
                                                 double[] output_layer_outputs, int desired_outputs) {

        double[] output_layer_betas = new double[num_outputs];

        // TODO! Calculate output layer betas.
        for(int i=0; i<num_outputs; i++) {

            if(i == desired_outputs) {
                double beta = 1 - output_layer_outputs[i];
                output_layer_betas[i] = beta;
            }

            else {
                double beta = 0 - output_layer_outputs[i];
                output_layer_betas[i] = beta;
            }

        }

        //System.out.println("OL betas: " + Arrays.toString(output_layer_betas));

        double[] hidden_layer_betas = new double[num_hidden];

        // TODO! Calculate hidden layer betas.

        // iterating through all the hidden node.
        for(int i=0; i<num_hidden; i++) {

            double[] weights = output_layer_weights[i];
            double beta = 0.0;

            // iterating through each output node.
            for(int j=0; j<num_outputs; j++) {

                double Wij = weights[j];

                double Oj  = output_layer_outputs[j];

                double Betaj = output_layer_betas[j];

                beta = beta + Wij * Oj * (1-Oj) * Betaj;

            }

            hidden_layer_betas[i] = beta;

        }

        //System.out.println("HL betas: " + Arrays.toString(hidden_layer_betas));

        // This is a HxO array (H hidden nodes, O outputs)
        double[][] delta_output_layer_weights = new double[num_hidden][num_outputs];

        // TODO! Calculate output layer weight changes.
        for(int i=0; i<num_hidden; i++) {

            for(int j=0; j<num_outputs; j++) {

                double hidden_output = hidden_layer_outputs[i];

                double output_output = output_layer_outputs[j];

                double beta_output = output_layer_betas[j];

                double change = learning_rate * hidden_output * output_output * (1-output_output) * beta_output;

                delta_output_layer_weights[i][j] = change;

            }

        }

        // This is a IxH array (I inputs, H hidden nodes)
        double[][] delta_hidden_layer_weights = new double[num_inputs][num_hidden];

        // TODO! Calculate hidden layer weight changes.
        for(int i=0; i<num_inputs; i++) {

            for(int j=0; j<num_hidden; j++) {

                double input = inputs[i];

                double hidden_output = hidden_layer_outputs[j];

                double beta_hidden = hidden_layer_betas[j];

                double change = learning_rate * input * hidden_output * (1-hidden_output) * beta_hidden;

                delta_hidden_layer_weights[i][j] = change;

            }
        }

        //TODO! Calculate the hidden_layer_bias changes.
        double[] delta_hidden_layer_bias = new double[num_hidden];

        for(int i=0; i<num_hidden; i++) {
            double Oi = hidden_layer_outputs[i];

            double beta_i = hidden_layer_betas[i];

            double change = learning_rate * Oi * (1-Oi) * beta_i;

            delta_hidden_layer_bias[i] = change;
        }

        //TODO! Calculate the output_layer_bias changes.
        double[] delta_output_layer_bias = new double[num_outputs];

        for(int i=0; i<num_outputs; i++) {
            double Oi = output_layer_outputs[i];

            double beta_i = output_layer_betas[i];

            double change = learning_rate * Oi * (1-Oi) * beta_i;

            delta_output_layer_bias[i] = change;
        }

        // Return the weights we calculated, so they can be used to update all the weights.
        return new double[][][]{delta_output_layer_weights, delta_hidden_layer_weights};
    }

    public void update_weights(double[][] delta_output_layer_weights, double[][] delta_hidden_layer_weights) {
        // TODO! Update the weights

        int rows = hidden_layer_weights.length;
        int cols = hidden_layer_weights[0].length;

        for(int row=0; row<rows; row++) {

            for(int col=0; col<cols; col++) {
                hidden_layer_weights[row][col] = hidden_layer_weights[row][col] + delta_hidden_layer_weights[row][col];
            }

        }

        int rows2 = output_layer_weights.length;
        int cols2 = output_layer_weights[0].length;

        for(int row=0; row<rows2; row++) {

            for(int col=0; col<cols2; col++) {
                output_layer_weights[row][col] = output_layer_weights[row][col] + delta_output_layer_weights[row][col];
            }

        }

    }


    public void train(double[][] instances, int[] desired_outputs, int epochs) {

        // train the model to get new set of weights.
        for (int epoch = 0; epoch < epochs; epoch++) {

            System.out.println("epoch = " + epoch);
//          int[] predictions = new int[instances.length];

            for (int i = 0; i < instances.length; i++) {
                double[] instance = instances[i];
                double[][] outputs = forward_pass(instance);
                double[][][] delta_weights = backward_propagate_error(instance, outputs[0], outputs[1], desired_outputs[i]);

//                int predicted_class = -1; // TODO!
//                predictions[i] = predicted_class;

                //We use online learning, i.e. update the weights after every instance.
                update_weights(delta_weights[0], delta_weights[1]);
            }

//          Print new weights
            System.out.println("Hidden layer weights \n" + Arrays.deepToString(hidden_layer_weights));
            System.out.println("Output layer weights  \n" + Arrays.deepToString(output_layer_weights));

            int[] predictions = predict(instances, false);

            // TODO: Print accuracy achieved over this epoch
            int correct = 0;

            for(int i=0; i<desired_outputs.length; i++) {
                if(desired_outputs[i] == predictions[i]) {
                    correct++;
                }
            }

            double acc = (double) correct / (double)desired_outputs.length;

            System.out.println("Accuracy = " + acc + "\n");
        }
    }

    public int[] predict(double[][] instances, boolean report) {

        int[] predictions = new int[instances.length];

        for (int i = 0; i < instances.length; i++) {
            double[] instance = instances[i];
            double[][] outputs = forward_pass(instance);

            double[] output_layer_outputs = outputs[1];

            int predicted_class = 0;
            double highest = -1.0;


            // getting the index of the highest output node value.
            for(int j=0; j<output_layer_outputs.length; j++) {

                if(output_layer_outputs[j] > highest) {
                    highest = output_layer_outputs[j];
                    predicted_class = j;
                }
            }

            if(report){
                System.out.println(Arrays.toString(output_layer_outputs));
            }

            predictions[i] = predicted_class;
        }

        return predictions;
    }

    public double[][] getHiddenLayerWeight(){
        return this.hidden_layer_weights;
    }

    public double[][] getOutputLayerWeight(){
        return this.output_layer_weights;
    }

}
