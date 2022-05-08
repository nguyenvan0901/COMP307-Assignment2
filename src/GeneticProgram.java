import org.jgap.InvalidConfigurationException;
import org.jgap.gp.CommandGene;
import org.jgap.gp.GPProblem;
import org.jgap.gp.function.Add;
import org.jgap.gp.function.Divide;
import org.jgap.gp.function.Multiply;
import org.jgap.gp.function.Subtract;
import org.jgap.gp.impl.DeltaGPFitnessEvaluator;
import org.jgap.gp.impl.GPConfiguration;
import org.jgap.gp.impl.GPGenotype;
import org.jgap.gp.terminal.Terminal;
import org.jgap.gp.terminal.Variable;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

/**
 * @author carlos
 *
 */
public class GeneticProgram extends GPProblem {
    @SuppressWarnings("boxing")
    private static Double[] INPUT;

    private static double[] OUTPUT;

    private Variable _xVariable;

    public GeneticProgram() throws InvalidConfigurationException {
        super(new GPConfiguration());
        this.loadData();
        GPConfiguration config = getGPConfiguration();

        _xVariable = Variable.create(config, "X", CommandGene.DoubleClass);
        // DeltaGPFitnessEvaluation gives the lowest fitness value.
        config.setGPFitnessEvaluator(new DeltaGPFitnessEvaluator());
        // graph looks like a parabol (quadratic) therefore I don't use a depth too big.
        config.setMaxInitDepth(6);
        // large population size to keep the variation.
        config.setPopulationSize(1000);

        config.setFitnessFunction(new FitnessFunction(INPUT, OUTPUT, _xVariable));
        config.setStrictProgramCreation(true);
    }

    public void loadData(){
        try{
            File datafile = new File("regression.txt");
            Scanner sc = new Scanner(datafile);
            ArrayList<Double> x = new ArrayList<>();
            ArrayList<Double> y = new ArrayList<>();
            int length = 0;

            sc.nextLine();
            sc.nextLine();

            while (sc.hasNextDouble()) {
                x.add(sc.nextDouble());
                y.add(sc.nextDouble());
                length++;
            }

            INPUT = new Double[length];
            OUTPUT = new double[length];

            for(int i=0; i<x.size(); i++){
                INPUT[i] = x.get(i);
                OUTPUT[i] = y.get(i);
            }

            System.out.println("x: " + Arrays.toString(INPUT));
            System.out.println("y: " + Arrays.toString(OUTPUT));

        }catch(IOException e){
            System.out.println("Something is wrong");
        }
    }

    @Override
    public GPGenotype create() throws InvalidConfigurationException {

        GPConfiguration config = getGPConfiguration();
        config.setCrossoverProb(0.85f);
        config.setMutationProb(0.10f);
        config.setReproductionProb(0.05f);
        System.out.println("mutation probability: " + config.getMutationProb());
        System.out.println("Cross over probabity: " + config.getCrossoverProb());
        System.out.println("Reproduction probabity: " + config.getReproductionProb());

        // The return type of the GP program.
        Class[] types = { CommandGene.DoubleClass };

        // Arguments of result-producing chromosome: none
        Class[][] argTypes = { {} };

        // Define the set of available GP commands and terminals to use.
        CommandGene[][] nodeSets = {
                {
                        _xVariable,
                        new Add(config, CommandGene.DoubleClass),
                        new Multiply(config, CommandGene.DoubleClass),
                        new Subtract(config, CommandGene.DoubleClass),
                        new Divide(config, CommandGene.DoubleClass),
                        new Terminal(config, CommandGene.DoubleClass, 1.0, 10.0, true)
                }
        };

        GPGenotype result = GPGenotype.randomInitialGenotype(config, types, argTypes,
                nodeSets, 20, true);

        return result;
    }

    public static void main(String[] args) throws Exception {
        GPProblem problem = new GeneticProgram();

        GPGenotype gp = problem.create();
        gp.setVerboseOutput(true);

        int iteration = 0;
        int max_iteration = 100;
        double best_fit = 100.0;

        while(best_fit > 0.1 && iteration < max_iteration){
            gp.evolve();

            if(gp.getFittestProgram().getFitnessValue() < best_fit){
                System.out.println("new best fit at iteration " + iteration + ": " + gp.getFittestProgram().getFitnessValue());
                best_fit = gp.getFittestProgram().getFitnessValue();
            }

            iteration ++;
        }

        System.out.println("Best final model:\n ");
        gp.outputSolution(gp.getFittestProgram());
        System.out.println("Final Model: x^4 - 2x^3 + x^2 + 1");

    }

}