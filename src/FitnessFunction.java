import org.jgap.gp.GPFitnessFunction;
import org.jgap.gp.IGPProgram;
import org.jgap.gp.terminal.Variable;

public class FitnessFunction extends GPFitnessFunction {

    private Double[] _input;
    private double[] _output;
    private Variable _xVariable;


    private static Object[] NO_ARGS = new Object[0];

    public FitnessFunction(Double input[], double output[], Variable x) {
        _input = input;
        _output = output;
        _xVariable = x;
    }

    @Override
    protected double evaluate(final IGPProgram program) {
        double result = 0.0f;
        for (int i = 0; i < _input.length; i++) {
            // Set the input values
            _xVariable.set(_input[i]);

            // Execute the genetically engineered algorithm
            double value = program.execute_double(0, NO_ARGS);

            // The closer longResult gets to 0 the better the algorithm.
            result += Math.abs(value - _output[i]);
        }

        return result;
    }
}