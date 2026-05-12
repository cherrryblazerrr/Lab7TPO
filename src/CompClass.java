import mpi.*;

interface MatrixMultiplier {
    void multiply(int n, double[] A, double[] B, double[] C, int rank, int size) throws MPIException;
}

class CompClass {
    public static void ijkCompute(double[] localA, double[] B, double[] localC, int rows, int n) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += localA[i * n + k] * B[k * n + j];
                }
                localC[i * n + j] = sum;
            }
        }
    }

    public static void calculateDistributions(int n, int size, int[] sendCounts, int[] displs) {
        int avgRows = n / size;
        int extraRows = n % size;
        int currentDispl = 0;
        for (int i = 0; i < size; i++) {
            int rowsForProc = (i < extraRows) ? avgRows + 1 : avgRows;
            sendCounts[i] = rowsForProc * n;
            displs[i] = currentDispl;
            currentDispl += sendCounts[i];
        }
    }
}