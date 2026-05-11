class Sequential implements MatrixMultiplier {
    @Override
    public void multiply(int n, double[] A, double[] B, double[] C, int rank, int size) {
        if (rank == 0) {
            for (int i = 0; i < n * n; i++) C[i] = 0;

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double sum = 0;
                    for (int k = 0; k < n; k++) {
                        sum += A[i * n + k] * B[k * n + j];
                    }
                    C[i * n + j] = sum;
                }
            }
        }
    }
}