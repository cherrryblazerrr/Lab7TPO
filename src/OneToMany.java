import mpi.*;

class OneToMany implements MatrixMultiplier {
    public void multiply(int n, double[] A, double[] B, double[] C, int rank, int size) throws MPIException {
        int[] counts = new int[size];
        int[] displs = new int[size];
        CompClass.calculateDistributions(n, size, counts, displs);

        int localRows = counts[rank] / n;
        double[] localA = new double[counts[rank]];
        double[] localC = new double[counts[rank]];

        MPI.COMM_WORLD.Scatterv(A, 0, counts, displs, MPI.DOUBLE, localA, 0, counts[rank], MPI.DOUBLE, 0);
        MPI.COMM_WORLD.Bcast(B, 0, n * n, MPI.DOUBLE, 0);

        CompClass.ijkCompute(localA, B, localC, localRows, n);

        if (rank == 0) {
            System.arraycopy(localC, 0, C, 0, counts[0]);
            for (int i = 1; i < size; i++) {
                MPI.COMM_WORLD.Recv(C, displs[i], counts[i], MPI.DOUBLE, i, 100);
            }
        } else {
            MPI.COMM_WORLD.Send(localC, 0, counts[rank], MPI.DOUBLE, 0, 100);
        }
    }
}