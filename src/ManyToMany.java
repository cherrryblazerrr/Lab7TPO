import mpi.*;

class ManyToMany implements MatrixMultiplier {
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

        int[] sendCounts = new int[size];
        int[] sendDispls = new int[size];
        int[] recvCounts = new int[size];
        int[] recvDispls = new int[size];

        for (int i = 0; i < size; i++) {
            sendCounts[i] = counts[rank];
            sendDispls[i] = 0;

            recvCounts[i] = counts[i];
            recvDispls[i] = displs[i];
        }

        MPI.COMM_WORLD.Alltoallv(localC, 0, sendCounts, sendDispls, MPI.DOUBLE,
                C, 0, recvCounts, recvDispls, MPI.DOUBLE);
    }
}