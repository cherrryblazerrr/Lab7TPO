import mpi.*;

class PointToPoint implements MatrixMultiplier {
    public void multiply(int n, double[] A, double[] B, double[] C, int rank, int size) throws MPIException {
        int[] counts = new int[size];
        int[] displs = new int[size];
        CompClass.calculateDistributions(n, size, counts, displs);

        int localRows = counts[rank] / n;
        double[] localA = new double[counts[rank]];
        double[] localC = new double[counts[rank]];
        double[] sharedB = new double[n * n];

        if (rank == 0) {
            System.arraycopy(B, 0, sharedB, 0, n * n);
            System.arraycopy(A, 0, localA, 0, counts[0]);
            for (int i = 1; i < size; i++) {
                MPI.COMM_WORLD.Send(A, displs[i], counts[i], MPI.DOUBLE, i, 10);
                MPI.COMM_WORLD.Send(B, 0, n * n, MPI.DOUBLE, i, 20);
            }
        } else {
            MPI.COMM_WORLD.Recv(localA, 0, counts[rank], MPI.DOUBLE, 0, 10);
            MPI.COMM_WORLD.Recv(sharedB, 0, n * n, MPI.DOUBLE, 0, 20);
        }

        CompClass.ijkCompute(localA, sharedB, localC, localRows, n);

        if (rank == 0) {
            System.arraycopy(localC, 0, C, 0, counts[0]);
            for (int i = 1; i < size; i++) MPI.COMM_WORLD.Recv(C, displs[i], counts[i], MPI.DOUBLE, i, 30);
        } else {
            MPI.COMM_WORLD.Send(localC, 0, counts[rank], MPI.DOUBLE, 0, 30);
        }
    }
}