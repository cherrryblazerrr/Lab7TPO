import mpi.MPI;

import mpi.*;

import java.util.Arrays;

public class Benchmark {
    public static void main(String[] args) throws Exception {
        MPI.Init(args);
        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();

        int[] sizes = {200, 400, 500, 600, 1000, 1200};

        for (int N : sizes) {
            double[] A = new double[N * N];
            double[] B = new double[N * N];
            double[] C = new double[N * N];

            if (rank == 0) {
                fill(A);
                fill(B);
            }

            if (rank == 0) {
                System.out.println("\n==========================================");
                System.out.println("Benchmark for N = " + N);
                System.out.println("MPI Nodes: " + size);
                System.out.println("==========================================");
            }

            long startSeq = System.currentTimeMillis();
            new Sequential().multiply(N, A, B, C, rank, size);
            long timeSeq = System.currentTimeMillis() - startSeq;

            if (rank == 0) {
                System.out.printf("%-22s : %d ms\n", "Sequential", timeSeq);
            }

            MatrixMultiplier[] tests = {
                    new PointToPoint(),
                    new OneToMany(),
                    new ManyToOne(),
                    new ManyToMany()
            };

            for (MatrixMultiplier test : tests) {
                if (rank == 0) Arrays.fill(C, 0);

                MPI.COMM_WORLD.Barrier();
                long startPar = System.currentTimeMillis();

                test.multiply(N, A, B, C, rank, size);

                MPI.COMM_WORLD.Barrier();
                long timePar = System.currentTimeMillis() - startPar;

                if (rank == 0) {
                    String name = test.getClass().getSimpleName();
                    double speedup = (timePar > 0) ? (double) timeSeq / timePar : 0;
                    System.out.printf("%-22s : %d ms | Speedup: %.2fx\n", name, timePar, speedup);
                }
            }
        }
        MPI.Finalize();
    }

    private static void fill(double[] m) {
        for (int i = 0; i < m.length; i++) {
            m[i] = Math.random() * 10;
        }
    }
}